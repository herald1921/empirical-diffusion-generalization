"""Immutable order/index contract for the paper's existing CelebA dataset."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

DEFAULT_CONTRACT = Path(__file__).resolve().parent / 'paper_contract'

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8 << 20),b''):h.update(b)
    return h.hexdigest()

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def load_contract(directory=DEFAULT_CONTRACT):
    directory=Path(directory)
    c=json.loads((directory/'contract.json').read_text())
    for filename,key in [(c['order_file'],'order_sha256'),(c['removed_file'],'removed_sha256')]:
        if file_sha(directory/filename)!=c[key]:raise ValueError(f'Changed paper contract: {filename}')
    names=(directory/c['order_file']).read_text().splitlines()
    removed=np.load(directory/c['removed_file'],allow_pickle=False)
    if len(names)!=c['raw_count'] or len(set(names))!=len(names):raise ValueError('Invalid image order')
    if removed.dtype.kind not in 'iu' or removed.ndim!=1 or len(np.unique(removed))!=len(removed):raise ValueError('Invalid removed indices')
    if len(removed) and (removed.min()<0 or removed.max()>=len(names)):raise ValueError('Removed index out of range')
    keep=np.setdiff1d(np.arange(c['raw_count'],dtype=np.int64),removed)
    if len(keep)!=c['retained_count'] or len(keep)//2!=c['group_size']:raise ValueError('Inconsistent paper counts')
    for key,values in [('group_a_raw_indices',keep[:c['group_size']]),('group_b_raw_indices',keep[-c['group_size']:])]:
        if array_sha(values)!=c['partition_array_sha256'][key]:raise ValueError(f'Changed partition: {key}')
    return c,names,removed

def ordered_sources(entries,names):
    by_name={Path(x.filename).name:x for x in entries}
    if len(by_name)!=len(entries) or set(by_name)!=set(names):raise ValueError('Archive filenames differ from the released paper order')
    return [by_name[name] for name in names]

def validate_shard(array,record):
    if array.shape!=(record['count'],80,80) or array.dtype!=np.uint16:
        raise ValueError(f"Invalid decoded array: {record['filename']}")
    if array_sha(array)!=record['array_sha256']:
        raise ValueError(f"Pixel content differs from the paper: {record['filename']}")

def verify_dataset(dataset,contract_dir=DEFAULT_CONTRACT):
    dataset=Path(dataset);c,names,removed=load_contract(contract_dir)
    if file_sha(dataset/'raw_image_order.txt')!=c['order_sha256']:raise ValueError('Dataset order mismatch')
    with np.load(dataset/'paper_partitions.npz',allow_pickle=False) as p:
        for key,digest in c['partition_array_sha256'].items():
            if array_sha(p[key])!=digest:raise ValueError(f'Dataset partition mismatch: {key}')
        a,b=p['group_a_raw_indices'],p['group_b_raw_indices']
        if np.intersect1d(a,b).size:raise ValueError('Overlapping image indices')
        stored_subsets=p['paper_subset_sizes'].tolist()
    m=json.loads((dataset/'manifest.json').read_text())
    if m['decode_divisor']!=c['decode_divisor']:raise ValueError('Decode divisor mismatch')
    for record in c['shards']:
        with np.load(dataset/record['filename'],allow_pickle=False) as z:validate_shard(z['gray_rgb_sum'],record)
    return {'verified_shards':len(c['shards']),'retained':c['retained_count'],'group_size':c['group_size'],
            'image_disjoint':True,'identity_disjoint_verified':False,'historical_manifest_unchanged':file_sha(dataset/'manifest.json')==c['historical_manifest_sha256'],
            'stored_subset_metadata':stored_subsets,'canonical_subset_sizes':c['subset_sizes'],
            'note':'Historical metadata is preserved; actual ordered pixel arrays and both group-index arrays are verified.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--contract',type=Path,default=DEFAULT_CONTRACT)
    a=p.parse_args();print(json.dumps(verify_dataset(a.dataset,a.contract),indent=2))
