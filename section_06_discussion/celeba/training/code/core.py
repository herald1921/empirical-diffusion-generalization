from pathlib import Path
import json,numpy as np,torch
DATA=Path('/root/shared-nvme/experiment-real-data/datasets/celeba80_author_order')
def load_data():
    manifest=json.loads((DATA/'manifest.json').read_text())
    with np.load(DATA/'paper_partitions.npz') as p: ids=[p['group_a_raw_indices'],p['group_b_raw_indices']]
    assert len(ids[0])==len(ids[1])==98023 and not np.intersect1d(*ids).size
    raw=np.empty((manifest['num_raw_images'],6400),np.uint16);cursor=0
    for sh in manifest['shards']:
        with np.load(DATA/sh['filename']) as a:v=a['gray_rgb_sum'].reshape(sh['count'],6400)
        raw[cursor:cursor+len(v)]=v;cursor+=len(v)
    groups=[torch.from_numpy(np.ascontiguousarray(raw[i])).float().div_(manifest['decode_divisor']) for i in ids]
    return groups,ids

def active(groups,ids,g,n):
    if g=='s1':return groups[0][:n],ids[0][:n]
    return groups[1][-n:],ids[1][-n:]

def nearest(y,ref):
    q=y-y.mean(1,keepdim=True);q=q/q.norm(dim=1,keepdim=True).clamp_min(1e-12)
    best=torch.full((len(y),),-torch.inf,device='cuda');idx=torch.full((len(y),),-1,device='cuda',dtype=torch.long)
    for b in range(0,len(ref),4096):
        r=ref[b:b+4096];r=r-r.mean(1,keepdim=True);r=r/r.norm(dim=1,keepdim=True).clamp_min(1e-12)
        v,k=(q@r.T).max(1);sel=v>best;idx[sel]=k[sel]+b;best=torch.maximum(v,best)
    return best,idx