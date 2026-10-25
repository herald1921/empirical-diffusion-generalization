"""Recompute training neighbors at all captured empirical-CFM stopping times (CPU)."""
from pathlib import Path
import sys,json,hashlib,time,os
import numpy as np
import torch
R=Path('/root/figure07_cfm_20260921');sys.path.insert(0,str(R/'code'))
from core import load_data,active

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(4)
    out=R/'data/empirical_earlystop';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    groups,ids=load_data();result={};sources={}
    for n in [1,10,100,1000,10000,98023]:
        path=R/'data/empirical'/f'N{n}'/'samples.npz';sources[str(n)]=sha(path)
        with np.load(path) as samples:
            result['tau']=samples['tau'];result['reverse_t']=samples['reverse_t'];result['seeds']=np.arange(4)
            for group in ['s1','s2']:
                raw=samples['trace_'+group];q=torch.from_numpy(np.clip(raw.reshape(-1,6400),0,1));q=q-q.mean(1,keepdim=True);q=q/q.norm(dim=1,keepdim=True).clamp_min(1e-12)
                ref,raw_ids=active(groups,ids,group,n);best=torch.full((len(q),),-torch.inf);indices=torch.full((len(q),),-1,dtype=torch.long)
                for start in range(0,len(ref),4096):
                    block=ref[start:start+4096];block=block-block.mean(1,keepdim=True);block=block/block.norm(dim=1,keepdim=True).clamp_min(1e-12);v,k=(q@block.T).max(1);win=v>best;indices[win]=k[win]+start;best=torch.maximum(best,v)
                prefix=f'N{n}_{group}_';result[prefix+'readout']=raw;result[prefix+'state']=samples['state_trace_'+group];result[prefix+'nearest']=ref[indices].numpy().reshape(raw.shape);result[prefix+'nearest_raw_index']=raw_ids[indices.numpy()].reshape(raw.shape[:2]);result[prefix+'similarity']=best.numpy().reshape(raw.shape[:2])
        print('EARLYSTOP_NEIGHBORS',n,flush=True)
    np.savez_compressed(out/'samples.npz',**result)
    manifest={'source':'current audited CFM protocol; not legacy finite-noise VE trajectories','n_values':[1,10,100,1000,10000,98023],'samples_per_case':4,'source_seeds':[0,1,2,3],'source_sha256':sources,'sha256':sha(out/'samples.npz'),'neighbor_metric':'centered cosine on clipped one-step readouts; recomputed independently at every stopping time','readout':'x_tau + (1-tau)*v_tau(x_tau); empirical posterior mean','pca':False,'computed_device':'CPU','finished':time.time()}
    (out/'complete.json').write_text(json.dumps(manifest,indent=2))
    print('EMPIRICAL_EARLYSTOP_COMPLETE',flush=True)
if __name__=='__main__':
    with torch.inference_mode():main()
