"""Reuse the current Figure 5 records; evaluate only missing N=1,10 cases.

All generated files stay in a new directory. Existing experiments are read only.
"""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import torch

BASE = Path('/root/figure07_cfm_20260921')
GAP = Path('/root/figure08_gap_20260922')
OUT = Path('/dev/shm/figure5_redesign_20261017')
sys.path.insert(0, str(GAP/'code'))
from common import trajectory, model, checkpoint, empirical_field, nearest_rmse, setup

NS = [1, 10, 100, 1000, 10000, 98023]
CONDITIONS = ['empirical_stop', 'empirical_end', 'mlp', 'unet']
GROUPS = ['s1', 's2']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def small_training_sets():
    data=Path('/root/shared-nvme/experiment-real-data/datasets/celeba80_author_order')
    manifest=json.loads((data/'manifest.json').read_text())
    with np.load(data/'paper_partitions.npz') as p:
        selected=[p['group_a_raw_indices'][:10], p['group_b_raw_indices'][-10:]]
    result=[]
    for ids in selected:
        values=np.empty((10,6400),np.float32)
        for s in manifest['shards']:
            hit=(ids>=s['start']) & (ids<s['start']+s['count'])
            if hit.any():
                with np.load(data/s['filename']) as z:
                    raw=z['gray_rgb_sum'].reshape(s['count'],6400)
                    values[hit]=raw[ids[hit]-s['start']].astype(np.float32)/np.float32(manifest['decode_divisor'])
        result.append(torch.from_numpy(values).cuda())
    return result,selected

def main():
    setup();OUT.mkdir(exist_ok=True)
    noise=np.load(BASE/'noise.npz')['noise'];noise_hash=sha(BASE/'noise.npz')
    old_ess=np.load(GAP/'ess_archived/ess.npz')
    times=old_ess['tau'];stop=int(np.flatnonzero(np.isclose(times,.15))[0])
    report={'noise_sha256':noise_hash,'conditions':CONDITIONS,'n_values':NS,
            'groups':GROUPS,'seed':0,'statistical_seeds':256,'ess_seeds_per_direction':8,
            'stop_tau':float(times[stop]),'terminal_tau':float(times[-1]),
            'distance':'minimum pixel RMSE on unclipped clean readout',
            'cases':[],'new_generation':'N=1,10 only, same checkpoints/noise and 160-step-plus-32-captures Heun protocol',
            'existing_data_preserved':True}
    arrays={
        'n_values':np.array(NS),'tau':times,'seeds':np.arange(256),
        'images':np.empty((4,6,2,6400),np.float32),
        'nearest_train':np.empty((4,6,2,6400),np.float32),
        'd_train':np.empty((4,6,2,256),np.float32),
        'd_ref':np.empty((4,6,2,256),np.float32),
        'ess':np.empty((6,2,len(times),8),np.float64),
    }
    small,small_ids=small_training_sets()
    for ni,n in enumerate(NS):
        for gi,g in enumerate(GROUPS):
            for method in ['empirical','mlp','unet']:
                tag=f'{method}_N{n}_{g}_baseline200k'
                if n>=100:
                    folder=GAP/'data'/tag
                    metrics=np.load(folder/'metrics.npz');gallery=np.load(folder/'gallery.npz')
                    metadata=json.loads((folder/'complete.json').read_text())
                    assert metadata['noise_sha256']==noise_hash
                    assert metadata['metrics_sha256']==sha(folder/'metrics.npz')
                    assert np.array_equal(metrics['tau'],times)
                    assert np.array_equal(metrics['seeds'],np.arange(256))
                    item={'case':tag,'reused':True,'metrics_sha256':sha(folder/'metrics.npz'),
                          'gallery_sha256':sha(folder/'gallery.npz'),'checkpoint_sha256':metadata['checkpoint_sha256']}
                    if method=='empirical':arrays['ess'][ni,gi]=old_ess[f'N{n}_{g}']
                    for condition,k in ([('empirical_stop',stop),('empirical_end',-1)] if method=='empirical' else [(method,-1)]):
                        ci=CONDITIONS.index(condition)
                        arrays['images'][ci,ni,gi]=gallery['readout'][k,0]
                        arrays['nearest_train'][ci,ni,gi]=gallery['nearest_train'][k,0]
                        arrays['d_train'][ci,ni,gi]=metrics['readout_train'][k]
                        arrays['d_ref'][ci,ni,gi]=metrics['readout_reference'][k]
                else:
                    cached=OUT/(tag+'.npz');meta_path=OUT/(tag+'.json')
                    if cached.exists() and meta_path.exists():
                        fresh=np.load(cached);item=json.loads(meta_path.read_text())
                    else:
                        own=small[gi][:n] if gi==0 else small[gi][-n:]
                        other=small[1-gi][:n] if gi==1 else small[1-gi][-n:]
                        if method=='empirical':field=empirical_field(own);network=None;checkpoint_hash='exact_empirical'
                        else:
                            cp=checkpoint(method,n,g);network=model(method,cp);field=network;checkpoint_hash=sha(cp)
                        result={'d_train':[],'d_ref':[]};initial_equal=True
                        with torch.inference_mode():
                            for start in range(0,256,32):
                                eps=torch.from_numpy(noise[start:start+32]).cuda()
                                states,clean,grid=trajectory(field,eps,160)
                                initial_equal &= torch.equal(states[0],eps)
                                train=[];reference=[]
                                for k in [stop,len(times)-1]:
                                    d,ix=nearest_rmse(clean[k],own);dr,_=nearest_rmse(clean[k],other)
                                    # Exhaustive float64 check is cheap for N<=10.
                                    exact=(clean[k].double()[:,None,:]-own.double()[None,:,:]).square().mean(2).sqrt()
                                    assert torch.equal(ix,exact.argmin(1)) and float((d.double()-exact.min(1).values).abs().max())<1e-7
                                    train.append(d.cpu().numpy());reference.append(dr.cpu().numpy())
                                    if start==0:
                                        j=0 if k==stop else 1
                                        result[f'image_{j}']=clean[k,0].cpu().numpy()
                                        result[f'nearest_{j}']=own[ix[0]].cpu().numpy()
                                result['d_train'].append(np.stack(train));result['d_ref'].append(np.stack(reference))
                                if start==0 and method=='empirical':
                                    ref=own.double();norm=ref.square().sum(1);ess=[]
                                    for k,t in enumerate(times):
                                        x=states[k,:8].double()
                                        logits=(2*t*(x@ref.T)-t*t*norm[None,:])/(2*(1-t)**2)
                                        weights=logits.softmax(1);ess.append((1/weights.square().sum(1)).cpu().numpy())
                                    result['ess']=np.stack(ess)
                        for key in ['d_train','d_ref']:result[key]=np.concatenate(result[key],axis=1)
                        result['tau']=times;result['seeds']=np.arange(256)
                        np.savez_compressed(cached,**result)
                        item={'case':tag,'reused':False,'checkpoint_sha256':checkpoint_hash,'noise_sha256':noise_hash,
                              'initial_state_exactly_equals_saved_noise':bool(initial_equal),'sampling_steps':160,
                              'sampling_grid':grid.tolist(),'metrics_sha256':sha(cached),
                              'nearest_validation':'all candidates directly checked in float64','finished_unix':time.time()}
                        assert initial_equal
                        meta_path.write_text(json.dumps(item,indent=2));fresh=np.load(cached)
                        del field,network,states,clean;torch.cuda.empty_cache()
                    if method=='empirical':arrays['ess'][ni,gi]=fresh['ess']
                    for condition,k in ([('empirical_stop',0),('empirical_end',1)] if method=='empirical' else [(method,1)]):
                        ci=CONDITIONS.index(condition)
                        arrays['images'][ci,ni,gi]=fresh[f'image_{k}']
                        arrays['nearest_train'][ci,ni,gi]=fresh[f'nearest_{k}']
                        arrays['d_train'][ci,ni,gi]=fresh['d_train'][k]
                        arrays['d_ref'][ci,ni,gi]=fresh['d_ref'][k]
                report['cases'].append(item)
                print('READY',tag,flush=True)
    assert np.isfinite(arrays['images']).all()
    assert np.all(arrays['ess']>=1-1e-8)
    for ni,n in enumerate(NS):
        assert np.all(arrays['ess'][ni]<=n*(1+1e-8))
        assert np.allclose(arrays['ess'][ni,:,0],n)
    exact=np.sqrt(((arrays['images'].astype(np.float64)-arrays['nearest_train'].astype(np.float64))**2).mean(-1))
    report['seed0_distance_validation_max_abs_error']=float(abs(exact-arrays['d_train'][...,0]).max())
    assert report['seed0_distance_validation_max_abs_error']<1e-7
    np.savez_compressed(OUT/'figure5_inputs.npz',**arrays)
    report['export_sha256']=sha(OUT/'figure5_inputs.npz')
    (OUT/'provenance.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',report['export_sha256'],flush=True)

if __name__=='__main__':main()
