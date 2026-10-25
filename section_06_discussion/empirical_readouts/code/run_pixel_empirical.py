"""Direct 6400-dimensional pixel empirical probability flow: no PCA or projection.
Uses the prepared CelebA80 shards and fixed image-disjoint S1/S2 partitions.
Outputs the empirical posterior mean at each stopping state (Tweedie readout).
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch

SIGMAS=[8.,4.,2.,1.,.5,.2,.1]
SEEDS=[0,1]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_groups(data,device):
    manifest=json.loads((data/'manifest.json').read_text())
    with np.load(data/'paper_partitions.npz',allow_pickle=False) as a:
        ids=[a['group_a_raw_indices'],a['group_b_raw_indices']]
    assert len(ids[0])==len(ids[1])==98023
    assert len(np.intersect1d(*ids))==0
    raw=np.empty((int(manifest['num_raw_images']),6400),dtype=np.uint16)
    cursor=0
    for entry in manifest['shards']:
        with np.load(data/entry['filename'],allow_pickle=False) as a:
            images=a['gray_rgb_sum'].reshape(int(entry['count']),6400)
        raw[cursor:cursor+len(images)]=images
        cursor+=len(images)
    assert cursor==len(raw)
    groups=[torch.from_numpy(np.ascontiguousarray(raw[idx])).to(device=device,dtype=torch.float32)/float(manifest['decode_divisor']) for idx in ids]
    return groups,ids,manifest

@torch.inference_mode()
def posterior(q,ref,norms,sigma):
    # Query squared norm cancels from normalized weights. Full empirical sum.
    logits=(2*(q@ref.T)-norms[None,:])/(2*float(sigma)**2)
    weights=torch.softmax(logits,dim=1)
    return weights@ref,weights

@torch.inference_mode()
def nearest(q,ref,chunk=8192):
    centered=q-q.mean(1,keepdim=True)
    centered=centered/centered.norm(dim=1,keepdim=True).clamp_min(1e-12)
    values=torch.full((len(q),),-torch.inf,device=q.device)
    indices=torch.full((len(q),),-1,device=q.device,dtype=torch.long)
    for start in range(0,len(ref),chunk):
        r=ref[start:start+chunk]
        r=r-r.mean(1,keepdim=True)
        r=r/r.norm(dim=1,keepdim=True).clamp_min(1e-12)
        v,i=(centered@r.T).max(1)
        update=v>values
        indices[update]=i[update]+start
        values=torch.maximum(v,values)
    return values,indices

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('/root/shared-nvme/experiment-real-data/datasets/celeba80_author_order'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=80)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda')
    started=time.time()
    refs,rawids,manifest=load_groups(a.data_dir,device)
    print('Loaded pixel references',flush=True)
    eps=torch.stack([torch.randn(6400,generator=torch.Generator(device='cpu').manual_seed(seed)) for seed in SEEDS]).to(device)
    initial=16*eps
    base=np.linspace(16**(1/7),.1**(1/7),a.steps,dtype=np.float64)**7
    base[0]=16.;base[-1]=.1
    schedule=np.array(sorted(set(base.tolist()+SIGMAS),reverse=True))
    arrays={'seeds':np.array(SEEDS),'sigmas':np.array(SIGMAS),'reverse_t':1/(1+np.array(SIGMAS)),
            'initial_state':initial.cpu().numpy(),'noise':eps.cpu().numpy(),'schedule':schedule}
    checks={}
    for group_index,ref in enumerate(refs):
        g=f's{group_index+1}'
        # Computing squared norms in blocks avoids a full 2.5 GB square temporary.
        norms=torch.cat([(ref[i:i+8192].double().square().sum(1)).float() for i in range(0,len(ref),8192)])
        state=initial.clone();captured={}
        for step,(s0,s1) in enumerate(zip(schedule[:-1],schedule[1:]),1):
            mean,_=posterior(state,ref,norms,s0)
            drift=(state-mean)/s0
            pred=state+(s1-s0)*drift
            mean_pred,_=posterior(pred,ref,norms,s1)
            state=state+.5*(s1-s0)*(drift+(pred-mean_pred)/s1)
            assert torch.isfinite(state).all()
            if any(np.isclose(s1,s,rtol=0,atol=1e-11) for s in SIGMAS): captured[float(s1)]=state.clone()
            if step%20==0: print(g,'step',step,'/',len(schedule)-1,'sigma',s1,flush=True)
        states=torch.stack([captured[s] for s in SIGMAS])
        flat=states.reshape(-1,6400)
        means=[];ess=[];weights_max=[];posterior_idx=[]
        for i,sigma in enumerate(SIGMAS):
            m,w=posterior(states[i],ref,norms,sigma)
            means.append(m);ess.append(1/w.square().sum(1));weights_max.append(w.max(1).values);posterior_idx.append(w.argmax(1))
        outputs=torch.stack(means)
        clipped=outputs.clamp(0,1)
        ccc,idx=nearest(clipped.reshape(-1,6400),ref)
        near=ref[idx].reshape(len(SIGMAS),len(SEEDS),6400)
        arrays.update({f'state_{g}':states.cpu().numpy(),f'output_{g}':clipped.cpu().numpy(),
            f'nearest_{g}':near.cpu().numpy(),f'nearest_index_{g}':idx.reshape(len(SIGMAS),len(SEEDS)).cpu().numpy(),
            f'nearest_raw_index_{g}':rawids[group_index][idx.cpu().numpy()].reshape(len(SIGMAS),len(SEEDS)),
            f'nearest_ccc_{g}':ccc.reshape(len(SIGMAS),len(SEEDS)).cpu().numpy(),
            f'ess_{g}':torch.stack(ess).cpu().numpy(),f'max_weight_{g}':torch.stack(weights_max).cpu().numpy(),
            f'posterior_index_{g}':torch.stack(posterior_idx).cpu().numpy()})
        # Independent float64 posterior with direct differences checks the final readout.
        errors=[]
        for i,sigma in enumerate(SIGMAS):
            q=states[i].double(); logits=[]
            for j in range(0,len(ref),4096):
                rr=ref[j:j+4096].double()
                d2=torch.stack([(rr-v).square().sum(1) for v in q])
                logits.append(-d2/(2*sigma*sigma))
            w64=torch.softmax(torch.cat(logits,dim=1),dim=1)
            mean64=torch.zeros_like(q)
            for j in range(0,len(ref),4096):mean64+=w64[:,j:j+4096]@ref[j:j+4096].double()
            errors.append(float((mean64-outputs[i]).abs().max()))
        checks[g]={'readout_float64_max_abs_errors':errors,'late_max_abs_to_selected_reference':float((clipped[-1]-near[-1]).abs().max())}
        print('Completed',g,checks[g],flush=True)
    np.savez_compressed(a.output_dir/'pixel_trajectories.npz',**arrays)
    metadata=dict(format='celeba80_direct_pixel_empirical_v1',pca=False,projection=False,dimension=6400,N=98023,
        seeds=SEEDS,noise_generator='independent CPU torch.Generator.manual_seed(seed)',
        initialization='one shared initial_state=16*epsilon in original uncentered pixel coordinates',
        output='full empirical posterior mean at the saved VE probability-flow state; no PCA, shrinkage, learned denoiser or extra projection',
        ode='d y / d sigma = (y - D_sigma(y))/sigma',solver='Heun',base_schedule_points=a.steps,
        actual_schedule_points=len(schedule),sigmas=SIGMAS,reverse_time='t=1/(1+sigma)',
        nearest='maximum centered cosine over original pixel reference in the corresponding group',
        data_dir=str(a.data_dir),manifest_sha256=sha(a.data_dir/'manifest.json'),partitions_sha256=sha(a.data_dir/'paper_partitions.npz'),
        image_index_disjoint=True,identity_disjoint_verified=False,elapsed_seconds=time.time()-started,
        torch_version=torch.__version__,numpy_version=np.__version__,gpu=torch.cuda.get_device_name(),
        source_sha256=sha(__file__),checks=checks,arrays_sha256=sha(a.output_dir/'pixel_trajectories.npz'))
    (a.output_dir/'run_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('DONE',metadata['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
