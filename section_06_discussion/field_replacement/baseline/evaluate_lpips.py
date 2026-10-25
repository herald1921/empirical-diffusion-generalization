"""Full-training-reference LPIPS search for paired raw tau=.99 outputs."""
from pathlib import Path
import csv, importlib.util, json, os, sys, time
import numpy as np
root=Path(__file__).resolve().parent
dependency_root=Path('/root/shared-nvme/section64_visual_20261018')
sys.path.insert(0,str(dependency_root/'deps'))
deadline=time.time()+10800
while not all((root/p).exists() for p in ['main/complete.json','refinement/complete.json']):
    if time.time()>deadline: raise TimeoutError('Prerequisite run not completed')
    time.sleep(15)
import torch, lpips
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
device=torch.device('cuda')
out=root/'perceptual';out.mkdir(exist_ok=False)
start=time.time()
model=lpips.LPIPS(net='alex',version='0.1').to(device).eval()
manifest=json.loads((root/'main/manifest.json').read_text())
methods=[m['id'] for m in manifest['methods']]
parts={m:[] for m in methods}
for p in sorted((root/'main').glob('outputs_*.npz')):
    with np.load(p) as z:
        for m in methods: parts[m].append(z[m+'__0.99'])
outputs={m:np.concatenate(parts[m]) for m in methods}
n=len(outputs[methods[0]])
query=np.concatenate([outputs[m] for m in methods])

@torch.inference_mode()
def prepare(x):
    return x.to(device).reshape(-1,1,80,80).clamp(0,1).repeat(1,3,1,1)*2-1

@torch.inference_mode()
def embedding(x):
    feats=model.net.forward(model.scaling_layer(prepare(x)))
    result=[]
    for k,f in enumerate(feats):
        f=lpips.normalize_tensor(f)
        w=model.lins[k].model[-1].weight.reshape(1,-1,1,1)
        assert (w>=0).all(), 'Nonnegative LPIPS calibration required for Euclidean embedding'
        result.append((f*(w/(f.shape[2]*f.shape[3])).sqrt()).flatten(1))
    return torch.cat(result,1)

q=torch.cat([embedding(torch.from_numpy(query[i:i+64])) for i in range(0,len(query),64)])
with torch.inference_mode():
    exact=model(prepare(torch.from_numpy(query[:16])),prepare(torch.from_numpy(query[n:n+16]))).flatten()
    embedded=(q[:16].double()-q[n:n+16].double()).square().sum(1)
    embedding_error=float((exact-embedded).abs().max())
    assert embedding_error<2e-6, embedding_error
print('query embeddings ready',q.shape,'library check',embedding_error,flush=True)
project=Path('/root/shared-nvme/experiment-real-data/projects/celeba_NN_stage')
sys.path.insert(0,str(project/'src'))
from celeba_nn_stage.data import load_full_groups_cpu
s1,s2,_,_=load_full_groups_cpu(data_dir=Path('/root/shared-nvme/experiment-real-data/datasets/celeba80_author_order'),
    base_repo=Path('/root/shared-nvme/experiment-real-data/projects/celeba80_cloud_bundle-main'))
del s2
best=torch.full((len(q),16),float('inf'),device=device)
ids=torch.zeros((len(q),16),dtype=torch.long,device=device)
qnorm=q.square().sum(1)
with torch.inference_mode():
    for i in range(0,len(s1),128):
        r=embedding(s1[i:i+128])
        d=qnorm[:,None]+r.square().sum(1)[None,:]-2*(q@r.T)
        val,idx=d.topk(min(16,len(r)),largest=False,dim=1)
        merged=torch.cat([best,val],1); indices=torch.cat([ids,idx+i],1)
        best,sel=merged.topk(16,largest=False,dim=1);ids=indices.gather(1,sel)
        if i%8192==0: print('references',i,'/',len(s1),'seconds',round(time.time()-start,1),flush=True)
    # Rerank the global search candidates with direct float64 feature differences.
    flat_ids=ids.cpu().flatten()
    qids=torch.arange(len(q),device=device).repeat_interleave(16)
    direct=[]
    for i in range(0,len(flat_ids),128):
        f=embedding(s1[flat_ids[i:i+128]])
        direct.append((f.double()-q[qids[i:i+128]].double()).square().sum(1).cpu())
    direct=torch.cat(direct).reshape(len(q),16)
    distances,j=direct.min(1)
    nearest_ids=ids.cpu()[torch.arange(len(q)),j]
    empirical=q[methods.index('full_emp')*n:(methods.index('full_emp')+1)*n]
    neural=q[methods.index('full_nn')*n:(methods.index('full_nn')+1)*n]
    rows=[];gallery={}
    for k,m in enumerate(methods):
        f=q[k*n:(k+1)*n]
        de=(f.double()-empirical.double()).square().sum(1).cpu().numpy()
        dn=(f.double()-neural.double()).square().sum(1).cpu().numpy()
        for seed in range(n):
            i=k*n+seed
            rows.append(dict(method=m,seed=seed,tau=.99,lpips_train=float(distances[i]),
                lpips_train_index=int(nearest_ids[i]),lpips_from_emp=de[seed],lpips_from_nn=dn[seed]))
        gallery[m+'__nearest_lpips']=s1[nearest_ids[k*n:k*n+4]].numpy()
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'gallery.npz',**gallery)
    # Compare direct library evaluation at selected nearest neighbours.
    ix=np.linspace(0,len(query)-1,32,dtype=int)
    check=model(prepare(torch.from_numpy(query[ix])),prepare(s1[nearest_ids[ix]])).flatten().cpu().double()
    nearest_library_error=float((check-distances[ix]).abs().max())
    assert nearest_library_error<2e-6, nearest_library_error
(out/'complete.json').write_text(json.dumps(dict(seconds=time.time()-start,rows=len(rows),
    metric='LPIPS AlexNet, package 0.1.4, metric version 0.1',
    preprocessing='clip raw X_0.99 to [0,1], replicate grayscale to RGB, scale to [-1,1], native 80x80 resolution',
    search='all 98,023 training images; float32 global top-16 feature distances, float64 direct reranking',
    feature_dimension=q.shape[1],embedding_library_max_error=embedding_error,
    nearest_library_max_error=nearest_library_error),indent=2))
print('perceptual search complete',flush=True)
