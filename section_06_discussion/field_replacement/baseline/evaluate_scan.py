"""Paired permanent-switch scan and duration-matched temporary interventions."""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,importlib.util,json,time
import numpy as np
import torch
import sys
sys.path.insert(0, "/root/figure08_gap_20260922/code")
from nearest_exact import nearest_rmse as exact_pixel_nearest

pa=argparse.ArgumentParser()
pa.add_argument('--steps',type=int,default=200)
pa.add_argument('--seeds',type=int,default=64)
pa.add_argument('--tag',default='main')
cfg=pa.parse_args()
root=Path(__file__).resolve().parent
out=root/cfg.tag;out.mkdir(exist_ok=False)
previous=Path('/root/shared-nvme/section64_visual_20261018/main')
project=Path('/root/shared-nvme/experiment-real-data/projects/celeba_NN_stage')
original=Path('/root/shared-nvme/experiment-real-data/runs/celeba_NN_stage_S1_t_uniform')
spec=importlib.util.spec_from_file_location('eval_windows',project/'eval_windows.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
from celeba_nn_stage.window_experiment import build_grid,integrate_grid
from celeba_nn_stage.sampling import make_fixed_noise
torch.set_num_threads(2)
torch.manual_seed(20260810)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
device=torch.device('cuda')
if torch.cuda.mem_get_info()[0]<8*1024**3:raise RuntimeError('Insufficient free GPU memory')
args=SimpleNamespace(data_dir=Path('/root/shared-nvme/experiment-real-data/datasets/celeba80_author_order'),
 base_repo=Path('/root/shared-nvme/experiment-real-data/projects/celeba80_cloud_bundle-main'),
 checkpoint=original/'checkpoints/checkpoint_step_200000.pt',weights='ema',precision='fp32',
 reference_chunk_size=4096,batch_size=16)
start=time.time()
s1,s2,nn,emp,shape,provenance=ev.real_fields(args,device)
oldmanifest=json.loads((previous/'manifest.json').read_text())
for k in ['checkpoint_sha256','core_sha256']:assert provenance[k]==oldmanifest[k]
assert ev.tensor_hash(s1)==oldmanifest['s1_sha256']
assert ev.tensor_hash(s2)==oldmanifest['s2_sha256']
noise=make_fixed_noise(cfg.seeds,6400,20260810)
with np.load(original/'window_interventions/full_20260906_160410/samples/full_nn.npz') as z:
 assert np.array_equal(noise.numpy(),z['source_noise'][:cfg.seeds])
fields={'nn':nn,'emp':lambda x,t:emp(x,t)[0]}
T=.99
switches=[round(i*.05,2) for i in range(1,20)]+[.97,.98]
methods=[dict(id='full_nn',kind='baseline',base='nn'),dict(id='full_emp',kind='baseline',base='emp')]
for base,alt in [('emp','nn'),('nn','emp')]:
 for c in switches:methods.append(dict(id=f'{base}_to_{alt}_{c:g}',kind='switch',base=base,alternate=alt,start=c,end=T))
 for w in [.05,.10]:
  starts=sorted({round(i*.1,2) for i in range(10) if round(i*.1+w,2)<=T}|{round(T-w,2)})
  for a in starts:methods.append(dict(id=f'{base}_{alt}_pulse_{a:g}_{w:g}',kind='pulse',base=base,alternate=alt,start=a,end=round(a+w,2),width=w))
assert len({m['id'] for m in methods})==len(methods)
diagnostic_times=sorted(set([0,.02]+switches+[T]))
captures=sorted(set(diagnostic_times+[m['start'] for m in methods[2:]]+[m['end'] for m in methods[2:]]))
grid=tuple(t for t in build_grid(cfg.steps,[],captures) if t<=T)
prefix_matches=list(grid)==[t for t in oldmanifest['grid'] if t<=T]
provenance.update(dict(methods=methods,grid=grid,nominal_steps_per_unit=cfg.steps,num_seeds=cfg.seeds,
 noise_seed=20260810,noise_hash=ev.tensor_hash(noise),endpoint=T,
 dataset_sizes=[len(s1),len(s2)],s1_sha256=oldmanifest['s1_sha256'],s2_sha256=oldmanifest['s2_sha256'],
 source_sha256=ev.file_hash(Path(__file__)),readout='unclipped integrated X_.99; no terminal denoising',
 diagnostic_times=diagnostic_times,previous_grid_prefix_identical=prefix_matches,
 gallery_selection='first four inputs fixed before evaluating scan',
 nearest='full-reference exhaustive float64 direct Euclidean distances; no candidate truncation'))
(out/'manifest.json').write_text(json.dumps(provenance,indent=2))
oldrows={(v['method'],int(v['seed'])):v for v in csv.DictReader((previous/'metrics.csv').open()) if float(v['tau'])==T}
metrics=[];diagnostics=[];gallery={};reuse=[]

@torch.inference_mode()
def nearest(q,ref):
 q=q.to(device)
 val,ix=exact_pixel_nearest(q,ref.to(device))
 return val.cpu().numpy(),ix.cpu().numpy(),ref[ix.cpu()].numpy()

def save_csv(path,rows):
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)

@torch.inference_mode()
def run():
 for b in range(0,cfg.seeds,16):
  n=min(16,cfg.seeds-b);x0=noise[b:b+n].to(device);basecaps={};outputs={}
  for base in ['emp','nn']:
   _,caps=integrate_grid(fields[base],x0,grid,captures);basecaps[base]=caps;outputs['full_'+base]=caps[T]
   for t in diagnostic_times:
    x=caps[t].to(device);_,post=emp(x,t)
    for j in range(n):diagnostics.append(dict(base=base,seed=b+j,tau=t,ess=float(post['ess'][j]),top_probability=float(post['top_probability'][j])))
  oldpath=previous/f'outputs_{b:03d}.npz'
  old=np.load(oldpath) if oldpath.exists() else None
  can_reuse=bool(prefix_matches and old is not None and all(np.array_equal(outputs['full_'+base].numpy(),old[f'full_{base}__0.99'][:n]) for base in ['emp','nn']))
  reuse.append(dict(first_seed=b,count=n,baseline_bitwise_equal=can_reuse))
  for m in methods[2:]:
   oldkey=m['id']+'__0.99'
   if can_reuse and oldkey in old:
    outputs[m['id']]=torch.from_numpy(old[oldkey][:n].copy());continue
   a,e=m['start'],m['end']
   def active(x,t):return fields[m['alternate'] if a<=t<e else m['base']](x,t)
   y,_=integrate_grid(active,basecaps[m['base']][a].to(device),[t for t in grid if t>=a])
   outputs[m['id']]=y.cpu()
  for m in methods:
   mid=m['id'];y=outputs[mid]
   existing=can_reuse and mid+'__0.99' in old
   if existing:
    q=[oldrows[mid,b+j] for j in range(n)]
    d1=np.array([float(v['train']) for v in q]);d2=np.array([float(v['reference']) for v in q])
    i1=np.array([int(v['train_index']) for v in q]);i2=np.array([int(v['reference_index']) for v in q])
    im1=s1[i1].numpy();im2=s2[i2].numpy()
   else:d1,i1,im1=nearest(y,s1);d2,i2,im2=nearest(y,s2)
   dn=(y.double()-outputs['full_nn'].double()).square().mean(1).sqrt().numpy()
   de=(y.double()-outputs['full_emp'].double()).square().mean(1).sqrt().numpy()
   for j in range(n):metrics.append(dict(method=mid,seed=b+j,tau=T,train=d1[j],reference=d2[j],train_index=int(i1[j]),reference_index=int(i2[j]),change_from_nn=dn[j],change_from_emp=de[j],reused=existing))
   if b==0:
    for key,arr in [('output',y.numpy()),('train',im1),('reference',im2)]:gallery[mid+'__0.99__'+key]=arr[:4]
  np.savez_compressed(out/f'outputs_{b:03d}.npz',**{m+'__0.99':v.numpy() for m,v in outputs.items()})
  np.savez_compressed(out/'gallery.npz',**gallery)
  save_csv(out/'metrics.csv',metrics);save_csv(out/'diagnostics.csv',diagnostics)
  (out/'status.json').write_text(json.dumps(dict(completed_seeds=b+n,seconds=time.time()-start,reuse=reuse)))
  print(cfg.tag,'seeds',b,b+n,'methods',len(methods),'seconds',round(time.time()-start,1),'reuse',can_reuse,flush=True)
  if old is not None:old.close()
 (out/'complete.json').write_text(json.dumps(dict(seconds=time.time()-start,rows=len(metrics),diagnostic_rows=len(diagnostics),reuse=reuse),indent=2))
run()
