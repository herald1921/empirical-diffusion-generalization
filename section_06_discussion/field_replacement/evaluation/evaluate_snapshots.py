"""Exploratory wider-window extension, same paired inputs and model."""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,importlib.util,json,time
import numpy as np
import torch

pa=argparse.ArgumentParser()
pa.add_argument('--steps',type=int,default=200)
pa.add_argument('--seeds',type=int,default=64)
pa.add_argument('--tag',default='main')
cfg=pa.parse_args()
root=Path(__file__).resolve().parent
out=root/cfg.tag;out.mkdir(exist_ok=False)
previous=Path('/root/shared-nvme/section64_switch_scan_20261018/main')
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

methods=[]
for base,alt in [('emp','nn'),('nn','emp')]:
 for w in [.10,.20,.30,.40]:
  for a in sorted(set([.10,.50,round(T-w,2)])):
   methods.append(dict(id=f'{base}_{alt}_pulse_{a:g}_{w:g}',base=base,alternate=alt,start=a,end=round(a+w,2),width=w))
captures=sorted(set([0,T]+[m[k] for m in methods for k in ['start','end']]+[round((m['start']+m['end'])/2,3) for m in methods]))
grid=tuple(t for t in build_grid(cfg.steps,[],captures) if t<=T)
assert len(noise)==16 and cfg.steps==200
snapshots={};differences=[]
with torch.inference_mode():
 x0=noise.to(device);basecaps={}
 for base in ['emp','nn']:
  _,basecaps[base]=integrate_grid(fields[base],x0,grid,captures)
  for t,v in basecaps[base].items():snapshots[f'baseline_{base}__{t:g}']=v.numpy()[:4]
 stored={}
 for w in [.1,.2,.3,.4]:
  folder=Path('/root/shared-nvme/section64_sliding_20261018') if w==.1 else root
  stored[w]=np.load(folder/'main/outputs_000.npz')
 for m in methods:
  a,b=m['start'],m['end'];mid=round((a+b)/2,3)
  def active(x,t):return fields[m['alternate'] if a<=t<b else m['base']](x,t)
  y,cap=integrate_grid(active,basecaps[m['base']][a].to(device),[t for t in grid if t>=a],[a,mid,b,T])
  for t,v in cap.items():snapshots[m['id']+f'__{t:g}']=v.numpy()[:4]
  existing=stored[m['width']][m['id']+'__0.99']
  error=np.sqrt(np.mean((y.cpu().numpy().astype(float)-existing.astype(float))**2,1))
  differences.append(dict(method=m['id'],maximum_rmse=float(error.max())))
  print('snapshot',m['id'],'max replay RMSE',float(error.max()),flush=True)
 # Gallery repeats the same integration; it never applies any clean readout.
np.savez_compressed(out/'states.npz',**snapshots)
provenance.update(dict(methods=methods,grid=grid,noise_hash=ev.tensor_hash(noise),
 snapshot_selection='first four fixed initial noise inputs; 16-input integration batches',
 output='raw integrated states, no posterior-mean/Tweedie readout',replay=differences))
(out/'manifest.json').write_text(json.dumps(provenance,indent=2))
(out/'complete.json').write_text(json.dumps(dict(seconds=time.time()-start,methods=len(methods),
 maximum_replay_rmse=max(d['maximum_rmse'] for d in differences)),indent=2))
