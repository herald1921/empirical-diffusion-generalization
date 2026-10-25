"""Evaluate posterior-weight ESS at archived raw empirical CFM states; no retraining."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
sys.path.insert(0,'/root/figure08_gap_20260922/code')
from common import load_data,active,setup
R=Path('/root/figure08_gap_20260922'); O=R/'ess_archived';O.mkdir(exist_ok=True)
setup();groups,ids=load_data();result={};manifest={}
with torch.inference_mode():
 for n in [100,1000,10000,98023]:
  for g in ['s1','s2']:
   p=R/'data'/f'empirical_N{n}_{g}_baseline200k'
   z=np.load(p/'gallery.npz');m=np.load(p/'metrics.npz');ref=active(groups,ids,g,n)[0].cuda().double();norm=ref.square().sum(1)
   vals=[];errors=[]
   for k,tau in enumerate(m['tau']):
    x=torch.from_numpy(z['state'][k]).cuda().double()
    logits=(2*tau*(x@ref.T)-tau*tau*norm[None,:])/(2*(1-tau)**2)
    w=logits.softmax(1);ess=1/w.square().sum(1);y=w@ref
    vals.append(ess.cpu().numpy());errors.append(float((y-torch.from_numpy(z['readout'][k]).cuda()).square().mean(1).sqrt().max()))
   vals=np.array(vals);assert np.isfinite(vals).all() and vals.min()>=1-1e-9 and vals.max()<=n*(1+1e-9)
   assert np.allclose(vals[0],n)
   result[f'N{n}_{g}']=vals
   manifest[f'N{n}_{g}']={'gallery_sha256':hashlib.sha256((p/'gallery.npz').read_bytes()).hexdigest(),'posterior_vs_archived_readout_rmse_max':max(errors),'samples':8}
   print(n,g,'ESS_DONE',float(vals[-1].mean()),max(errors),flush=True)
   del ref,norm,x,w,logits,y;torch.cuda.empty_cache()
 result['tau']=m['tau'];result['reverse_t']=m['reverse_t'];result['seeds']=np.arange(8)
 np.savez_compressed(O/'ess.npz',**result)
 (O/'manifest.json').write_text(json.dumps({'cases':manifest,'formula':'1/sum_i w_i(x_tau,tau)^2; w=softmax((2*tau*x dot z_i-tau^2*norm(z_i)^2)/(2*(1-tau)^2))','state':'archived raw state, before one-step readout','dtype':'float64','no_new_trajectories':True},indent=2))
