from pathlib import Path
import json,numpy as np,torch
from core import load_data
from sampling import integrate,schedule,empirical_field
R=Path('/root/figure07_cfm_20260921');c=json.loads((R/'protocol/experiment.json').read_text(encoding='utf-8-sig'));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode():
 g,_=load_data();noise=torch.stack([torch.randn(6400,generator=torch.Generator().manual_seed(i)) for i in range(4)]).cuda();f=empirical_field(g[0][:10000].cuda());rows=[]
 _,_,a=integrate(f,noise,schedule(160,captures=c['capture_tau']),c['capture_tau'])
 _,_,b=integrate(f,noise,schedule(320,captures=c['capture_tau']),c['capture_tau'])
 for t in c['capture_tau']:rows.append({'tau':t,'state_rmse':(a[t][0]-b[t][0]).square().mean(1).sqrt().cpu().tolist(),'readout_rmse':(a[t][1]-b[t][1]).square().mean(1).sqrt().cpu().tolist()})
 out={'n':10000,'group':'s1','seeds':[0,1,2,3],'comparison':'160 vs 320 base intervals, same inserted capture times','rows':rows};(R/'qa/earlystop_refinement.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
