from pathlib import Path
import sys,json,os,hashlib,gc
import numpy as np,torch
BASE=Path('/root/figure07_cfm_20260921');ROOT=Path('/root/figure08_gap_20260922');sys.path.insert(0,str(BASE/'code'))
from core import load_data,active
from models import make_model
from sampling import schedule,empirical_field
from nearest_exact import nearest_rmse
CFG=json.loads((ROOT/'protocol/followup_experiment.json').read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(o,indent=2)+'\n');os.replace(q,p)
def status(**kw):dump(ROOT/'status.json',kw)
def setup():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ROOT.mkdir(exist_ok=True)
def checkpoint(kind,n,g):return BASE/'runs'/kind/f'N{n}'/g/'final.pt'
def model(kind,p):
 ck=torch.load(p,map_location='cpu',weights_only=False);m=make_model(kind).cuda().eval();m.load_state_dict(ck['ema_model']);return m
@torch.inference_mode()
def trajectory(field,noise,steps=160):
 captures=CFG['gap']['tau'];grid=schedule(steps,.999,captures);x=noise.clone();states={0.:x.clone()};outs={0.:x+field(x,0.)}
 for t0,t1 in zip(grid[:-1],grid[1:]):
  v=field(x,float(t0));xp=x+(t1-t0)*v;x=x+.5*(t1-t0)*(v+field(xp,float(t1)))
  if t1 in captures:states[float(t1)]=x.clone();outs[float(t1)]=x+(1-t1)*field(x,float(t1))
 return torch.stack([states[t] for t in captures]),torch.stack([outs[t] for t in captures]),grid
@torch.inference_mode()
def calibration(train,reference,validation):
 queries={'training_copy':train[:32],'heldout_validation':validation[:32],'training_mean':train.mean(0,keepdim=True)};report={}
 for name,q in queries.items():
  a,ia=nearest_rmse(q,train);b,ib=nearest_rmse(q,reference)
  report[name]={'d_train':a.cpu().tolist(),'d_reference':b.cpu().tolist(),'gap':(b-a).cpu().tolist()}
  if name=='training_copy':assert a.max()<1e-7
 q=validation[:2];small=train[:57];v,ix=nearest_rmse(q,small,17);direct=((q.double()[:,None,:]-small.double()[None,:,:]).square().mean(2)).sqrt();exact,expected=direct.min(1)
 assert torch.allclose(v.double(),exact,atol=1e-7) and torch.equal(ix,expected)
 reversed_v,_=nearest_rmse(q,small.flip(0),19);assert torch.allclose(v,reversed_v,atol=1e-7)
 report['direct_float64_max_error']=float((v-exact).abs().max());return report
@torch.inference_mode()
def validation_errors(m,train,val):
 out={};noise_seed=CFG['adequacy']['validation_noise_seed'];taus=CFG['adequacy']['validation_tau'];size=min(2048,len(train))
 rng=np.random.default_rng(20260922);ix=rng.choice(len(train),size,replace=False);datasets={'training_subset':train[torch.as_tensor(ix,device=train.device)],'heldout_validation':val}
 for name,data in datasets.items():
  rows=[]
  for t in taus:
   gen=torch.Generator(device='cuda').manual_seed(noise_seed);sl=sc=sb=0.;num=0
   for rep in range(4):
    for b in range(0,len(data),32):
     z=data[b:b+32];eps=torch.randn(z.shape,device=z.device,generator=gen);x=(1-t)*eps+t*z;v=m(x,t);y=x+(1-t)*v
     sl+=float((v-(z-eps)).square().sum());sc+=float((y-z).square().sum());sb+=float((x-z).square().sum());num+=z.numel()
   rows.append({'tau':t,'velocity_mse':sl/num,'clean_mse':sc/num,'raw_baseline_mse':sb/num,'images':len(data),'noise_repeats':4})
  out[name]=rows
 return out
