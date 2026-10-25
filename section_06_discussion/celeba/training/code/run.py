"""Audited CFM rerun. Independent directory, lock and checkpoints; never stops the older run."""
from pathlib import Path
import os,json,time,copy,hashlib,csv,gc,fcntl,traceback,subprocess,sys
import numpy as np,torch
from models import make_model
from core import load_data,active,nearest,DATA
from sampling import schedule,integrate,empirical_field
R=Path('/root/figure07_cfm_20260921');C=R/'code'
def dump(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2)+'\n');os.replace(q,p)
def save(p,obj):
 q=p.with_suffix('.tmp');torch.save(obj,q);os.replace(q,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def append(p,row):
 exists=p.exists()
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(row))
  if not exists:w.writeheader()
  w.writerow(row)
@torch.inference_mode()
def evaluate(m,ref):
 gen=torch.Generator(device='cuda').manual_seed(901);ix=torch.randint(len(ref),(64,),device='cuda',generator=gen);z=ref[ix];eps=torch.randn(z.shape,device='cuda',generator=gen)
 report=[]
 for t in [.1,.3,.5,.8,.95,.99]:
  x=(1-t)*eps+t*z;v=m(x,t);pred=x+(1-t)*v
  report.append({'tau':t,'velocity_mse':float((v-(z-eps)).square().mean()),'clean_mse':float((pred-z).square().mean()),'raw_state_baseline':float((x-z).square().mean()),'rescaled_state_baseline':float((x/t-z).square().mean())})
 return report

def train(kind,n,g,groups,ids,cfg):
 d=R/'runs'/kind/f'N{n}'/g;d.mkdir(parents=True,exist_ok=True)
 if (d/'final.pt').exists():return
 signature=hashlib.sha256((json.dumps(cfg,sort_keys=True)+kind+str(n)+g+sha(C/'models.py')+sha(C/'velocity_unet.py')).encode()).hexdigest()
 if kind=='unet' and n==98023 and g=='s1':
  source=Path(cfg['reuse_candidate']);p=torch.load(source,map_location='cpu',weights_only=False)
  assert sha(source)==json.loads((R/'qa/preflight.json').read_text())['reuse']['sha256']
  save(d/'final.pt',{'model':p['model'],'ema_model':p['ema_model'],'step':p['step'],'signature':signature,'reused_from':str(source),'source_sha256':sha(source)})
  dump(d/'status.json',{'state':'reused_ema_verified','kind':kind,'n':n,'group':g,'step':p['step'],'source_sha256':sha(source)});return
 torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
 m=make_model(kind).cuda();ema=copy.deepcopy(m).eval().requires_grad_(False)
 opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 stream=torch.Generator(device='cuda').manual_seed(20260921)
 ref=active(groups,ids,g,n)[0].cuda();start=0;elapsed=0.
 if (d/'resume.pt').exists():
  p=torch.load(d/'resume.pt',map_location='cpu',weights_only=False);assert p['signature']==signature
  m.load_state_dict(p['model']);ema.load_state_dict(p['ema_model']);opt.load_state_dict(p['optimizer']);stream.set_state(p['stream_state']);start=p['step'];elapsed=p['elapsed'];del p
 dump(d/'config.json',{'kind':kind,'n':n,'group':g,'signature':signature,'parameters':sum(p.numel() for p in m.parameters()),'config':cfg})
 begin=time.time();sumloss=0.;count=0;m.train()
 for step in range(start+1,cfg['steps']+1):
  ix=torch.randint(n,(cfg['batch_size'],),device='cuda',generator=stream);z=ref[ix];eps=torch.randn(z.shape,device='cuda',generator=stream);t=torch.rand(len(z),device='cuda',generator=stream)
  x=(1-t[:,None])*eps+t[:,None]*z;target=z-eps
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=m(x,t);loss=(pred.float()-target).square().mean()
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['gradient_clip'])
  if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradients')
  opt.step()
  with torch.no_grad():
   for ep,p in zip(ema.parameters(),m.parameters()):ep.mul_(cfg['ema_decay']).add_(p,alpha=1-cfg['ema_decay'])
   for eb,b in zip(ema.buffers(),m.buffers()):eb.copy_(b)
  sumloss+=float(loss.detach());count+=1
  if step%1000==0:dump(R/'status.json',{'state':'training','method':kind,'n':n,'group':g,'step':step,'max_steps':cfg['steps'],'seconds':elapsed+time.time()-begin})
  if step%cfg['eval_every']==0:
   validation=evaluate(ema,ref);row={'step':step,'loss':sumloss/count,'ema_clean_mse_mean':float(np.mean([a['clean_mse'] for a in validation])),'gradient_norm_before_clip':float(grad),'seconds':elapsed+time.time()-begin}
   append(d/'training.csv',row);dump(d/'validation.json',{'step':step,'by_tau':validation});print('TRAIN',kind,n,g,json.dumps(row),flush=True);sumloss=0.;count=0
  if step%cfg['checkpoint_every']==0:
   save(d/'resume.pt',{'model':m.state_dict(),'ema_model':ema.state_dict(),'optimizer':opt.state_dict(),'stream_state':stream.get_state(),'step':step,'signature':signature,'elapsed':elapsed+time.time()-begin})
 if kind=='mlp':
  with torch.inference_mode():
   z=ref[:64];x=.5*(z+torch.randn_like(z));h=m.norm(m.in_linear(x));emb=m.time_embed(torch.full((len(x),),.5,device='cuda'));s,b=m.film(emb).chunk(2,1);h=h*(1+s)+b;sig=h.sigmoid();der=sig*(1+h*(1-sig))
   dump(d/'activation_audit.json',{'derivative_fraction_above_1e-4':float((der.abs()>1e-4).float().mean())})
 save(d/'final.pt',{'model':m.state_dict(),'ema_model':ema.state_dict(),'step':cfg['steps'],'signature':signature})
 dump(d/'status.json',{'state':'trained_pending_quality_review','kind':kind,'n':n,'group':g,'step':cfg['steps']})
 # Retain final raw+EMA weights; remove only this completed job's optimizer snapshot.
 (d/'resume.pt').unlink(missing_ok=True)
 del m,ema,opt,ref;gc.collect();torch.cuda.empty_cache()

@torch.inference_mode()
def sample(kind,n,groups,ids,cfg):
 d=R/'data'/kind/f'N{n}';d.mkdir(parents=True,exist_ok=True)
 if (d/'complete.json').exists():return
 noise=np.load(R/'noise.npz')['noise'];grid=schedule(cfg['sampler_points'],cfg['tau_end'],cfg['capture_tau'])
 result={'seeds':np.arange(cfg['samples']),'tau':np.array(cfg['capture_tau']),'reverse_t':2/np.pi*np.arctan(np.array(cfg['capture_tau'])/(1-np.array(cfg['capture_tau']))),'noise_sha256':np.array(sha(R/'noise.npz')),'grid_tau':grid}
 for g in ['s1','s2']:
  ref_cpu,rawids=active(groups,ids,g,n);ref=ref_cpu.cuda();m=None
  if kind=='empirical':field=empirical_field(ref)
  else:
   checkpoint=R/'runs'/kind/f'N{n}'/g/'final.pt';p=torch.load(checkpoint,map_location='cpu',weights_only=False);m=make_model(kind).cuda();m.load_state_dict(p['ema_model']);m.eval();field=m;del p
   result['checkpoint_sha256_'+g]=np.array(sha(checkpoint))
  out=[];trace=None;states=None
  for b in range(0,len(noise),32):
   x,pred,caps=integrate(field,torch.from_numpy(noise[b:b+32]).cuda(),grid,cfg['capture_tau'] if b==0 else ())
   if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite sampling')
   out.append(pred.cpu().numpy())
   if b==0:
    trace=np.stack([caps[t][1][:4].cpu().numpy() for t in cfg['capture_tau']]);states=np.stack([caps[t][0][:4].cpu().numpy() for t in cfg['capture_tau']])
  raw=np.concatenate(out);clipped=np.clip(raw,0,1);values,index=nearest(torch.from_numpy(clipped).cuda(),ref)
  result.update({f'raw_{g}':raw,f'output_{g}':clipped,f'nearest_{g}':ref[index].cpu().numpy(),f'nearest_similarity_{g}':values.cpu().numpy(),f'nearest_raw_index_{g}':rawids[index.cpu().numpy()],f'trace_{g}':trace,f'state_trace_{g}':states,f'clip_fraction_{g}':((raw<0)|(raw>1)).mean(1)})
  del field,m,ref,x,pred;gc.collect();torch.cuda.empty_cache()
 x=result['output_s1'].astype(float);y=result['output_s2'].astype(float);x-=x.mean(1,keepdims=True);y-=y.mean(1,keepdims=True)
 result['pair_similarity']=(x*y).sum(1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)).clip(1e-12)
 np.savez_compressed(d/'samples.npz',**result);dump(d/'complete.json',{'method':kind,'n':n,'samples':cfg['samples'],'sha256':sha(d/'samples.npz'),'protocol_sha256':sha(R/'protocol/experiment.json')})
 print('SAMPLED',kind,n,flush=True)
 subprocess.run([sys.executable,str(C/'plot_results.py'),'--root',str(R)],check=True)

def main():
 R.mkdir(exist_ok=True);lock=(R/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=json.loads((R/'protocol/experiment.json').read_text(encoding='utf-8-sig'));assert json.loads((R/'qa/preflight.json').read_text())['passed'],'Preflight gate failed'
 dump(R/'supervisor.json',{'pid':os.getpid(),'state':'running','started':time.time()})
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if not (R/'noise.npz').exists():np.savez_compressed(R/'noise.npz',noise=np.stack([torch.randn(6400,generator=torch.Generator().manual_seed(i)).numpy() for i in range(cfg['samples'])]))
 dump(R/'data_contract.json',{'manifest_sha256':sha(DATA/'manifest.json'),'partitions_sha256':sha(DATA/'paper_partitions.npz'),'code_sha256':sha(Path(__file__)),'training_code_version':'cfm_linear_v1','source_indices':'S1 first N; S2 last N','image_disjoint':True,'identity_disjoint_verified':False})
 groups,ids=load_data()
 # Materialize the audited existing EMA before scheduling any new long run.
 train('unet',98023,'s1',groups,ids,cfg)
 for n in cfg['n_values']:
  dump(R/'status.json',{'state':'empirical_sampling','n':n});sample('empirical',n,groups,ids,cfg)
 for n in [98023,10000,100,1,10,1000]:
  for kind in ['unet','mlp']:
   for g in ['s1','s2']:train(kind,n,g,groups,ids,cfg)
   dump(R/'status.json',{'state':'sampling','method':kind,'n':n});sample(kind,n,groups,ids,cfg)
 dump(R/'status.json',{'state':'awaiting_quality_review'})
 dump(R/'supervisor.json',{'pid':os.getpid(),'state':'completed_pending_review','finished':time.time()})
if __name__=='__main__':
 try:main()
 except Exception:
  dump(R/'status.json',{'state':'error','traceback':traceback.format_exc()});dump(R/'supervisor.json',{'state':'error','pid':os.getpid()});raise

