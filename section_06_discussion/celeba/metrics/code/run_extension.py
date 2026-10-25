from common import *
from evaluate_gap import run_case
import time,copy,fcntl,traceback,shutil,subprocess
setup();lock=(ROOT/'extension.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def es(**kw):dump(ROOT/'extension_status.json',kw)
def save(p,obj):
 if shutil.disk_usage(ROOT).free<3_000_000_000:raise RuntimeError('Insufficient space for atomic optimizer checkpoint; existing training was not stopped')
 q=p.with_suffix('.tmp');torch.save(obj,q);os.replace(q,p)
try:
 es(state='waiting_for_independent_validation',pid=os.getpid());groups,ids=load_data()
 for g in ['s1','s2']:
  d=ROOT/'extension'/g;d.mkdir(parents=True,exist_ok=True)
  if (d/'complete.json').exists() and json.loads((d/'complete.json').read_text()).get('state')=='extension_budget_completed_pending_review':continue
  basecase=ROOT/'data'/f'mlp_N10000_{g}_baseline200k';ucase=ROOT/'data'/f'unet_N10000_{g}_baseline200k'
  while not all((c/'complete.json').exists() for c in [basecase,ucase]):es(state='waiting_for_independent_validation',group=g,pid=os.getpid());time.sleep(30)
  bv=json.loads((basecase/'validation.json').read_text());uv=json.loads((ucase/'validation.json').read_text());br=np.mean([v['clean_mse'] for v in bv['heldout_validation']]);ur=np.mean([v['clean_mse'] for v in uv['heldout_validation']]);refine=json.loads((basecase/'refinement.json').read_text())
  # This is a budget allocation trigger, never a convergence/quality proof.
  if refine['requires_review']:
   es(state='sampler_review_required_before_extension',group=g,refinement=refine,pid=os.getpid());raise RuntimeError('Sampler refinement must be resolved before extending MLP')
  late_b=np.mean([v['clean_mse'] for v in bv['heldout_validation'] if v['tau'] in [.8,.95]])
  late_u=np.mean([v['clean_mse'] for v in uv['heldout_validation'] if v['tau'] in [.8,.95]])
  if br<=1.15*ur and late_b<=1.5*late_u:
   dump(d/'complete.json',{'state':'extension_not_triggered','reason':'validation within 15 percent of UNet; visual review still required','mlp_clean_mse':br,'unet_clean_mse':ur});continue
  if (d/'complete.json').exists():
   previous=json.loads((d/'complete.json').read_text());dump(d/'initial_allocation_decision.json',previous);dump(d/'complete.json',{'state':'allocation_revised_training_pending','reason':'late-time errors and fixed-source visual artifacts despite small aggregate error difference'})
  source=checkpoint('mlp',10000,g);cp=torch.load(source,map_location='cpu',weights_only=False);m=make_model('mlp').cuda();m.load_state_dict(cp['model']);ema=copy.deepcopy(m).eval().requires_grad_(False);ema.load_state_dict(cp['ema_model']);del cp
  opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=0.);gen=torch.Generator(device='cuda').manual_seed(20260922);start=200000;ref=active(groups,ids,g,10000)[0].cuda()
  if (d/'resume.pt').exists():
   cp=torch.load(d/'resume.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['model']);ema.load_state_dict(cp['ema_model']);opt.load_state_dict(cp['optimizer']);gen.set_state(cp['stream_state']);start=cp['step'];del cp
  dump(d/'config.json',{'source_sha256':sha(source),'initial_step':200000,'max_step':300000,'optimizer_reset':True,'stream_seed':20260922,'lr':.0002,'batch':32,'ema':.9999,'allocation_trigger':'aggregate validation ratio >1.15 OR late-time tau .8/.95 ratio >1.5; operational exploratory allocation, revised after qualitative audit; numerical gate <=.01','late_mlp_clean_mse':late_b,'late_unet_clean_mse':late_u,'baseline_validation_mlp':br,'baseline_validation_unet':ur,'plateau_is_not_quality_acceptance':True})
  rolling=[];begin=time.time();m.train()
  for step in range(start+1,300001):
   z=ref[torch.randint(len(ref),(32,),device='cuda',generator=gen)];eps=torch.randn(z.shape,device='cuda',generator=gen);t=torch.rand(32,device='cuda',generator=gen);x=(1-t[:,None])*eps+t[:,None]*z
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):v=m(x,t);loss=(v.float()-(z-eps)).square().mean()
   assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(grad);opt.step()
   with torch.no_grad():
    for ep,p in zip(ema.parameters(),m.parameters()):ep.mul_(.9999).add_(p,alpha=.0001)
   rolling.append(float(loss))
   if step%1000==0:es(state='training_extension',group=g,step=step,max_steps=300000,optimizer_reset=True,pid=os.getpid(),seconds=time.time()-begin)
   if step%10000==0:print('EXTENSION',g,step,float(np.mean(rolling)),flush=True);rolling=[]
   if step%25000==0:save(d/'resume.pt',{'model':m.state_dict(),'ema_model':ema.state_dict(),'optimizer':opt.state_dict(),'stream_state':gen.get_state(),'step':step,'optimizer_reset_at':200000})
   if step%50000==0:
    stage=d/f'ema_step_{step}.pt';torch.save({'ema_model':ema.state_dict(),'step':step,'optimizer_reset_at':200000},stage);dump(d/f'ema_step_{step}.json',{'sha256':sha(stage),'stage':step,'kind':'EMA only','archive_after_evaluation':True})
    run_case('mlp',10000,g,groups,ids,tag=f'extension{step}',path=stage);subprocess.run([sys.executable,str(ROOT/'code/plot_results.py'),'--root',str(ROOT)],check=True)
  save(d/'final.pt',{'model':m.state_dict(),'ema_model':ema.state_dict(),'step':300000,'optimizer_reset_at':200000});(d/'resume.pt').unlink(missing_ok=True);dump(d/'complete.json',{'state':'extension_budget_completed_pending_review','step':300000,'optimizer_reset':True,'final_sha256':sha(d/'final.pt')});del m,ema,opt,ref;gc.collect();torch.cuda.empty_cache()
 es(state='extension_work_complete_pending_review',pid=os.getpid(),finished=time.time())
except Exception:
 es(state='error',pid=os.getpid(),traceback=traceback.format_exc());raise

