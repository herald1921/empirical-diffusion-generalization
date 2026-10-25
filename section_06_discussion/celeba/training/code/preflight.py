from pathlib import Path
import time,json,hashlib,copy
import numpy as np,torch
from models import make_model
from core import load_data
from sampling import integrate,schedule,empirical_field
R=Path('/root/figure07_cfm_20260921');cfg=json.loads((R/'protocol/experiment.json').read_text(encoding='utf-8-sig'))
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(0)
groups,ids=load_data();report={};arrays={}
old=Path(cfg['reuse_candidate']);p=torch.load(old,map_location='cpu',weights_only=False)
assert p['format']=='celeba80_cfm_t_uniform_velocity_v2' and p['step']==200000 and p['dataset_info']['group']=='S1' and p['dataset_info']['count']==98023 and p['dataset_info']['decode_divisor']==765.
m=make_model('unet').cuda();m.load_state_dict(p['ema_model'],strict=True);m.eval()
legacy=np.load('/root/shared-nvme/experiment-real-data/runs/celeba_NN_stage_S1_t_uniform/stage2_full_nn_baseline/samples/full_nn_ema_step_200000.npz')
with torch.inference_mode():
 oldnoise=torch.from_numpy(legacy['source_noise'][:4]).cuda()
 def ampfield(x,t):
  with torch.autocast('cuda',dtype=torch.bfloat16):return m(x,t).float()
 legacy_x,_,_=integrate(ampfield,oldnoise,schedule(80,end=1.))
 report['legacy_reproduction']={'max_abs':float(np.max(np.abs(legacy_x.cpu().numpy()-legacy['final'][:4]))),'rmse':float(np.sqrt(np.mean((legacy_x.cpu().numpy()-legacy['final'][:4])**2)))}
 noise=torch.stack([torch.randn(6400,generator=torch.Generator().manual_seed(i)) for i in range(4)]).cuda();arrays['noise']=noise.cpu().numpy()
 x,out,_=integrate(m,noise,schedule(160,captures=cfg['capture_tau']))
 fine,fout,_=integrate(m,noise,schedule(320,captures=cfg['capture_tau']))
 arrays['reused_unet_output']=out.cpu().numpy();arrays['legacy_recomputed']=legacy_x.cpu().numpy()
 report['unet_refinement']={'rmse_each':(out-fout).square().mean(1).sqrt().cpu().tolist(),'finite':bool(torch.isfinite(out).all())}
 for n in [1,100,10000,98023]:
  field=empirical_field(groups[0][:n].cuda());state,eout,_=integrate(field,noise,schedule(160,captures=cfg['capture_tau']))
  arrays[f'empirical_N{n}']=eout.cpu().numpy()
  if n in [1,98023]:
   _,efine,_=integrate(field,noise,schedule(320,captures=cfg['capture_tau']))
   report[f'empirical_N{n}_refinement']={'readout_rmse_each':(eout-efine).square().mean(1).sqrt().cpu().tolist()}
  if n==1:assert torch.allclose(eout,groups[0][0].cuda().expand_as(eout),atol=1e-5)
  del field
 # A direct squared-distance posterior independently checks the empirical velocity/readout.
 ref=groups[0][:10].cuda();query=noise[:2];t=.4
 weights=torch.softmax(-((query[:,None,:]-t*ref[None,:,:]).double().square().sum(2))/(2*(1-t)**2),1)
 direct=weights@ref.double();derived=query+(1-t)*empirical_field(ref)(query,t)
 report['readout_identity_max_abs']=float((direct-derived).abs().max())
 assert report['readout_identity_max_abs']<2e-4
report['reuse']={'allowed_case':'unet/N98023/s1','sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'weights':'ema_model','historical_dataset_hash_not_available':True,'same_declared_dataset_and_partition_path':True}
del m,p;torch.cuda.empty_cache()
print('REUSE_AUDIT',json.dumps(report),flush=True)
# Short fresh MLP pilot; never reused as a final model or a smaller-N warm start.
for n,steps in [(1,10000),(10000,10000)]:
 torch.manual_seed(0);m=make_model('mlp').cuda();opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
 ref=groups[0][:n].cuda();losses=[];tstart=time.time()
 for step in range(steps):
  ix=torch.randint(n,(32,),device='cuda');z=ref[ix];eps=torch.randn_like(z);t=torch.rand(32,device='cuda');x=(1-t[:,None])*eps+t[:,None]*z
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):v=m(x,t);loss=(v.float()-(z-eps)).square().mean()
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss))
  if (step+1)%500==0:print('MLP_PILOT',n,step+1,np.mean(losses[-100:]),flush=True)
 with torch.inference_mode():
  acts=[]
  hook=m.act.register_forward_pre_hook(lambda module,args:acts.append(args[0].detach()))
  z=ref[torch.arange(32,device='cuda')%n];eps=torch.randn_like(z);t=torch.full((32,),.5,device='cuda');x=.5*(z+eps)
  v=m(x,t);pred=x+.5*v;h=acts[-1];sig=h.sigmoid();der=sig*(1+h*(1-sig));hook.remove()
  clean_mse=float((pred-z).square().mean());initial_mse=float((.5*x-z).square().mean())
  _,output,_=integrate(m,noise,schedule(160));arrays[f'mlp_pilot_N{n}']=output.cpu().numpy()
  item={'steps':steps,'parameters':sum(v.numel() for v in m.parameters()),'loss_first100':float(np.mean(losses[:100])),'loss_last100':float(np.mean(losses[-100:])),
    'silu_derivative_fraction_above_1e-4':float((der.abs()>1e-4).float().mean()),'midtime_clean_mse':clean_mse,'initial_model_midtime_mse':initial_mse,
    'time_sensitivity_rms':float((m(x,.2)-m(x,.8)).square().mean().sqrt()),'seconds':time.time()-tstart,'raw_pilot_only_not_ema':True}
  item['passed']=item['loss_last100']<.8*item['loss_first100'] and item['silu_derivative_fraction_above_1e-4']>.9 and clean_mse<initial_mse and item['time_sensitivity_rms']>1e-4
  report[f'mlp_pilot_N{n}']=item;print('PILOT_RESULT',json.dumps(item),flush=True)
 del m,opt,ref;torch.cuda.empty_cache()
np.savez_compressed(R/'qa/audit_arrays.npz',**arrays)
report['passed']=all(report[f'mlp_pilot_N{n}']['passed'] for n in [1,10000]) and report['legacy_reproduction']['rmse']<.02 and max(report['unet_refinement']['rmse_each'])<.02
(R/'qa/preflight.json').write_text(json.dumps(report,indent=2))
print('PREFLIGHT_COMPLETE',report['passed'],flush=True)

