"""Common-input probes. Existing checkpoints are read only; no PCA."""
from pathlib import Path
import sys,json,time,gc,hashlib
import numpy as np,torch
import torch.nn.functional as F
R=Path('/root/figure10_mechanisms_20260923');sys.path.insert(0,'/root/figure08_gap_20260922/code')
from common import load_data,active,model,checkpoint,nearest_rmse,setup,dump
from sampling import empirical_field,integrate,schedule
setup();torch.manual_seed(20260923);groups,ids=load_data();O=R/'data';O.mkdir(parents=True,exist_ok=True)
taus=[.02,.1,.2,.3,.5,.8,.95];report={'definition':'same raw query and same tau; readout=x+(1-tau)*v; no time relabelling inside readout','fields':{},'jacobian':{},'patches':{},'checkpoint_hashes':{}}
@torch.inference_mode()
def post(x,t,ref,temp=1.):
    logits=(2*t*x@ref.T-t*t*ref.square().sum(1)[None,:])/(2*(1-t)**2*temp);w=logits.softmax(1)
    return w@ref,1/w.square().sum(1)
def rms(a,b):return float((a-b).square().mean().sqrt())
noise=torch.from_numpy(np.load('/root/figure07_cfm_20260921/noise.npz')['noise'][:32]).cuda()
gallery={}
for n in [100,1000,10000,98023]:
    ref=groups[0][:n].cuda();mu=ref.mean(0);u=model('unet',checkpoint('unet',n,'s1'));m=model('mlp',checkpoint('mlp',n,'s1'))
    for kind in ['unet','mlp']:report['checkpoint_hashes'][f'{kind}_{n}']=hashlib.sha256(checkpoint(kind,n,'s1').read_bytes()).hexdigest()
    rows=[]
    with torch.inference_mode():
        # S2 probes are never training examples of S1. Same images and noise for all N.
        z=groups[1][-32:].cuda()
        for t in taus:
            x=(1-t)*noise+t*z;yu=x+(1-t)*u(x,t);ym=x+(1-t)*m(x,t);ye,ess=post(x,t,ref)
            row={'tau':t,'mlp_to_unet':rms(ym,yu),'mlp_to_mean':rms(ym,mu),'unet_to_mean':rms(yu,mu),'mlp_to_empirical':rms(ym,ye),'unet_to_empirical':rms(yu,ye),'empirical_ess_median':float(ess.median()),'mlp_clean_mse':float((ym-z).square().mean()),'unet_clean_mse':float((yu-z).square().mean()),'empirical_clean_mse':float((ye-z).square().mean()),'temperature':{}}
            for temp in [4,16,64]:
                yt,et=post(x,t,ref,temp);row['temperature'][str(temp)]={'to_unet':rms(yt,yu),'to_mlp':rms(yt,ym),'clean_mse':float((yt-z).square().mean()),'ess_median':float(et.median())}
            rows.append(row)
        # Means and exact empirical high-noise identity at tau=0.
        y0=noise+m(noise,0.);gallery[f'mlp_N{n}_tau0']=y0[:4].cpu().numpy();gallery[f'mean_N{n}']=mu.cpu().numpy()
        a=np.load(f'/root/figure07_cfm_20260921/data/mlp/N{n}/samples.npz');out=torch.from_numpy(a['raw_s1']).cuda()
        realvar=float(ref.var(0,unbiased=False).mean());report['fields'][str(n)]={'rows':rows,'tau0_mean_rmse':rms(y0,mu),'output_to_mean_rmse':rms(out,mu),'output_variance_over_train':float(out.var(0,unbiased=False).mean())/realvar,'train_variance':realvar}
        # Same-input small perturbations distinguish shared mean output from shared field.
        t=.3;x=(1-t)*noise+t*z;delta=.001*torch.randn_like(x)
        j_u=((x+delta+(1-t)*u(x+delta,t))-(x-delta+(1-t)*u(x-delta,t)))/.002
        j_m=((x+delta+(1-t)*m(x+delta,t))-(x-delta+(1-t)*m(x-delta,t)))/.002
        report['fields'][str(n)]['directional_response']={'unet_rms':rms(j_u,torch.zeros_like(j_u)),'mlp_rms':rms(j_m,torch.zeros_like(j_m)),'difference_rms':rms(j_u,j_m),'cosine_mean':float(F.cosine_similarity(j_u,j_m,dim=1).mean())}
    if n==98023:
        # Actual denoiser Jacobian rows for two fixed sources at three times/four pixels.
        yy,xx=torch.meshgrid(torch.arange(80,device='cuda'),torch.arange(80,device='cuda'),indexing='ij')
        for kind,net in [('unet',u),('mlp',m)]:
            rows=[]
            for t in [.1,.5,.95]:
                for seed in [0,1]:
                    x=((1-t)*noise[seed:seed+1]+t*z[seed:seed+1]).detach().requires_grad_(True);y=x+(1-t)*net(x,t)
                    for py,px in [(25,25),(25,55),(45,40),(60,40)]:
                        grad=torch.autograd.grad(y[0,py*80+px],x,retain_graph=True)[0].detach().reshape(80,80);en=grad.square();total=en.sum();radius=torch.maximum((yy-py).abs(),(xx-px).abs());vals={str(k):float(en[radius<=k//2].sum()/total) for k in [3,9,17,33]}
                        rows.append({'tau':t,'seed':seed,'pixel':[py,px],'energy_within_patch':vals,'total_squared_gradient':float(total)})
                    del x,y
            report['jacobian'][kind]=rows
    del ref,u,m;gc.collect();torch.cuda.empty_cache();dump(O/'probes.json',report);print('FIELD_DONE',n,flush=True)
# Patch coverage: same-position vs pooled positions, with equal dictionary size control.
with torch.inference_mode():
    pos=np.random.default_rng(923).integers(0,6400,size=64);qimg=groups[1][-64:].cuda().reshape(-1,1,80,80)
    for k in [3,9,17]:
        qp=F.unfold(qimg,k,padding=k//2).transpose(1,2);q=qp[torch.arange(64,device='cuda'),torch.as_tensor(pos,device='cuda')]
        for n in [10,100,1000]:
            imgs=groups[0][:n].cuda().reshape(-1,1,80,80);patch=F.unfold(imgs,k,padding=k//2).transpose(1,2)
            dictionary=patch[:,::100,:].reshape(-1,k*k).contiguous();same=[]
            for j in range(64):same.append(float((patch[:,pos[j],:]-q[j]).square().mean(1).min().sqrt()))
            pooled,_=nearest_rmse(q,dictionary);choice=torch.as_tensor(np.random.default_rng(923).choice(len(dictionary),n,replace=False),device='cuda');equal,_=nearest_rmse(q,dictionary[choice])
            report['patches'][f'k{k}_N{n}']={'queries':64,'dictionary_same_position':n,'dictionary_pooled':len(dictionary),'same_position_rmse_mean':float(np.mean(same)),'pooled_rmse_mean':float(pooled.mean()),'pooled_equal_count_rmse_mean':float(equal.mean()),'warning':'finite-query pixel patch coverage, not manifold covering radius; patches correlated; padded edges retained'}
            del patch,dictionary,imgs
        del qp,q
dump(O/'probes.json',report);np.savez_compressed(O/'probe_gallery.npz',**gallery);print('PROBES_COMPLETE',flush=True)
