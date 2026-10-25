"""Matched, unconditional MNIST CFM experiment for the JMLR appendix.

All reported distances are pixel RMSE on UNCLIPPED readouts.  The two
training groups partition the official training split; the official test
split is reserved for training diagnostics.  No classifier is used to train
or choose samples.  Resume files contain optimizer and random stream state.
"""
from pathlib import Path
import argparse, copy, csv, fcntl, hashlib, json, math, os, sys, time, traceback
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/root/shared-nvme/experiment-real-data/datasets/mnist/MNIST/raw')
CFG = dict(n_values=[100,1000,10000], groups=['s1','s2'], steps=100000,
           batch_size=128, learning_rate=2e-4, weight_decay=0., ema_decay=.9999,
           gradient_clip=1., eval_every=10000, checkpoints=[25000,50000,100000],
           model_seed=0, split_seed=20260924, stream_seed=202609240,
           samples=256, sampler_intervals=320, tau_end=.999,
           capture_tau=[0.,.01,.02,.035,.05,.075,.1,.15,.2,.25,.3,.4,.5,.65,.8,.9,.95,.98,.995,.999],
           objective='independent Gaussian-to-data linear conditional flow matching',
           nearest_metric='pixel RMSE; unclipped readout; equal candidate counts',
           model_selection='fixed final EMA at 100000 steps; no best-test checkpoint selection')

def dump(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n')
    os.replace(tmp,path)

def save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def append(path,row):
    exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if not exists:w.writeheader()
        w.writerow(row)

class TimeEmbedding(nn.Module):
    def __init__(self,dim=128):
        super().__init__();self.dim=dim
        self.register_buffer('frequency',torch.exp(-math.log(10000)*torch.arange(dim//2)/(dim//2-1)))
        self.net=nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim))
    def forward(self,t):
        a=t[:,None]*1000*self.frequency[None,:]
        return self.net(torch.cat([a.sin(),a.cos()],1))

class MLP(nn.Module):
    def __init__(self):
        super().__init__();w=1024;e=128
        self.time=TimeEmbedding(e);self.input=nn.Linear(784,w)
        self.layers=nn.ModuleList([nn.Linear(w,w) for _ in range(2)])
        self.norms=nn.ModuleList([nn.LayerNorm(w) for _ in range(3)])
        self.films=nn.ModuleList([nn.Linear(e,2*w) for _ in range(3)])
        self.output=nn.Linear(w,784)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,t):
        t=torch.as_tensor(t,device=x.device,dtype=torch.float32).expand(len(x));e=self.time(t)
        h=self.input(x)
        for i in range(3):
            s,b=self.films[i](e).chunk(2,1);h=F.silu(self.norms[i](h)*(1+s)+b)
            if i<2:h=self.layers[i](h)
        return -x+self.output(h)

class Block(nn.Module):
    def __init__(self,a,b,e=128):
        super().__init__();self.n1=nn.GroupNorm(8,a);self.c1=nn.Conv2d(a,b,3,padding=1)
        self.n2=nn.GroupNorm(8,b);self.c2=nn.Conv2d(b,b,3,padding=1)
        self.emb=nn.Linear(e,2*b);self.skip=nn.Identity() if a==b else nn.Conv2d(a,b,1)
    def forward(self,x,e):
        h=self.c1(F.silu(self.n1(x)));s,b=self.emb(F.silu(e)).chunk(2,1)
        h=self.n2(h)*(1+s[:,:,None,None])+b[:,:,None,None]
        return self.skip(x)+self.c2(F.silu(h))

class UNet(nn.Module):
    def __init__(self):
        super().__init__();b=32;self.time=TimeEmbedding();self.stem=nn.Conv2d(1,b,3,padding=1)
        self.e1=Block(b,b);self.down1=nn.Conv2d(b,2*b,4,2,1);self.e2=Block(2*b,2*b)
        self.down2=nn.Conv2d(2*b,4*b,4,2,1);self.mid1=Block(4*b,4*b);self.mid2=Block(4*b,4*b)
        self.up2=nn.ConvTranspose2d(4*b,2*b,4,2,1);self.d2=Block(4*b,2*b)
        self.up1=nn.ConvTranspose2d(2*b,b,4,2,1);self.d1=Block(2*b,b)
        self.norm=nn.GroupNorm(8,b);self.out=nn.Conv2d(b,1,3,padding=1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,t):
        t=torch.as_tensor(t,device=x.device,dtype=torch.float32).expand(len(x));e=self.time(t)
        h=self.stem(x.reshape(-1,1,28,28));s1=self.e1(h,e);s2=self.e2(self.down1(s1),e)
        h=self.mid2(self.mid1(self.down2(s2),e),e)
        h=self.d2(torch.cat([self.up2(h),s2],1),e);h=self.d1(torch.cat([self.up1(h),s1],1),e)
        return -x+self.out(F.silu(self.norm(h))).flatten(1)

def model(kind):return MLP() if kind=='mlp' else UNet()

def load_data():
    images=np.fromfile(RAW/'train-images-idx3-ubyte',dtype=np.uint8,offset=16).reshape(60000,784)
    labels=np.fromfile(RAW/'train-labels-idx1-ubyte',dtype=np.uint8,offset=8)
    test=np.fromfile(RAW/'t10k-images-idx3-ubyte',dtype=np.uint8,offset=16).reshape(10000,784)
    rng=np.random.default_rng(CFG['split_seed']);perm=rng.permutation(60000)
    ids={'s1':perm[:30000],'s2':perm[30000:]}
    assert not np.intersect1d(ids['s1'],ids['s2']).size
    groups={k:torch.from_numpy(images[v].copy()).float().div_(255).cuda() for k,v in ids.items()}
    holdout=torch.from_numpy(test.copy()).float().div_(255).cuda()
    (ROOT/'data').mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'data/partitions.npz',**ids,train_labels=labels)
    dump(ROOT/'data_contract.json',dict(raw_sha256={p.name:sha(p) for p in RAW.glob('*ubyte')},
         training_partition='random permutation of official 60000 training images, split into two disjoint groups of 30000',
         heldout_diagnostic='official 10000 test images; not used for gradients or final checkpoint selection',
         counts={f'{g}_N{n}':np.bincount(labels[ids[g][:n]],minlength=10).tolist() for g in ids for n in CFG['n_values']},
         pixel_space='original 28x28 grayscale /255; no resizing, PCA, augmentation or label conditioning'))
    return groups,holdout

@torch.inference_mode()
def evaluate(m,train,holdout):
    gen=torch.Generator(device='cuda').manual_seed(202609249)
    noise=torch.randn((1024,784),device='cuda',generator=gen)
    tgrid=[.1,.3,.5,.8,.95];rows=[]
    for split,ref in [('train',train),('heldout',holdout)]:
        z=ref[torch.randint(len(ref),(1024,),device='cuda',generator=gen)]
        for t in tgrid:
            val=[]
            for b in range(0,len(z),128):
                xx=(1-t)*noise[b:b+128]+t*z[b:b+128]
                pred=xx+(1-t)*m(xx,t)
                val.append((pred-z[b:b+128]).square().mean(1))
            v=torch.cat(val);rows.append(dict(split=split,tau=t,clean_mse=float(v.mean()),query_count=len(v)))
    return rows

def train(kind,n,g,groups,holdout):
    d=ROOT/'runs'/kind/f'N{n}'/g;d.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256((json.dumps(CFG,sort_keys=True)+kind+str(n)+g+sha(__file__)).encode()).hexdigest()
    if (d/'complete.json').exists():
        assert json.loads((d/'complete.json').read_text())['signature']==signature
        return
    torch.manual_seed(CFG['model_seed']);torch.cuda.manual_seed_all(CFG['model_seed'])
    m=model(kind).cuda();ema=copy.deepcopy(m).eval().requires_grad_(False)
    opt=torch.optim.AdamW(m.parameters(),lr=CFG['learning_rate'],weight_decay=0.)
    stream=torch.Generator(device='cuda').manual_seed(CFG['stream_seed']);start=0;elapsed=0
    if (d/'resume.pt').exists():
        state=torch.load(d/'resume.pt',map_location='cpu',weights_only=False);assert state['signature']==signature
        m.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer'])
        stream.set_state(state['stream']);start=state['step'];elapsed=state['elapsed'];del state
    ref=groups[g][:n];begin=time.time();total=0.;count=0
    dump(d/'config.json',dict(kind=kind,n=n,group=g,signature=signature,parameters=sum(p.numel() for p in m.parameters()),config=CFG))
    ep=list(ema.parameters());mp=list(m.parameters())
    for step in range(start+1,CFG['steps']+1):
        z=ref[torch.randint(n,(CFG['batch_size'],),device='cuda',generator=stream)]
        noise=torch.randn(z.shape,device='cuda',generator=stream);t=torch.rand((len(z),1),device='cuda',generator=stream)
        x=(1-t)*noise+t*z;opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):v=m(x,t[:,0]);loss=(v.float()-(z-noise)).square().mean()
        loss.backward();grad=nn.utils.clip_grad_norm_(m.parameters(),CFG['gradient_clip'])
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite training')
        opt.step()
        with torch.no_grad():torch._foreach_mul_(ep,CFG['ema_decay']);torch._foreach_add_(ep,mp,alpha=1-CFG['ema_decay'])
        total+=float(loss.detach());count+=1
        if step%1000==0:
            dump(ROOT/'status.json',dict(state='training',kind=kind,n=n,group=g,step=step,total_steps=CFG['steps'],elapsed=elapsed+time.time()-begin,pid=os.getpid()))
        if step%CFG['eval_every']==0 or step in CFG['checkpoints']:
            val=evaluate(ema,ref,holdout)
            append(d/'training.csv',dict(step=step,loss=total/count,heldout_clean_mse=np.mean([a['clean_mse'] for a in val if a['split']=='heldout']),seconds=elapsed+time.time()-begin))
            dump(d/f'validation_{step:06d}.json',val);print('TRAIN',kind,n,g,step,elapsed+time.time()-begin,flush=True)
            total=0.;count=0
            save(d/'resume.pt',dict(model=m.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),stream=stream.get_state(),step=step,signature=signature,elapsed=elapsed+time.time()-begin))
        if step in CFG['checkpoints']:
            save(d/f'ema_{step:06d}.pt',dict(ema=ema.state_dict(),step=step,signature=signature,kind=kind))
    dump(d/'complete.json',dict(signature=signature,step=CFG['steps'],checkpoint_sha256=sha(d/f"ema_{CFG['steps']:06d}.pt")))
    del m,ema,opt;torch.cuda.empty_cache()

class Empirical:
    def __init__(self,ref):self.ref=ref.double();self.norm=self.ref.square().sum(1)
    def posterior(self,x,t):
        # float64 and centering protect likelihood differences at small residual noise.
        xx=x.double();t=float(t);s=1-t
        if t==0:
            return self.ref.mean(0).expand(len(x),-1).to(x.dtype),torch.full((len(x),),len(self.ref),device=x.device,dtype=x.dtype)
        logits=(t*xx@self.ref.T-.5*t*t*self.norm[None,:])/(s*s)
        w=torch.softmax(logits,1);mean=w@self.ref;ess=1/w.square().sum(1)
        return mean.to(x.dtype),ess.to(x.dtype)
    def __call__(self,x,t):return (self.posterior(x,t)[0]-x)/(1-float(t))

@torch.inference_mode()
def nearest(x,ref):
    xx=x.double();best=torch.full((len(x),),float('inf'),device=x.device,dtype=torch.float64)
    ix=torch.zeros(len(x),device=x.device,dtype=torch.long)
    for start in range(0,len(ref),2048):
        r=ref[start:start+2048].double();c=(xx.square().sum(1)[:,None]+r.square().sum(1)[None,:]-2*xx@r.T).clamp_min(0)
        v,k=c.min(1);take=v<best;ix[take]=k[take]+start;best=torch.minimum(best,v)
    return (best/784).sqrt().float(),ix

@torch.inference_mode()
def integrate(field,x,intervals,capture):
    grid=np.unique(np.r_[np.linspace(0,CFG['tau_end'],intervals+1),capture])
    captured={0.:(x.clone(),x+field(x,0.))} if 0. in capture else {}
    wanted=set(capture)
    for a,b in zip(grid[:-1],grid[1:]):
        h=b-a;v=field(x,float(a));pred=x+h*v;x=x+.5*h*(v+field(pred,float(b)))
        if not torch.isfinite(x).all():raise RuntimeError('nonfinite integration')
        if float(b) in wanted:captured[float(b)]=(x.clone(),x+(1-float(b))*field(x,float(b)))
    return captured

@torch.inference_mode()
def sample(kind,n,g,groups):
    d=ROOT/'data'/kind/f'N{n}'/g;d.mkdir(parents=True,exist_ok=True)
    if (d/'complete.json').exists():return
    ref=groups[g][:n];other=groups['s2' if g=='s1' else 's1'][:n]
    if kind=='empirical':field=Empirical(ref);checkpoint=None
    else:
        checkpoint=ROOT/'runs'/kind/f'N{n}'/g/f"ema_{CFG['steps']:06d}.pt"
        field=model(kind).cuda().eval();state=torch.load(checkpoint,map_location='cpu',weights_only=False);field.load_state_dict(state['ema']);del state
    noise=torch.stack([torch.randn(784,generator=torch.Generator().manual_seed(i)) for i in range(CFG['samples'])]).cuda()
    capture=CFG['capture_tau'];outs=[];raws=[];ess_all=[]
    for b in range(0,len(noise),64):
        cap=integrate(field,noise[b:b+64].clone(),CFG['sampler_intervals'],capture)
        outs.append(torch.stack([cap[t][1] for t in capture]).cpu().numpy())
        raws.append(torch.stack([cap[t][0] for t in capture]).cpu().numpy())
        if kind=='empirical':ess_all.append(torch.stack([field.posterior(cap[t][0],t)[1] for t in capture]).cpu().numpy())
        dump(ROOT/'status.json',dict(state='sampling',kind=kind,n=n,group=g,completed=b+len(noise[b:b+64]),pid=os.getpid()))
    out=np.concatenate(outs,1);raw=np.concatenate(raws,1)
    dist_train=[];dist_ref=[];indices_train=[];indices_ref=[];sensitivity=[]
    for i,t in enumerate(capture):
        y=torch.from_numpy(out[i]).cuda();dt,it=nearest(y,ref);dr,ir=nearest(y,other)
        dist_train.append(dt.cpu().numpy());dist_ref.append(dr.cpu().numpy());indices_train.append(it.cpu().numpy());indices_ref.append(ir.cpu().numpy())
        if i in [0,6,10,14,19]:
            for label,v in [('raw',raw[i]),('clipped',out[i].clip(0,1))]:
                aa,_=nearest(torch.from_numpy(v).cuda(),ref);bb,_=nearest(torch.from_numpy(v).cuda(),other)
                sensitivity.append(dict(tau=t,output=label,dtrain=float(aa.mean()),dref=float(bb.mean())))
    arrays=dict(tau=np.array(capture),reverse_t=2/np.pi*np.arctan(np.array(capture)/(1-np.array(capture))),
                readout=out,raw_trace=raw[:,:8],noise=noise.cpu().numpy(),dtrain=np.array(dist_train),dref=np.array(dist_ref),
                nearest_train_indices=np.array(indices_train),nearest_reference_indices=np.array(indices_ref),
                display_train_neighbors=ref[torch.as_tensor(np.array(indices_train)[:,:8],device='cuda')].cpu().numpy(),
                display_reference_neighbors=other[torch.as_tensor(np.array(indices_ref)[:,:8],device='cuda')].cpu().numpy())
    if ess_all:arrays['ess']=np.concatenate(ess_all,1)
    np.savez_compressed(d/'samples.npz',**arrays);dump(d/'readout_sensitivity.json',sensitivity)
    # Numerical refinement on 32 paired trajectories; never call it a global error bound.
    check_times=[.05,.1,.2,.3,.5,.8,.95,.999]
    fine=integrate(field,noise[:32].clone(),2*CFG['sampler_intervals'],check_times)
    checks=[]
    for t in check_times:
        base=torch.from_numpy(out[capture.index(t),:32]).cuda();delta=(fine[t][1]-base).square().mean(1).sqrt()
        checks.append(dict(tau=t,mean_rmse=float(delta.mean()),max_rmse=float(delta.max()),trajectories=32))
    dump(d/'solver_refinement.json',checks)
    dump(d/'complete.json',dict(kind=kind,n=n,group=g,samples=CFG['samples'],checkpoint_sha256=sha(checkpoint) if checkpoint else None,samples_sha256=sha(d/'samples.npz')))
    print('SAMPLED',kind,n,g,flush=True)
    del field;torch.cuda.empty_cache()

def smoke():
    groups,holdout=load_data();rows=[]
    for kind in ['mlp','unet']:
        torch.manual_seed(0);m=model(kind).cuda();opt=torch.optim.AdamW(m.parameters(),lr=2e-4)
        z=groups['s1'][:128];noise=torch.randn_like(z);t=torch.rand(128,device='cuda');x=(1-t[:,None])*noise+t[:,None]*z
        start=time.time()
        for i in range(100):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):v=m(x,t);loss=(v.float()-(z-noise)).square().mean()
            loss.backward();opt.step()
        torch.cuda.synchronize()
        assert torch.isfinite(loss);rows.append(dict(kind=kind,loss=float(loss.detach()),seconds_per_step=(time.time()-start)/100,parameters=sum(p.numel() for p in m.parameters())))
        del m,opt;torch.cuda.empty_cache()
    e=Empirical(groups['s1'][:1]);mean,ess=e.posterior(torch.randn(3,784,device='cuda'),.4)
    assert torch.allclose(mean,groups['s1'][:1].expand(3,-1)) and torch.all(ess==1)
    dt,_=nearest(groups['s1'][:3],groups['s1'][:100]);assert float(dt.max())<1e-7
    dump(ROOT/'audit_smoke.json',dict(passed=True,benchmarks=rows,empirical_single_atom=True,nearest_self=True));print(rows,flush=True)

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if a.smoke:return smoke()
    lock=(ROOT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'audit_smoke.json').read_text())['passed']
    dump(ROOT/'protocol.json',dict(**CFG,code_sha256=sha(__file__),torch_version=torch.__version__,python=sys.version))
    dump(ROOT/'supervisor.json',dict(state='running',pid=os.getpid(),started=time.time()))
    groups,holdout=load_data()
    for n in CFG['n_values']:
        for g in CFG['groups']:sample('empirical',n,g,groups)
    for n in [10000,1000,100]:
        for kind in ['unet','mlp']:
            for g in CFG['groups']:
                train(kind,n,g,groups,holdout);sample(kind,n,g,groups)
    dump(ROOT/'status.json',dict(state='complete_pending_review',pid=os.getpid()))
    dump(ROOT/'supervisor.json',dict(state='complete_pending_review',pid=os.getpid(),finished=time.time()))

if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'status.json',dict(state='error',pid=os.getpid(),traceback=traceback.format_exc()));raise
