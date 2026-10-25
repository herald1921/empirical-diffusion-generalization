"""Frozen-protocol supplement. No deletions; writes only this round's results."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if (ROOT/'deps').exists(): sys.path.insert(0,str(ROOT/'deps'))
import argparse,json,time,hashlib,shutil,math
import numpy as np
from scipy.special import ndtri,ndtr,roots_legendre,softmax,erfcx
from scipy.stats import qmc
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.integrate import cumulative_trapezoid
from concurrent.futures import ProcessPoolExecutor,as_completed
import ot
CODE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RESULTS=ROOT/'results'/CODE_SHA256[:12]
ETA=np.unique(np.r_[np.logspace(-3,0,25),.03,.04,.08,.12,.18,.27,.4,.6,.9])

def dump(p,v):
    if shutil.disk_usage(ROOT).free<250_000_000: raise RuntimeError('Disk free below 250 MB; no further writes')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False))

def ckey(c):return f"{c['shape']}_N{c['n']}_{c.get('sampling','design')}_r{c.get('rep',0)}"

def angles(u,a=.1):
    t=2*np.pi*u
    for _ in range(15):t-=(t+a*np.sin(t)-2*np.pi*u)/(1+a*np.cos(t))
    return t

class Eight:
    def __init__(self):
        t=np.linspace(0,2*np.pi,65537);p=self.point(t);ar=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        self.arc=ar/ar[-1];self.t=t
        u=(np.arange(65536)+.5)/65536
        z=self.point(np.interp(u,self.arc,t));self.scale=1/np.sqrt(np.mean(np.sum(z*z,axis=1)))
        self.grid=np.linspace(0,2*np.pi,16384,endpoint=False);self.tree=cKDTree(self.point(self.grid))
    @staticmethod
    def point(t):return np.column_stack([np.sin(t),np.sin(2*t)])
    def sample(self,u):return self.scale*self.point(np.interp(u,self.arc,self.t))
    def project(self,x):
        y=x/self.scale;_,ix=self.tree.query(y,k=4);t=self.grid[ix]
        for _ in range(8):
            rr=np.stack([np.sin(t),np.sin(2*t)],axis=-1)-y[:,None,:]
            dd=np.stack([np.cos(t),2*np.cos(2*t)],axis=-1);d2=np.stack([-np.sin(t),-4*np.sin(2*t)],axis=-1)
            h=np.sum(dd*dd+rr*d2,axis=-1);t-=np.clip(np.sum(rr*dd,axis=-1)/np.where(abs(h)>1e-12,h,1e-12),-.002,.002)
        z=np.stack([np.sin(t),np.sin(2*t)],axis=-1);j=np.argmin(np.sum((z-y[:,None,:])**2,axis=-1),axis=1)
        return self.scale*z[np.arange(len(x)),j]
EIGHT=None
def get_eight():
    global EIGHT
    if EIGHT is None:EIGHT=Eight()
    return EIGHT

def sample(shape,n,mode='design',seed=0):
    rng=np.random.default_rng(seed);u=rng.random(n) if mode=='iid' else (np.arange(n)+.5)/n
    if shape=='line':return np.column_stack([np.sqrt(3)*(2*u-1),np.zeros(n)])
    if shape=='eight':return get_eight().sample(u)
    if shape in ['circle','ellipse']:
        t=angles(u);return np.column_stack([2*np.cos(t),(1 if shape=='ellipse' else 2)*np.sin(t)])
    if mode=='iid':
        z=rng.normal(size=(n,3));return 2*z/np.linalg.norm(z,axis=1)[:,None]
    if mode=='target':
        uu=qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.log2(n)));z=1-2*uu[:,0];t=2*np.pi*uu[:,1]
    else:z=1-2*(np.arange(n)+.5)/n;t=np.arange(n)*np.pi*(3-np.sqrt(5))
    rr=np.sqrt(1-z*z);return 2*np.column_stack([rr*np.cos(t),rr*np.sin(t),z])

def project(x,shape):
    if shape=='eight':return get_eight().project(x)
    if shape=='line':return np.column_stack([np.clip(x[:,0],-np.sqrt(3),np.sqrt(3)),np.zeros(len(x))])
    if shape in ['circle','sphere']:
        rr=np.linalg.norm(x,axis=1);z=2*x/np.maximum(rr[:,None],1e-300);z[rr==0]=np.r_[2.,np.zeros(x.shape[1]-1)];return z
    aa=np.array([4.,1.]);y=abs(x);off=y[:,1]>1e-14;ans=np.zeros_like(x)
    if off.any():
        v=y[off];lo=np.full(len(v),np.nextafter(-1.,0.));hi=np.maximum(1.,2*np.linalg.norm(v,axis=1))
        for _ in range(85):
            mid=(lo+hi)/2;res=(aa*v*v/(aa+mid[:,None])**2).sum(1)-1;lo=np.where(res>0,mid,lo);hi=np.where(res>0,hi,mid)
        ans[off]=aa*v/(aa+((lo+hi)/2)[:,None])
    px=np.minimum(2.,4*y[~off,0]/3);ans[~off]=np.column_stack([px,np.sqrt(np.maximum(0,1-px*px/4))])
    return ans*np.where(x<0,-1.,1.)

def post(x,z,e):return softmax(-cdist(x,z,'sqeuclidean')/(2*e*e),axis=1)@z

def ot_dist(x,y):
    a=np.full(len(x),1/len(x));b=np.full(len(y),1/len(y));cost=cdist(x,y,'sqeuclidean')
    plan,log=ot.emd(a,b,cost,log=True,numItermax=5000000,numThreads=1,check_marginals=True)
    v=float(np.einsum('ij,ij->',plan,cost));marginal=max(float(abs(plan.sum(1)-a).max()),float(abs(plan.sum(0)-b).max()))
    gap=abs(v-a@log['u']-b@log['v']);violation=max(float((log['u'][i:i+256,None]+log['v'][None,:]-cost[i:i+256]).max()) for i in range(0,len(x),256))
    if log.get('warning') or gap>2e-8*max(1,v) or violation>2e-8 or marginal>2e-9:raise RuntimeError(dict(warning=log.get('warning'),gap=gap,violation=violation,marginal=marginal))
    return float(np.sqrt(max(0,v))),dict(gap=float(gap),violation=float(max(0,violation)),marginal=marginal,result_code=int(log['result_code']))

def noise_nodes(z,m,seed):
    k=m//len(z);assert k*len(z)==m and k>=2
    blocks=[]
    for i in range(len(z)):
        u=qmc.Sobol(z.shape[1],scramble=True,seed=seed+101*i).random_base2(int(np.log2(k))-1)
        g=ndtri(np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps));blocks.append(np.r_[g,-g])
    return np.repeat(z,k,axis=0),np.concatenate(blocks)

def line_metrics(z,e,level):
    z=z[:,0];a=np.sqrt(3);edges=np.unique(np.r_[(z[:,None]+e*np.array([-12,-8,-4,-2,-1,0,1,2,4,8,12])).ravel(),-a,a]);t,w=roots_legendre(32*2**level)
    x=((edges[1:]+edges[:-1])[:,None]/2+np.diff(edges)[:,None]*t/2).ravel();w=(np.diff(edges)[:,None]*w/2).ravel()
    u=(x[:,None]-z)/e;v=np.exp(-u*u/2);d=v.mean(1)/(np.sqrt(2*np.pi)*e);F=ndtr(u).mean(1);target=a*(2*F-1);ww=w*d
    den=softmax(-u*u/2,axis=1)@z;proj=np.clip(x,-a,a)
    f=lambda v:float(np.sqrt(max(0,v)))
    r=dict(raw_w2=f(ww@((x-target)**2)+e*e),raw_wt=f(ww@((proj-target)**2)),raw_wn=f(ww@((x-proj)**2)+e*e),tweedie_w2=f(ww@((den-target)**2)),tweedie_wt=f(ww@((den-target)**2)),tweedie_wn=0.,mass=float(ww.sum()))
    z=np.sort(z);lo=a*(2*np.arange(len(z))/len(z)-1);hi=lo+2*a/len(z);r['endpoint']=f(np.sum(((hi-z)**3-(lo-z)**3)/(6*a)))
    assert abs(r['mass']-1)<1e-7
    return r

def ellipse_density(theta,z,e):
    q=np.column_stack([2*np.cos(theta),np.sin(theta)]);d=np.sqrt(np.cos(theta)**2/4+np.sin(theta)**2)
    normal=np.column_stack([np.cos(theta)/2,np.sin(theta)])/d[:,None]
    tangent=np.column_stack([-2*np.sin(theta),np.cos(theta)])/(2*d[:,None]);kappa=1/(4*d**3);L=-d
    coefficient=np.maximum(0,1+kappa*L);density=np.zeros(len(theta))
    for zz in z:
        delta=zz-q;a=np.sum(delta*normal,axis=1);b=np.sum(delta*tangent,axis=1);v=(a-L)/e
        phi=np.exp(-v*v/2)/np.sqrt(2*np.pi);cdf=ndtr(v);moment=np.empty_like(v);pos=v>=0
        moment[pos]=phi[pos]+v[pos]*cdf[pos]
        moment[~pos]=phi[~pos]*np.maximum(0,1+v[~pos]*np.sqrt(np.pi/2)*erfcx(-v[~pos]/np.sqrt(2)))
        density+=(2*d)*np.exp(-b*b/(2*e*e))/(np.sqrt(2*np.pi)*e)*(coefficient*cdf+kappa*e*moment)
    return density/len(z)

def compute(job):
    started=time.time();kind=job['kind'];shape=job['shape'];n=job['n'];rep=job.get('rep',0);level=job.get('level',0);scramble=job.get('scramble',0);e=job['eta'];seed=2026100900+rep
    z=sample(shape,n,job.get('sampling','design'),seed);r=dict(job,code_sha256=CODE_SHA256,python=sys.version,numpy=np.__version__,POT=ot.__version__,training_seed=seed,training_sha256=hashlib.sha256(z.tobytes()).hexdigest());m=job.get('source_nodes',1024*2**level);nt=job.get('target_nodes',2048*2**level)
    if kind=='normal':
        zz,g=noise_nodes(z,m,2026109000+rep*100003+scramble*10000001);x=zz+e*g;den=post(x,z,e);px=project(x,shape);pd=project(den,shape)
        for name,xx,pp in [('raw',x,px),('tweedie',den,pd)]:
            val=np.sum((xx-pp)**2,axis=1);r[name+'_wn']=float(np.sqrt(val.mean()));r[name+'_risk']=float(val.mean())
        r.update(source_nodes=m,reach=.5,normal_upper=float(np.sqrt(8)*e*e/.5),note='O(eta^2) upper bound, not an asserted slope-2 equivalence; antithetic scrambled Sobol quadrature, no iid standard-error interpretation')
    elif shape=='line':r.update(line_metrics(z,e,level))
    elif kind=='ellipse_turn':
        ng=job.get('angle_nodes',131073);theta=np.linspace(0,2*np.pi,ng);density=ellipse_density(theta,z,e)
        cumulative=np.r_[0,cumulative_trapezoid(density,theta)];mass=float(cumulative[-1]);assert abs(mass-1)<1e-8,(mass,e,n)
        tt=np.interp((np.arange(nt)+.5)/nt,cumulative/mass,theta);source=np.column_stack([2*np.cos(tt),np.sin(tt)]);target=sample('ellipse',nt)
        r['raw_wt'],r['ot']=ot_dist(source,target);r.update(source_nodes=nt,target_nodes=nt,angle_nodes=ng,mass=mass,underflow_zero_density_nodes=int((density==0).sum()),method='analytic normal-fibre angular marginal; nonnegative tail-moment representation')
    elif kind=='turn' and shape=='circle':
        site=np.arctan2(z[:,1],z[:,0]);theta=np.linspace(0,2*np.pi,131073);delta=theta[:,None]-site;aa=2*np.cos(delta)/e
        density=np.maximum((np.exp(-2/e**2)/(2*np.pi)+aa/np.sqrt(2*np.pi)*np.exp(-2*np.sin(delta)**2/e**2)*ndtr(aa)).mean(1),0)
        cum=np.r_[0,cumulative_trapezoid(density,theta)];mass=cum[-1];assert abs(mass-1)<1e-8
        tt=np.interp((np.arange(nt)+.5)/nt,cum/mass,theta);x=2*np.column_stack([np.cos(tt),np.sin(tt)]);target=sample('circle',nt)
        r['raw_wt'],r['ot']=ot_dist(x,target);r.update(source_nodes=nt,target_nodes=nt,mass=float(mass))
    else:
        zz,g=noise_nodes(z,m,2026110000+rep*100003+scramble*10000001);x=zz+e*g;target=sample(shape,nt,'target',777+scramble)
        if kind=='turn':r['raw_wt'],r['ot']=ot_dist(project(x,shape),target)
        else:
            y=post(x,z,e);px=project(x,shape);py=project(y,shape);r['ot_checks']={}
            for name,v in [('raw_w2',x),('raw_wt',px),('tweedie_w2',y),('tweedie_wt',py)]:r[name],r['ot_checks'][name]=ot_dist(v,target)
            r['raw_wn']=float(np.sqrt(np.mean(np.sum((x-px)**2,axis=1))));r['tweedie_wn']=float(np.sqrt(np.mean(np.sum((y-py)**2,axis=1))));r['endpoint'],r['ot_checks']['endpoint']=ot_dist(z,target)
            for pref in ['raw','tweedie']:
                assert r[pref+'_wn']<=r[pref+'_w2']+1e-7
                assert r[pref+'_w2']<=r[pref+'_wt']+r[pref+'_wn']+1e-7
        r.update(source_nodes=m,target_nodes=nt)
    r['seconds']=time.time()-started
    jid=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:16];dump(RESULTS/kind/(jid+'.json'),r);return r

def topology():
    import gudhi
    out=[]
    for sh in ['line','eight']:
        for n in [4,8,16,32,64]:
            z=sample(sh,n);st=gudhi.AlphaComplex(points=z.tolist(),precision='exact').create_simplex_tree();st.persistence(min_persistence=1e-12)
            ints={str(k):[[float(np.sqrt(max(0,b))),None if not np.isfinite(d) else float(np.sqrt(d))] for b,d in st.persistence_intervals_in_dimension(k)] for k in [0,1]}
            out.append(dict(shape=sh,n=n,training=z.tolist(),training_sha256=hashlib.sha256(z.tobytes()).hexdigest(),intervals_radius=ints,target_betti=[1,0] if sh=='line' else [1,2],closed_manifold_guarantee=None,reason='boundary' if sh=='line' else 'self-intersection',c_delta=float(np.sqrt(2)+np.sqrt(2*np.log(20)))))
    dump(RESULTS/'topology.json',out);return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--topology',action='store_true');a=ap.parse_args()
    if a.topology:topology();print('topology complete',flush=True)
    if a.jobs:
        jobs=json.loads(Path(a.jobs).read_text());pending=[]
        for j in jobs:
            jid=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()[:16]
            if not (RESULTS/j['kind']/(jid+'.json')).exists():pending.append(j)
        print(json.dumps(dict(total=len(jobs),pending=len(pending))),flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(compute,j):j for j in pending}
            for i,f in enumerate(as_completed(futures)):
                row=f.result();print(json.dumps(dict(done=i+1,total=len(pending),kind=row['kind'],shape=row['shape'],eta=row['eta'],seconds=row['seconds'])),flush=True)
        print('COMPLETE',flush=True)
if __name__=='__main__':main()
