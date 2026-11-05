"""Uniform-circle rerun of the repository's unregularized transport protocol.

Source: upstream_repo/section_04_distribution/figure02/make_figure_02.py
and section_05_early_stopping/numerics/numerics/{make_figures_04_05,refine_integrals}.py.
The population and the sample are deliberately changed to uniform / equispaced.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import argparse,json,time,csv,hashlib,subprocess
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import ndtri,ndtr,i0e,i1e,softmax,roots_legendre,logsumexp
from scipy.spatial.distance import cdist
from scipy.integrate import cumulative_trapezoid
from scipy.stats import qmc
import ot

ROOT=Path(__file__).resolve().parent
N=16;R=2.;SEED=20260920
THETA=2*np.pi*np.arange(N)/N
P=R*np.column_stack((np.cos(THETA),np.sin(THETA)))
SN=R*np.sqrt(2-2*np.sin(np.pi/N)/(np.pi/N))

def noisy_nodes(eta,power,seed):
    u=qmc.Sobol(2,scramble=True,seed=seed).random_base2(power-1)
    g=ndtri(u);g=np.r_[g,-g]
    c,s=np.cos(THETA)[:,None],np.sin(THETA)[:,None]
    noise=np.stack([c*g[None,:,0]-s*g[None,:,1],s*g[None,:,0]+c*g[None,:,1]],axis=-1)
    return (P[:,None,:]+eta*noise).reshape(-1,2)

def target(count):
    # Midpoints are reflection symmetric around each training-site direction.
    a=2*np.pi*(np.arange(count)+.5)/count-np.pi/N
    return R*np.column_stack([np.cos(a),np.sin(a)])

def project(x):
    return R*x/np.linalg.norm(x,axis=1)[:,None]

def tweedie(x,eta):
    w=softmax(-cdist(x,P,'sqeuclidean')/(2*eta**2),axis=1)
    return w@P

def normal_exact(eta):
    z=R*R/(4*eta*eta)
    mean=eta*np.sqrt(np.pi/2)*((1+2*z)*i0e(z)+2*z*i1e(z))
    return np.sqrt(max(0.,2*R*R+2*eta*eta-2*R*mean))

def exact_ot(x,y):
    a=np.full(len(x),1/len(x));b=np.full(len(y),1/len(y))
    cost=np.ascontiguousarray(cdist(x,y,'sqeuclidean'))
    plan,log=ot.emd(a,b,cost,log=True,numItermax=2000000,numThreads=1)
    primal=float(np.einsum('ij,ij->',plan,cost))
    gap=abs(primal-float(a@log['u']+b@log['v']))
    marg=max(abs(plan.sum(0)-b).max(),abs(plan.sum(1)-a).max())
    violation=max(float(np.max(log['u'][k:k+256,None]+log['v'][None,:]-cost[k:k+256])) for k in range(0,len(x),256))
    if log.get('warning') or gap>2e-8 or marg>2e-9 or violation>2e-8:
        raise RuntimeError((log.get('warning'),gap,marg,violation))
    return primal,dict(gap=gap,marginal=float(marg),violation=violation)

def symmetry_ot(x,y):
    """Exact quotient of finite OT by simultaneous N-fold rotation.

    x and y contain one representative of every rotational orbit. Average
    any transport plan over the group; between two orbits the least cost is
    attained at a relative rotation. Conversely any quotient plan lifts to
    a group-invariant full plan of the same cost. Thus this is the full
    unregularized Euclidean transport problem, not an approximation to OT.
    """
    r=np.linalg.norm(x,axis=1);phi=np.arctan2(x[:,1],x[:,0])
    theta=np.arctan2(y[:,1],y[:,0])
    delta=(phi[:,None]-theta[None,:]+np.pi/N)%(2*np.pi/N)-np.pi/N
    cost=np.ascontiguousarray((r[:,None]-R)**2+2*R*r[:,None]*(1-np.cos(delta)))
    a=np.full(len(x),1/len(x));b=np.full(len(y),1/len(y))
    plan,log=ot.emd(a,b,cost,log=True,numItermax=2000000,numThreads=1)
    primal=float(np.einsum('ij,ij->',plan,cost));gap=abs(primal-float(a@log['u']+b@log['v']))
    marg=max(abs(plan.sum(0)-b).max(),abs(plan.sum(1)-a).max())
    violation=max(float(np.max(log['u'][k:k+256,None]+log['v'][None,:]-cost[k:k+256])) for k in range(0,len(x),256))
    if log.get('warning') or gap>2e-8 or marg>2e-9 or violation>2e-8: raise RuntimeError(('quotient OT failed',gap,marg,violation))
    return primal,dict(gap=gap,marginal=float(marg),violation=violation)

def polar_normal(eta,order=48):
    # N-fold symmetry reduces the repository's polar integral to one sector.
    u,w=roots_legendre(order)
    angles=np.r_[-np.pi/(2*N)+np.pi/(2*N)*u,np.pi/(2*N)+np.pi/(2*N)*u]
    aw=np.tile(w*np.pi/(2*N),2)
    v,rw=roots_legendre(2*order)
    lo=max(0.,R-9*eta);hi=R+9*eta
    radius=(lo+hi)/2+(hi-lo)/2*v;rw=rw*(hi-lo)/2
    x=(radius[None,:,None]*np.column_stack([np.cos(angles),np.sin(angles)])[:,None,:]).reshape(-1,2)
    logits=-cdist(x,P,'sqeuclidean')/(2*eta*eta)
    weight=np.exp(logsumexp(logits,axis=1))/(2*np.pi*eta*eta)*np.outer(aw,rw*radius).ravel()
    tw=softmax(logits,axis=1)@P
    mass=float(weight.sum())
    if abs(mass-1)>1e-7: raise RuntimeError(('polar mass',eta,order,mass))
    return float(np.sqrt(weight@(np.linalg.norm(tw,axis=1)-R)**2)),mass

def angular_wt(eta,count=8192):
    theta=np.linspace(-np.pi/N,np.pi/N,32769)
    delta=theta[:,None]-THETA
    a=R*np.cos(delta)/eta
    density=(np.exp(-R*R/(2*eta*eta))/(2*np.pi)+a/np.sqrt(2*np.pi)*np.exp(-R*R*np.sin(delta)**2/(2*eta*eta))*ndtr(a)).sum(axis=1)
    cdf=np.r_[0.,cumulative_trapezoid(density,theta)]
    mass=float(cdf[-1]);cdf/=mass
    u=(np.arange(count)+.5)/count
    qs=np.interp(u,cdf,theta);qt=(2*u-1)*np.pi/N
    return float(R*np.sqrt(np.mean(2-2*np.cos(qs-qt)))),mass

def solve(task):
    eta,power,count,seed,level=task;t=time.perf_counter()
    if eta==0: return dict(eta=0.,power=power,count=count,seed=seed,level=level,raw_w2=SN,raw_wt=SN,raw_wn=0.,tweedie_w2=SN,tweedie_wn=0.,raw_discrete=SN,tw_discrete=SN,normal_tw_check=0.,polar_mass=1.,angular_mass=1.,gap=0.,marginal=0.,violation=0.,seconds=0.)
    x=noisy_nodes(eta,power,seed)[:2**power];y=target(count)[:count//N];tw=tweedie(x,eta)
    raw,ra=symmetry_ot(x,y);den,de=symmetry_ot(tw,y)
    wn=normal_exact(eta);wnx=np.mean((np.linalg.norm(x,axis=1)-R)**2)
    if eta>=.04:
        wnt,pm=polar_normal(eta,48);wnt2,_=polar_normal(eta,72)
    else:
        wnt=np.sqrt(np.mean((np.linalg.norm(tw,axis=1)-R)**2));wnt2=wnt;pm=1.
    raw2=wn*wn+max(0.,raw-wnx)
    den2=wnt2*wnt2+max(0.,den-np.mean((np.linalg.norm(tw,axis=1)-R)**2))
    wt,am=angular_wt(eta)
    return dict(eta=eta,power=power,count=count,seed=seed,level=level,raw_w2=np.sqrt(raw2),raw_wt=wt,raw_wn=wn,tweedie_w2=np.sqrt(den2),tweedie_wn=wnt2,raw_discrete=np.sqrt(raw),tw_discrete=np.sqrt(den),normal_tw_check=abs(wnt-wnt2),polar_mass=pm,angular_mass=am,gap=max(ra['gap'],de['gap']),marginal=max(ra['marginal'],de['marginal']),violation=max(ra['violation'],de['violation']),seconds=time.perf_counter()-t)

def write_rows(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(tasks,name,workers):
    path=ROOT/name;rows=[]
    if path.exists():
        rows=[{k:(v if k=='level' else float(v)) for k,v in r.items()} for r in csv.DictReader(path.open())]
    done={(round(r['eta'],9),int(r['power']),int(r['count']),int(r['seed'])) for r in rows}
    tasks=[t for t in tasks if (round(t[0],9),t[1],t[2],t[3]) not in done]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs=[pool.submit(solve,t) for t in tasks]
        for f in as_completed(futs):
            r=f.result();rows.append(r);rows.sort(key=lambda x:(x['eta'],x['level']))
            write_rows(path,rows)
            print(f"{name}: {len(rows)} eta={r['eta']:.5f} raw={r['raw_w2']/SN:.5f} den={r['tweedie_w2']/SN:.5f} {r['seconds']:.1f}s",flush=True)
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--audit',action='store_true');ap.add_argument('--dense',action='store_true');ap.add_argument('--verify-symmetry',action='store_true');args=ap.parse_args()
    if args.verify_symmetry:
        checks=[]
        for e in [.11,.2,.35,.8]:
            x=noisy_nodes(e,7,SEED);y=target(1024)
            for name,xx in [('raw',x),('tweedie',tweedie(x,e))]:
                full,_=exact_ot(xx,y);quot,_=symmetry_ot(xx[:128],y[:64])
                checks.append(dict(eta=e,law=name,full=full,quotient=quot,difference=abs(full-quot)))
        assert max(c['difference'] for c in checks)<1e-12
        (ROOT/'symmetry_ot_check.json').write_text(json.dumps(checks,indent=2));print(checks)
    elif args.dense:
        grid=np.unique(np.round(np.r_[np.linspace(0,.8,65),np.arange(.09,.131,.002),np.arange(.3,.411,.01),.35,.5],9))
        run([(float(e),12,4096,SEED,'main_dense') for e in grid],'curves_dense.csv',args.workers)
    elif args.audit:
        run([(e,p,c,s,'refined' if p==8 else 'scramble') for e in [.1,.12,.14,.16,.175,.19,.2,.35,.5] for p,c,s in [(8,2048,SEED),(7,1024,SEED+1)]],'audit.csv',args.workers)
    else:
        grid=np.unique(np.r_[np.linspace(0,.8,49),np.arange(.1,.241,.005),.35,.5])
        run([(float(e),7,1024,SEED,'main') for e in grid],'curves.csv',args.workers)
