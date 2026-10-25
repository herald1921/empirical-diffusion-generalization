"""Deterministic polar refinement and dense angular transport for Figures 4/5."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import roots_legendre,logsumexp
from scipy.spatial.distance import cdist
import ot
import make_figures_04_05 as f

def polar_nodes(eta,order):
    theta=np.sort(np.mod(np.arctan2(f.P[:,1],f.P[:,0]),2*np.pi))
    gap=np.mod(np.roll(theta,-1)-theta,2*np.pi)
    breaks=np.sort(np.r_[theta,theta+gap/2]); starts=breaks
    ends=np.r_[breaks[1:],breaks[0]+2*np.pi]
    u,w=roots_legendre(order)
    angles=((starts[:,None]+ends[:,None])/2+(ends-starts)[:,None]/2*u).ravel()
    aw=((ends-starts)[:,None]/2*w).ravel()
    v,rw=roots_legendre(2*order)
    lo=max(0.,f.R-9*eta); hi=f.R+9*eta
    radius=(lo+hi)/2+(hi-lo)/2*v; rw=rw*(hi-lo)/2
    x=(radius[None,:,None]*np.stack([np.cos(angles),np.sin(angles)],axis=1)[:,None,:]).reshape(-1,2)
    logits=-cdist(x,f.P,'sqeuclidean')/(2*eta*eta)
    weight=np.exp(logsumexp(logits,axis=1))/(f.N*2*np.pi*eta*eta)
    weight*=np.outer(aw,rw*radius).ravel()
    if abs(weight.sum()-1)>1e-8: raise RuntimeError(f'Polar mass check failed at {eta}: {weight.sum()}')
    return x,weight

def moments(eta,order):
    x,w=polar_nodes(eta,order)
    tw,rad=f.tweedie(x,eta); ideal=f.core.project(x); poly=f.polygon(x)
    return dict(eta=eta,order=order,mass=float(w.sum()),
                tweedie_wn=float(np.sqrt(w@rad**2)),
                polygon_wn=float(np.sqrt(w@np.sum((poly-ideal)**2,axis=1))),
                tweedie_bias=float(np.sqrt(w@np.sum((tw-ideal)**2,axis=1))))

def dense_angular(eta,order=64,bins=2048):
    # Equal-mass angular quantiles of the dense polar pushforward.
    # `bins` is retained as a cache field name; it means quantile count.
    x,w=polar_nodes(eta,order)
    tw,_=f.tweedie(x,eta)
    angle=np.mod(np.arctan2(tw[:,1],tw[:,0]),2*np.pi)
    perm=np.argsort(angle); angle=angle[perm]; w=w[perm]; w/=w.sum()
    cumulative=np.cumsum(w)-w/2
    theta=np.interp((np.arange(bins)+.5)/bins,cumulative,angle)
    source=f.R*np.column_stack([np.cos(theta),np.sin(theta)])
    y=f.core.target_design(bins)
    result=f.core.exact_ot(source,y)
    return dict(eta=eta,wt=float(result[0]),order=order,bins=bins,
                duality_gap=result[1],marginal_error=result[2],dual_violation=result[3])

def refinement_worker(eta):
    coarse=moments(eta,32); fine=moments(eta,64)
    fine['sensitivity']=abs(fine['tweedie_wn']-coarse['tweedie_wn'])
    fine['coarse_wn']=coarse['tweedie_wn']
    return fine

def run(workers=2):
    bases=f.read(f.ROOT/'reference/data/distances.csv')
    normals=f.read(f.DATA/'figure04_normal.csv')
    grid=sorted(set(round(r['eta'],12) for r in bases+normals if r['eta']>=.06))
    rows=f.read(f.DATA/'polar_moments.csv'); have={round(r['eta'],12) for r in rows}
    print(f'Polar refinement: {len(set(grid)-have)} scales pending',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(refinement_worker,e) for e in grid if e not in have]):
            rows.append(future.result()); rows.sort(key=lambda r:r['eta']); f.write(f.DATA/'polar_moments.csv',rows)
    angular=f.read(f.DATA/'tweedie_angular_dense.csv'); have={round(r['eta'],12) for r in angular}
    grid=[r['eta'] for r in bases if r['eta']>=.08 and round(r['eta'],12) not in have]
    print(f'Dense Tweedie angular OT: {len(grid)} scales pending',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(dense_angular,e) for e in grid]):
            angular.append(future.result()); angular.sort(key=lambda r:r['eta']); f.write(f.DATA/'tweedie_angular_dense.csv',angular)
            if len(angular)%20==0: print(f'Dense angular: {len(angular)} done',flush=True)
    return rows,angular

if __name__=='__main__': run()