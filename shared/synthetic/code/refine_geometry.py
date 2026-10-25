"""Use exact radial moments and a circle angular marginal as variance controls.

Raw discrete values are retained. This improves quadrature, not model fitting.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,math,time
import numpy as np
from scipy.special import hyp1f1,gammaln,ndtr
from scipy.integrate import cumulative_trapezoid
from run_synthetic import ROOT,DATA,key,points,dump
from scipy.spatial.distance import cdist
import ot


def exact_ot(x,y):
    # Fixed permutations avoid severe simplex degeneracy for ordered circular
    # quantiles. The LP is identical, and every primal/dual check is retained.
    rng=np.random.default_rng(5124)
    xx=x[rng.permutation(len(x))];yy=y[rng.permutation(len(y))]
    a=np.full(len(xx),1/len(xx));b=np.full(len(yy),1/len(yy))
    cost=np.ascontiguousarray(cdist(xx,yy,'sqeuclidean'))
    plan,log=ot.emd(a,b,cost,log=True,numItermax=50000000,numThreads=1,check_marginals=True)
    primal=float(np.einsum('ij,ij->',plan,cost))
    gap=abs(primal-float(a@log['u']+b@log['v']))
    marginal=max(float(np.abs(plan.sum(1)-a).max()),float(np.abs(plan.sum(0)-b).max()))
    violation=max(0.,float((log['u'][:,None]+log['v'][None,:]-cost).max()))
    assert not log.get('warning') and gap<2e-8*max(1.,primal) and marginal<2e-9 and violation<2e-8,dict(warning=log.get('warning'),gap=gap,marginal=marginal,violation=violation)
    return dict(distance=float(np.sqrt(max(0.,primal))),duality_gap=gap,marginal_error=marginal,dual_violation=violation,solver='network simplex; fixed row/column permutation; 50M iteration cap')

def angular_job(task):
    c,eta,count=task
    path=DATA/'angular'/key(c)/f'eta{eta:.6f}_n{count}.json'
    if path.exists():return json.loads(path.read_text())
    z=points('circle',c['n'],c['amplitude'],c['sampling'],1000+c['rep'])
    site=np.arctan2(z[:,1],z[:,0]);theta=np.linspace(0,2*np.pi,131073)
    # Avoid positive-exponential products. Opposite-direction cancellation is
    # harmless after a nonnegative clip and an explicit normalization check.
    delta=theta[:,None]-site;aa=2*np.cos(delta)/eta
    density=np.maximum((np.exp(-2/(eta*eta))/(2*np.pi)+aa/np.sqrt(2*np.pi)*np.exp(-2*np.sin(delta)**2/(eta*eta))*ndtr(aa)).mean(1),0.)
    cumulative=np.r_[0.,cumulative_trapezoid(density,theta)]
    mass=float(cumulative[-1]);assert abs(mass-1)<1e-8,(key(c),eta,mass)
    cumulative/=mass
    q=np.interp((np.arange(count)+.5)/count,cumulative,theta)
    source=2*np.column_stack([np.cos(q),np.sin(q)])
    target=points('circle',count,c['amplitude'],'target',777)
    row=dict(configuration=c,eta=eta,count=count,mass=mass,**exact_ot(source,target))
    dump(path,row);return row

def main():
    rows=[json.loads(p.read_text()) for p in (DATA/'rows').glob('*/*.json')]
    tasks={}
    for r in rows:
        if r['shape']=='circle':
            c={k:r[k] for k in ['shape','n','amplitude','sampling','rep']}
            tasks[(key(c),r['eta'],r['target_nodes'])]=(c,r['eta'],r['target_nodes'])
    lookup={}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(angular_job,t) for t in tasks.values()]):
            r=f.result();lookup[(key(r['configuration']),r['eta'],r['count'])]=r
    corrected=[]
    for row in rows:
        r=dict(row)
        if r['shape'] in ['circle','sphere']:
            d=2 if r['shape']=='circle' else 3;eta=r['eta'];radius=2.
            mean_radius=eta*np.sqrt(2)*np.exp(gammaln((d+1)/2)-gammaln(d/2))*hyp1f1(-.5,d/2,-radius*radius/(2*eta*eta))
            normal2=float(2*radius*radius+d*eta*eta-2*radius*mean_radius)
            assert normal2>=0 and np.isfinite(normal2)
            residual=r['raw_w2']**2-r['raw_wn']**2
            assert residual>=-1e-8
            r.update(raw_w2_discrete=r['raw_w2'],raw_wn_qmc=r['raw_wn'],raw_w2=float(np.sqrt(normal2+max(0.,residual))),raw_wn=float(np.sqrt(normal2)))
        if r['shape']=='circle':
            a=lookup[(key(r),r['eta'],r['target_nodes'])]
            r.update(raw_wt_qmc=r['raw_wt'],raw_wt=a['distance'],angular_mass=a['mass'],angular_ot_gap=a['duality_gap'])
        r['triangle_numerical_excess']=float(max(0.,r['raw_w2']-r['raw_wt']-r['raw_wn']))
        corrected.append(r)
    dump(DATA/'corrected_rows.json',corrected)
    dump(ROOT/'geometry_refinement_status.json',dict(state='complete_pending_review',rows=len(corrected),angular_jobs=len(tasks),
              max_triangle_numerical_excess=max(r['triangle_numerical_excess'] for r in corrected),finished=time.time()))
    print((ROOT/'geometry_refinement_status.json').read_text(),flush=True)

if __name__=='__main__':main()
