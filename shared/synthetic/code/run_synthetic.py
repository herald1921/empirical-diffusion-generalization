"""Controlled manifold extensions. Every displayed distance is an audited discrete OT.

Run: python run_synthetic.py --smoke; python run_synthetic.py --workers 8
No assertion of a continuous-law sign is made by this program: independent quadrature
and doubled-resolution results are saved for the manuscript's numerical audit.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse, hashlib, json, math, pathlib, platform, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import scipy
from scipy.special import ndtri, softmax
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.stats import qmc
import ot
import gudhi

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ETA = np.array([.008,.015,.025,.04,.06,.08,.11,.15,.20,.28,.38,.52,.72,1.,1.4,2.])
REFINE_ETA = [.025,.06,.11,.20,.38,.72,1.4]

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def angles(u, amplitude):
    theta = 2*np.pi*u
    for _ in range(15):
        theta -= (theta + amplitude*np.sin(theta) - 2*np.pi*u)/(1+amplitude*np.cos(theta))
    assert np.max(np.abs(theta+amplitude*np.sin(theta)-2*np.pi*u)) < 1e-12
    return theta

def points(shape, count, amplitude, mode, seed):
    rng = np.random.default_rng(seed)
    if shape != 'sphere':
        u = rng.random(count) if mode == 'iid' else (np.arange(count)+.5)/count
        theta = angles(u, amplitude)
        axes = np.array([2., 1. if shape == 'ellipse' else 2.])
        return np.column_stack([np.cos(theta),np.sin(theta)])*axes
    if mode == 'iid':
        z = rng.normal(size=(count,3))
        return 2*z/np.linalg.norm(z,axis=1,keepdims=True)
    if mode == 'target':
        u = qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.log2(count)))
        z, theta = 1-2*u[:,0], 2*np.pi*u[:,1]
    else:
        z = 1-2*(np.arange(count)+.5)/count
        theta = np.arange(count)*np.pi*(3-np.sqrt(5))
    r = np.sqrt(1-z*z)
    return 2*np.column_stack([r*np.cos(theta),r*np.sin(theta),z])

def project(x, shape):
    if shape != 'ellipse':
        r = np.linalg.norm(x,axis=1,keepdims=True)
        answer = 2*x/np.maximum(r,1e-300)
        answer[r[:,0]==0] = np.r_[2.,np.zeros(x.shape[1]-1)]
        return answer
    # Lagrange multiplier for the closest point on (x/2)^2+y^2=1.
    ax2 = np.array([4.,1.])
    y = np.abs(x)
    off_axis = y[:,1] > 1e-14
    ans = np.zeros_like(x)
    if off_axis.any():
        v = y[off_axis]
        lo = np.full(len(v),np.nextafter(-1.,0.))
        hi = np.maximum(1.,2*np.linalg.norm(v,axis=1))
        for _ in range(85):
            mid = .5*(lo+hi)
            residual = np.sum(ax2*v*v/(ax2+mid[:,None])**2,axis=1)-1
            lo = np.where(residual>0,mid,lo)
            hi = np.where(residual>0,hi,mid)
        ans[off_axis] = ax2*v/(ax2+(.5*(lo+hi))[:,None])
    # The medial-axis branch has two equally near points; choose positive y.
    if (~off_axis).any():
        u = y[~off_axis,0]
        px = np.minimum(2.,4*u/3)
        ans[~off_axis] = np.column_stack([px,np.sqrt(np.maximum(0.,1-px*px/4))])
    signs = np.where(x<0,-1.,1.)
    return ans*signs

def noisy_nodes(train, count, seed):
    per_atom = count//len(train)
    assert per_atom >= 2 and per_atom*len(train) == count
    assert 2**round(np.log2(per_atom)) == per_atom
    blocks = []
    for i, z in enumerate(train):
        u = qmc.Sobol(train.shape[1],scramble=True,seed=seed+101*i).random_base2(int(np.log2(per_atom))-1)
        noise = ndtri(np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps))
        blocks.append(np.r_[noise,-noise])
    return np.repeat(train,per_atom,axis=0),np.vstack(blocks)

def exact_ot(x,y):
    a,b = np.full(len(x),1/len(x)),np.full(len(y),1/len(y))
    costs = np.ascontiguousarray(cdist(x,y,'sqeuclidean'))
    plan, log = ot.emd(a,b,costs,log=True,numItermax=2000000,numThreads=1,check_marginals=True)
    primal = float(np.einsum('ij,ij->',plan,costs))
    gap = abs(primal-float(a@log['u']+b@log['v']))
    marginal = max(float(np.max(np.abs(plan.sum(1)-a))),float(np.max(np.abs(plan.sum(0)-b))))
    violation = max(0.,float(np.max(log['u'][:,None]+log['v'][None,:]-costs)))
    if log.get('warning') or gap>2e-8*max(1.,primal) or marginal>2e-9 or violation>2e-8:
        raise RuntimeError(dict(warning=log.get('warning'),gap=gap,marginal=marginal,violation=violation))
    return dict(distance=float(np.sqrt(max(0.,primal))),duality_gap=gap,marginal_error=marginal,dual_violation=violation)

def configurations():
    configs=[]
    for shape in ['circle','ellipse','sphere']:
        for n in [8,16,32,64]:
            configs.append(dict(shape=shape,n=n,amplitude=.1 if shape!='sphere' else 0.,sampling='design',rep=0))
    for amplitude in [0.,.4]:
        configs.append(dict(shape='circle',n=16,amplitude=amplitude,sampling='design',rep=0))
    for shape,repeats,n in [('circle',8,16),('ellipse',8,16),('sphere',8,32)]:
        for rep in range(repeats):
            configs.append(dict(shape=shape,n=n,amplitude=.1 if shape!='sphere' else 0.,sampling='iid',rep=rep))
    return configs

def key(c):
    return f"{c['shape']}_N{c['n']}_a{c['amplitude']:.1f}_{c['sampling']}_r{c['rep']}"

def topology(train,shape):
    st=gudhi.AlphaComplex(points=train.tolist(),precision='exact').create_simplex_tree()
    st.persistence(min_persistence=1e-12)
    intervals={str(k):[[float(np.sqrt(max(0,b))),float(np.sqrt(d)) if np.isfinite(d) else None]
                        for b,d in st.persistence_intervals_in_dimension(k)] for k in range(train.shape[1])}
    if shape=='sphere':
        # Largest empty spherical cap from supporting facets of the convex hull.
        equations=ConvexHull(train).equations
        plane_distance=-equations[:,-1]
        cover=float(np.sqrt(max(0.,4*(2-plane_distance.min()))))
        certified=True
    else:
        axes=np.array([2.,1. if shape=='ellipse' else 2.])
        theta=np.sort(np.mod(np.arctan2(train[:,1]/axes[1],train[:,0]/axes[0]),2*np.pi))
        max_gap=float(np.diff(np.r_[theta,theta[0]+2*np.pi]).max())
        cover=float(4*np.sin(max_gap/4)) if shape=='circle' else min(4.,max_gap)
        certified=True
    return dict(intervals_radius=intervals,cover_radius_or_upper_bound=cover,cover_is_upper_bound=certified,
                cover_type='exact spherical cap' if shape=='sphere' else 'exact chord' if shape=='circle' else 'arc-speed upper bound',
                reach=.5 if shape=='ellipse' else 2.,dimension=train.shape[1],
                c_delta=float(np.sqrt(train.shape[1])+np.sqrt(2*np.log(20))),delta=.05)

def solve(task):
    c, eta, resolution, scramble = task
    suffix=f"eta{eta:.6f}_{resolution}_q{scramble}"
    path=DATA/'rows'/key(c)/(suffix+'.json')
    if path.exists():return json.loads(path.read_text())
    started=time.time()
    count,target_count=(1024,2048) if resolution=='base' else (2048,4096)
    train=points(c['shape'],c['n'],c['amplitude'],c['sampling'],1000+c['rep'])
    target=points(c['shape'],target_count,c['amplitude'],'target',777+scramble)
    z,g=noisy_nodes(train,count,2000+17*c['rep']+100000*scramble)
    x=z+eta*g
    post=softmax(-cdist(x,train,'sqeuclidean')/(2*eta*eta),axis=1)@train
    xp,mp=project(x,c['shape']),project(post,c['shape'])
    raw,tangent,repair,repair_tangent=[exact_ot(a,target) for a in [x,xp,post,mp]]
    normal=float(np.sqrt(np.mean(np.sum((x-xp)**2,axis=1))))
    repair_normal=float(np.sqrt(np.mean(np.sum((post-mp)**2,axis=1))))
    for total,tang,norm in [(raw,tangent,normal),(repair,repair_tangent,repair_normal)]:
        assert norm <= total['distance']+1e-7
        assert total['distance'] <= tang['distance']+norm+1e-7
    row=dict(**c,eta=float(eta),reverse_t=float(2/np.pi*np.arctan(1/eta)),resolution=resolution,scramble=scramble,
             source_nodes=count,target_nodes=target_count,raw_w2=raw['distance'],raw_wt=tangent['distance'],raw_wn=normal,
             tweedie_w2=repair['distance'],tweedie_wt=repair_tangent['distance'],tweedie_wn=repair_normal,
             oracle_w2=tangent['distance'],oracle_wt=tangent['distance'],oracle_wn=0.,
             ot_max_duality_gap=max(v['duality_gap'] for v in [raw,tangent,repair,repair_tangent]),
             ot_max_marginal_error=max(v['marginal_error'] for v in [raw,tangent,repair,repair_tangent]),
             ot_max_dual_violation=max(v['dual_violation'] for v in [raw,tangent,repair,repair_tangent]),seconds=time.time()-started)
    dump(path,row)
    return row

def smoke():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(1000,2))*3
    p=project(x,'ellipse')
    surface_residual=float(np.max(np.abs(p[:,0]**2/4+p[:,1]**2-1)))
    assert surface_residual<1e-9
    # Cross-check the projector against a dense global boundary search, including medial axis.
    axis=np.array([[0.,0.],[.4,0.],[1.49,0.],[1.51,0.],[2.1,0.],[-.4,0.]])
    check=np.r_[x[:30],axis]
    t=np.linspace(0,2*np.pi,200001)
    mesh=np.column_stack([2*np.cos(t),np.sin(t)])
    exact=np.sum((check-project(check,'ellipse'))**2,axis=1)
    dense=cdist(check,mesh,'sqeuclidean').min(1)
    assert np.max(np.abs(exact-dense))<1e-8
    circle=points('circle',16,0.,'design',0)
    topo=topology(circle,'circle')
    h1=max(topo['intervals_radius']['1'],key=lambda a:a[1]-a[0])
    assert abs(h1[0]-2*np.sin(np.pi/16))<1e-9 and abs(h1[1]-2)<1e-9
    self_distance=exact_ot(circle,circle)['distance']
    assert self_distance<1e-10
    r=solve((dict(shape='circle',n=16,amplitude=.1,sampling='design',rep=0),.20,'base',0))
    dump(ROOT/'audit_smoke.json',dict(passed=True,ellipse_residual=surface_residual,ellipse_dense_max_error=float(np.max(np.abs(exact-dense))),
                                    circle_alpha_h1=h1,ot_self=self_distance,transport_bounds=True,seconds_one_scale=r['seconds']))
    print(json.dumps(json.loads((ROOT/'audit_smoke.json').read_text()),indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    if args.smoke:return smoke()
    import fcntl
    lock=(ROOT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'audit_smoke.json').read_text())['passed']
    signature=dict(code_sha256=sha(__file__),eta=ETA.tolist(),base_source=1024,base_target=2048,fine_source=2048,fine_target=4096,
                   python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,POT=ot.__version__,gudhi=gudhi.__version__)
    signature_path=ROOT/'protocol.json'
    if signature_path.exists():assert json.loads(signature_path.read_text())==signature,'Changed source or protocol: use a fresh output directory.'
    dump(signature_path,signature)
    configs=configurations();tasks=[]
    for c in configs:
        train=points(c['shape'],c['n'],c['amplitude'],c['sampling'],1000+c['rep'])
        folder=DATA/'designs'/key(c);folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'training.npy',train)
        detail=dict(**c,**topology(train,c['shape']),training_sha256=sha(folder/'training.npy'))
        detail['endpoint']={}
        for count in [2048,4096]:
            target=points(c['shape'],count,c['amplitude'],'target',777)
            detail['endpoint'][str(count)]=exact_ot(train,target)
        dump(folder/'metadata.json',detail)
        tasks.extend((c,float(e),'base',0) for e in ETA)
        # All three geometries: resolution and independent-integration checks for
        # the main deterministic setting and two independent training draws.
        selected=c['n']==(32 if c['shape']=='sphere' else 16) and c['amplitude'] in [0.,.1] and c['rep']<=1
        if selected:
            for e in REFINE_ETA:
                tasks.extend([(c,e,'fine',0),(c,e,'base',1)])
    dump(ROOT/'tasks.json',[dict(config=c,eta=e,resolution=r,scramble=s) for c,e,r,s in tasks])
    complete=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(solve,task) for task in tasks]
        for f in as_completed(futures):
            complete.append(f.result())
            dump(ROOT/'status.json',dict(state='computing',completed=len(complete),total=len(tasks),pid=os.getpid()))
            if len(complete)%16==0:print('COMPLETED',len(complete),'/',len(tasks),flush=True)
    dump(DATA/'all_rows.json',sorted(complete,key=lambda r:(key(r),r['resolution'],r['scramble'],r['eta'])))
    dump(ROOT/'status.json',dict(state='complete_pending_review',completed=len(complete),total=len(tasks),pid=os.getpid(),finished=time.time()))

if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'status.json',dict(state='error',pid=os.getpid(),traceback=traceback.format_exc()));raise
