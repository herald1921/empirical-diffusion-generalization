"""Figure 2: distributional error, normal/tangential mechanisms, dimension.

python make_figure_02.py compute --workers 2
python make_figure_02.py plot
python make_figure_02.py all --workers 2

Uses the exact same 16 training points and cosine-VP reverse time as Figure 1.
OT is unregularized Euclidean W2, not sliced or entropic transport.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse,csv,hashlib,json,platform,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
from pathlib import Path
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.special import ndtri,ndtr,i0e,i1e
from scipy.stats import qmc
from scipy.integrate import quad,cumulative_trapezoid
import ot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
OUT=ROOT/'output'
CONTEXT=json.loads((DATA/'figure_01_context.json').read_text())
CFG=CONTEXT['config']
R=CFG['radius']
AMP=CFG['angular_amplitude']
POINTS=np.load(DATA/'training_points.npy')
THETA=np.mod(np.arctan2(POINTS[:,1],POINTS[:,0]),2*np.pi)
SNAPS=np.array([CFG['eta_high'],CFG['eta_middle'],CFG['eta_low']])
RECOVERY=CONTEXT['time_convention']['theory_time_interval']
COL=dict(raw='#0072B2',tangent='#009E73',normal='#D55E00',bound='#A3ACB4',
         baseline='#46525D',dim8='#8B62AA')


def t_of_eta(eta):
    return 2/np.pi*np.arctan2(1.,np.asarray(eta))


def target_quantile(u):
    theta=2*np.pi*np.asarray(u)
    for _ in range(12):
        theta-=(theta+AMP*np.sin(theta)-2*np.pi*u)/(1+AMP*np.cos(theta))
    return theta


@lru_cache(None)
def target_design(count):
    theta=target_quantile((np.arange(count)+.5)/count)
    return R*np.column_stack([np.cos(theta),np.sin(theta)])


@lru_cache(None)
def normal_nodes(power,seed):
    # Exactly antithetic normals remove first-moment quadrature bias.
    u=qmc.Sobol(2,scramble=True,seed=seed).random_base2(power-1)
    z=ndtri(np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps))
    return np.concatenate([z,-z])


def noisy_nodes(eta,power,seed):
    z=normal_nodes(power,seed)
    c,s=np.cos(THETA)[:,None],np.sin(THETA)[:,None]
    noise=np.stack([c*z[None,:,0]-s*z[None,:,1],s*z[None,:,0]+c*z[None,:,1]],axis=-1)
    return (POINTS[:,None,:]+eta*noise).reshape(-1,2)


def project(x):
    norm=np.linalg.norm(x,axis=1)
    if np.any(norm==0):
        raise ValueError('Undefined circle projection at origin.')
    return R*x/norm[:,None]


def normal_exact(eta):
    """Exact normal RMS from the Rice radial mean, valid for this circle."""
    eta=float(eta)
    if eta==0:
        return 0.
    z=R*R/(4*eta*eta)
    mean=eta*np.sqrt(np.pi/2)*((1+2*z)*i0e(z)+2*z*i1e(z))
    return float(np.sqrt(max(0.,2*R*R+2*eta*eta-2*R*mean)))


def exact_ot(x,y):
    a=np.full(len(x),1/len(x)); b=np.full(len(y),1/len(y))
    costs=np.ascontiguousarray(cdist(x,y,'sqeuclidean'))
    plan,log=ot.emd(a,b,costs,log=True,numItermax=2000000,numThreads=1,check_marginals=True)
    primal=float(np.einsum('ij,ij->',plan,costs))
    gap=abs(primal-float(a@log['u']+b@log['v']))
    marginal=float(max(abs(plan.sum(1)-a).max(),abs(plan.sum(0)-b).max()))
    violation=0.
    for start in range(0,len(x),256):
        violation=max(violation,float(np.max(log['u'][start:start+256,None]+log['v'][None,:]-costs[start:start+256])))
    if log.get('warning') or gap>2e-8*max(1.,primal) or marginal>2e-9 or violation>2e-8:
        raise RuntimeError(dict(warning=log.get('warning'),gap=gap,marginal=marginal,violation=violation))
    return np.sqrt(max(0.,primal)),gap,marginal,violation


def solve_one(task):
    eta,power,target_count,seed,level=task
    start=time.perf_counter()
    x=noisy_nodes(eta,power,seed)
    y=target_design(target_count)
    raw=exact_ot(x,y)
    tangent=exact_ot(project(x),y)
    wn_qmc=float(np.sqrt(np.mean((np.linalg.norm(x,axis=1)-R)**2)))
    if raw[0]+1e-8<wn_qmc or raw[0]>tangent[0]+wn_qmc+1e-8:
        raise RuntimeError('Discrete transport bound failed.')
    # |x-y|^2=(|x|-R)^2 + 2*|x|*R*(1-cos(angle(x)-angle(y))).
    # The first term is independent of the transport plan. Integrate it
    # exactly as a control variate; keep the uncorrected LP value as well.
    wn_exact=normal_exact(eta)
    residual=max(0.,float(raw[0]**2-wn_qmc**2))
    corrected=float(np.sqrt(wn_exact**2+residual))
    if corrected > tangent[0]+wn_exact+1e-6:
        raise RuntimeError('Corrected estimate exceeds displayed upper bound.')
    return dict(eta=float(eta),t=float(t_of_eta(eta)),level=level,seed=seed,
                power=power,source_nodes=len(x),target_nodes=target_count,
                w2=corrected,w2_discrete=float(raw[0]),angular_residual_cost=residual,
                wt=float(tangent[0]),wn=wn_exact,wn_qmc=wn_qmc,
                max_duality_gap=max(raw[1],tangent[1]),max_marginal_error=max(raw[2],tangent[2]),
                max_dual_violation=max(raw[3],tangent[3]),seconds=time.perf_counter()-start)


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for k in row:
            if k!='level': row[k]=float(row[k])
    return rows


def compute_batch(tasks,path,workers):
    rows=read_csv(path)
    done={(round(r['eta'],12),r['level'],int(r['seed'])) for r in rows}
    todo=[t for t in tasks if (round(t[0],12),t[-1],t[-2]) not in done]
    print(f'{path.name}: {len(todo)} remaining scales',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(solve_one,t) for t in todo]
        for future in as_completed(futures):
            row=future.result(); rows.append(row)
            rows.sort(key=lambda r:(r['level'],r['seed'],r['t']))
            write_csv(path,rows)
            if len(rows)%8==0 or row['eta'] in SNAPS:
                print(json.dumps(dict(level=row['level'],complete=len(rows),eta=row['eta'],w2=row['w2'],wt=row['wt']),ensure_ascii=True),flush=True)
    return rows


def angular_ot_worker(task):
    eta,count=task
    angles=np.linspace(0,2*np.pi,131073)
    density=angular_density(angles,eta)
    cumulative=np.r_[0.,cumulative_trapezoid(density,angles)]
    mass=float(cumulative[-1])
    if abs(mass-1)>1e-8:
        raise RuntimeError('Angular CDF normalization failed.')
    cumulative/=mass
    theta=np.interp((np.arange(count)+.5)/count,cumulative,angles)
    source=R*np.column_stack([np.cos(theta),np.sin(theta)])
    result=exact_ot(source,target_design(count))
    return dict(eta=eta,target_nodes=count,wt=result[0],duality_gap=result[1],
                marginal_error=result[2],dual_violation=result[3],cdf_mass=mass)


def complete_angular_transport(rows,audits,workers):
    path=DATA/'angular_transport.csv'
    cache=read_csv(path)
    done={(round(r['eta'],12),int(r['target_nodes'])) for r in cache}
    required={(round(r['eta'],12),int(r['target_nodes'])) for r in rows+audits}
    todo=sorted(required-done)
    print(f'Analytic-angular transport: {len(todo)} remaining scales/resolutions',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(angular_ot_worker,t) for t in todo]):
            cache.append(future.result())
            cache.sort(key=lambda r:(r['eta'],r['target_nodes']))
            write_csv(path,cache)
            if len(cache)%15==0: print(f'Angular transport completed: {len(cache)}',flush=True)
    indexed={(round(r['eta'],12),int(r['target_nodes'])):r for r in cache}
    for r in rows+audits:
        r.setdefault('wt_projected_qmc',r['wt'])
        r['wt']=indexed[(round(r['eta'],12),int(r['target_nodes']))]['wt']
        if r['w2']>r['wn']+r['wt']+1e-6:
            raise RuntimeError('Final estimates fail displayed triangle inequality.')
    write_csv(DATA/'distances.csv',rows)
    write_csv(DATA/'quadrature_checks.csv',audits)
    return cache


def compute(workers):
    signature=dict(training_sha256=hashlib.sha256((DATA/'training_points.npy').read_bytes()).hexdigest(),
                   radius=R,amplitude=AMP,method='normal-control-variate + analytic-angular-OT-v1')
    signature_path=DATA/'cache_signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text())!=signature:
        raise RuntimeError('Cached results use a different experiment. Use a new data directory.')
    signature_path.write_text(json.dumps(signature,indent=2),encoding='utf-8')
    grid=np.unique(np.r_[np.geomspace(.002,2.4,35),np.arange(.04,.205,.005),SNAPS,
                         [.06,.08,.1,.12,.14,.2,.35],
                         1/np.tan(np.pi*np.linspace(.255,.86,32)/2),
                         CONTEXT['geometry']['theory_eta_lower'],CONTEXT['geometry']['theory_eta_upper']])
    # Round away duplicate floating-point values before caching/resumption.
    grid=np.unique(np.round(grid,12))
    rows=compute_batch([(float(e),7,2048,20260920,'main') for e in grid],DATA/'distances.csv',workers)
    ordered=sorted(rows,key=lambda r:r['eta'])
    best=min(range(len(ordered)),key=lambda i:ordered[i]['w2'])
    checks=np.unique(np.r_[SNAPS,[.06,.08,.12,.2,.35],
                            [ordered[i]['eta'] for i in range(max(0,best-1),min(len(ordered),best+2))]])
    tasks=[(float(e),8,4096,20260920,'refined') for e in checks]
    tasks += [(float(e),7,2048,20260921,'independent_scramble') for e in checks]
    audits=compute_batch(tasks,DATA/'quadrature_checks.csv',workers)
    angular_checks=complete_angular_transport(rows,audits,workers)
    baselines={str(n):float(exact_ot(POINTS,target_design(n))[0]) for n in (2048,4096,8192)}
    comparisons=[]
    for r in audits:
        base=min(rows,key=lambda z:abs(z['eta']-r['eta']))
        if abs(base['eta']-r['eta'])>1e-10:
            raise RuntimeError('Audit scale missing from main grid.')
        comparisons.append(dict(level=r['level'],eta=r['eta'],delta_w2=r['w2']-base['w2'],delta_wt=r['wt']-base['wt']))
    endpoint=baselines['8192']
    chosen=min(rows,key=lambda r:r['w2'])
    dim8=min(rows,key=lambda r:np.sqrt(r['w2']**2+6*r['eta']**2))
    certificate=2*np.pi*R/(np.sqrt(12)*(1-abs(AMP))*2048)
    report=dict(target='(1+0.1*cos(theta))/(2*pi), radius 2',training_points=len(POINTS),
                coordinate='U=X/alpha(t), cosine VP reverse time',
                phase_snapshots=CONTEXT['snapshots'],topology_recovery_time=RECOVERY,
                empirical_endpoint=endpoint,baseline_resolution=baselines,
                main_target_discretization_w2_bound=certificate,
                source_quadrature_error='Sensitivity checked; no rigorous source W2 error bound claimed.',
                total_estimator='Normal-cost control variate: W2_hat^2=WN_exact^2+W2_discrete^2-WN_discrete^2',
                tangential_estimator='Analytic angular CDF midpoint quantiles and unregularized Euclidean OT',
                max_angular_cdf_mass_error=max(abs(r['cdf_mass']-1) for r in angular_checks),
                main_grid_count=len(rows),checks=comparisons,
                max_refined_w2_change=max(abs(r['delta_w2']) for r in comparisons if r['level']=='refined'),
                max_refined_wt_change=max(abs(r['delta_wt']) for r in comparisons if r['level']=='refined'),
                main_sampled_minimum=dict(eta=chosen['eta'],t=chosen['t'],w2=chosen['w2'],wt=chosen['wt'],wn=chosen['wn'],gain_fraction=1-chosen['w2']/endpoint),
                dim8_sampled_minimum=dict(eta=dim8['eta'],t=dim8['t'],w2=float(np.sqrt(dim8['w2']**2+6*dim8['eta']**2))),
                max_ot_duality_gap=max(r['max_duality_gap'] for r in rows+audits),
                max_ot_marginal_error=max(r['max_marginal_error'] for r in rows+audits),
                max_ot_dual_violation=max(r['max_dual_violation'] for r in rows+audits),
                training_points_sha256=hashlib.sha256((DATA/'training_points.npy').read_bytes()).hexdigest(),
                software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,POT=ot.__version__,matplotlib=matplotlib.__version__))
    (DATA/'numerical_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


def angular_density(theta,eta):
    """Exact angular marginal of the 2D Gaussian mixture, integrated radially.

    The formula uses a nonpositive exponential and avoids exp(a*a/2).
    """
    delta=np.asarray(theta)[...,None]-THETA
    a=R*np.cos(delta)/eta
    constant=np.exp(-R*R/(2*eta*eta))/(2*np.pi)
    density=constant+a/np.sqrt(2*np.pi)*np.exp(-R*R*np.sin(delta)**2/(2*eta*eta))*ndtr(a)
    return np.maximum(density.mean(axis=-1),0.)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'mathtext.fontset':'dejavusans',
                        'axes.labelsize':8,'axes.edgecolor':'#BCC5CC','axes.linewidth':.65,
                        'xtick.labelsize':7,'ytick.labelsize':7,'xtick.color':'#62717D','ytick.color':'#62717D',
                        'text.color':'#253340','axes.labelcolor':'#455561','pdf.fonttype':42,
                        'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def decorate(ax,time_axis=True,shade=False):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5,pad=2)
    ax.grid(axis='y',color='#EDF0F2',lw=.5,zorder=0)
    if time_axis: ax.set_xlabel(r'Reverse diffusion time $t$')
    if shade:
        ax.axvspan(*RECOVERY,color='#EFF0F2',lw=0,zorder=0)
        for t in RECOVERY: ax.axvline(t,color='#B3BAC0',lw=.65,ls=(0,(3,3)),zorder=1)


def panel_title(fig,x,y,letter,title):
    fig.text(x,y,letter,weight='bold',size=9.5)
    fig.text(x+.032,y,title,weight='bold',size=9)


def plot():
    style(); OUT.mkdir(exist_ok=True)
    audit=json.loads((DATA/'numerical_audit.json').read_text())
    rows=sorted(read_csv(DATA/'distances.csv'),key=lambda r:r['t'])
    eta=np.array([r['eta'] for r in rows]+[0.])
    t=np.r_[[r['t'] for r in rows],1.]
    sn=audit['empirical_endpoint']
    raw=np.r_[[r['w2'] for r in rows],sn]
    wt=np.r_[[r['wt'] for r in rows],sn]
    wn=np.r_[[r['wn'] for r in rows],0.]
    fig=plt.figure(figsize=(6.75,5.9))
    a=fig.add_axes([.091,.563,.386,.335])
    b=fig.add_axes([.586,.563,.385,.335])
    c=fig.add_axes([.091,.103,.386,.31])
    d=fig.add_axes([.586,.103,.385,.31])
    panel_title(fig,.059,.927,'a','Distribution error over time')
    panel_title(fig,.555,.927,'b','Late-time trade-off')
    panel_title(fig,.059,.444,'c','Angular concentration')
    panel_title(fig,.555,.444,'d','Effect of ambient dimension')
    handles=[Line2D([],[],color=COL['raw'],lw=1.5),Line2D([],[],color=COL['tangent'],lw=1.5),
             Line2D([],[],color=COL['normal'],lw=1.5),Line2D([],[],color=COL['bound'],lw=1,ls='--'),
             Line2D([],[],color=COL['baseline'],lw=1,ls=(0,(4,2)))]
    fig.legend(handles,[r'Total $W_2$',r'Tangential $W_T$',r'Normal $W_N$',r'$W_T+W_N$',r'Endpoint $S_N$'],
               loc='upper center',bbox_to_anchor=(.53,.997),ncol=5,frameon=False,fontsize=7.2,
               handlelength=1.7,columnspacing=1.05,handletextpad=.5)
    for ax,norm in [(a,R),(b,sn)]:
        decorate(ax,shade=True)
        ax.plot(t,(wt+wn)/norm,color=COL['bound'],lw=1.05,ls=(0,(4,2)),zorder=2)
        ax.plot(t,raw/norm,color=COL['raw'],lw=1.6,zorder=4)
        ax.plot(t,wt/norm,color=COL['tangent'],lw=1.4,zorder=3)
        mask=wn>0 if ax is a else np.ones(len(wn),dtype=bool)
        ax.plot(t[mask],wn[mask]/norm,color=COL['normal'],lw=1.4,zorder=3)
        ax.axhline(sn/norm,color=COL['baseline'],lw=.95,ls=(0,(4,2)),zorder=1)
    a.set_yscale('log')
    positive=np.r_[wt[wt>0]/R,wn[wn>0]/R]
    a.set_ylim(max(1e-5,.6*positive.min()),2.0); a.set_xlim(.25,1)
    a.set_xticks([.25,.5,.75,1],['0.25','0.50','0.75','1'])
    a.set_ylabel(r'Distance / $R$')
    a.text(np.mean(RECOVERY),.97,'Topology recovery\ninterval',ha='center',va='top',
           transform=a.get_xaxis_transform(),size=6.5,color='#7A8289')
    for e,tag in zip(SNAPS,['I','II','III']):
        i=np.argmin(abs(eta-e))
        a.scatter(t[i],raw[i]/R,color=COL['raw'],s=17,edgecolors='white',lw=.45,zorder=5)
        a.annotate(tag,(t[i],raw[i]/R),xytext=(-2,7),textcoords='offset points',size=7,ha='center')
    b.set_xlim(.84,1.003); b.set_ylim(0,1.65)
    b.set_xticks([.85,.9,.95,1],['0.85','0.90','0.95','1'])
    b.set_ylabel(r'Distance / $S_N$')
    i=np.argmin(raw)
    b.scatter(t[i],raw[i]/sn,color=COL['raw'],s=23,edgecolors='white',lw=.6,zorder=6)
    b.annotate('Sampled minimum',(t[i],raw[i]/sn),xytext=(.851,.61),textcoords='data',
               fontsize=7,color=COL['raw'],arrowprops=dict(arrowstyle='-',lw=.65,color=COL['raw']))
    # Angular densities are evaluated analytically, not estimated from 512 display particles.
    theta=np.linspace(-np.arcsin(.26),np.arcsin(.26),2401)
    target=(1+AMP*np.cos(theta))/(2*np.pi)
    decorate(c,time_axis=False)
    phase_colors=['#8D979F',COL['tangent'],COL['dim8']]
    for e,tag,color in zip(SNAPS,['I','II','III'],phase_colors):
        c.plot(theta,angular_density(theta,e),color=color,lw=1.3,label=f'Phase {tag}')
    c.plot(theta,target,color='#253340',lw=1.1,ls=(0,(4,2)),label='Target')
    local=THETA.copy(); local[local>np.pi]-=2*np.pi
    for th in local[abs(local)<theta.max()]:
        c.axvline(th,color='#BDC5CB',lw=.65,ls=(0,(2,3)),zorder=0)
    c.set_yscale('log'); c.set_ylim(.003,10); c.set_xlim(theta.min(),theta.max())
    c.set_xticks([-.2,0,.2]); c.set_ylabel(r'Angular density (rad$^{-1}$)')
    c.set_xlabel(r'Local angle $\theta$ (rad)')
    c.legend(loc='lower center',ncol=2,frameon=False,fontsize=6.7,columnspacing=.9,handlelength=1.6)
    decorate(d)
    for dim,color in [(2,COL['raw']),(8,COL['dim8'])]:
        total=np.sqrt(raw*raw+(dim-2)*eta*eta)
        d.plot(t,total/sn,color=color,lw=1.6,label=rf'$D={dim}$')
        k=np.argmin(total)
        d.scatter(t[k],total[k]/sn,s=23,color=color,edgecolors='white',lw=.6,zorder=5)
    d.axhline(1,color=COL['baseline'],lw=.95,ls=(0,(4,2)))
    d.set_xlim(.84,1.003); d.set_ylim(.55,2.05)
    d.set_xticks([.85,.9,.95,1],['0.85','0.90','0.95','1'])
    d.set_ylabel(r'Total $W_{2,D}$ / $S_N$')
    d.legend(loc='upper right',frameon=False,fontsize=7)
    stem=OUT/'figure_02_distribution_geometry'
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':'Distributional error and geometric mechanisms of empirical diffusion'})
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=360)
    plt.close(fig)
    # Independent checks of the analytic angular marginal (whole circle, no truncation).
    integral_checks={}
    integration_points=np.sort(np.unique(np.r_[0.,THETA,2*np.pi]))
    for e in SNAPS:
        val=quad(lambda angle:float(angular_density(angle,e)),0,2*np.pi,
                 points=integration_points[1:-1],epsabs=2e-9,limit=800)[0]
        if abs(val-1)>2e-6: raise RuntimeError(f'Angular density normalization failed: {e} {val}')
        integral_checks[str(e)]=val
    (DATA/'angular_density_checks.json').write_text(json.dumps(integral_checks,indent=2),encoding='utf-8')
    print('Figure saved:',stem,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['compute','plot','all'])
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.mode in ('compute','all'): compute(args.workers)
    if args.mode in ('plot','all'): plot()


if __name__=='__main__':
    main()
