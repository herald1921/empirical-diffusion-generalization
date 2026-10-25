"""Figures 4/5: normal reduction and paired distributional repair.
python make_figures_04_05.py compute --workers 2
python make_figures_04_05.py plot
python make_figures_04_05.py all --workers 2
All paths relative to this file. Frozen Figure 2 inputs under reference/.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import sys,csv,json,argparse,time,hashlib,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import scipy
from scipy.special import softmax
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import ot
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
import figure02_core as core
DATA=ROOT/'data'; OUT=ROOT/'output'
R=core.R; P=core.POINTS; N=len(P)
PAIR=cdist(P,P,'sqeuclidean')
SN=float(json.loads((ROOT/'reference/data/numerical_audit.json').read_text())['empirical_endpoint'])
SEED=20260920
COL={'raw':'#0072B2','ideal':'#009E73','tweedie':'#D55E00','polygon':'#8B62AA','bound':'#7D8993'}
LABEL={'raw':'Raw early stop','ideal':'Ideal projection','tweedie':'Empirical Tweedie','polygon':'Radial polygon repair'}
METHODS=['raw','ideal','tweedie','polygon']
ETA_NOR=R/(2*np.sqrt(2*(2+2))+np.sqrt(2*(2+2)))
T_NOR=float(core.t_of_eta(ETA_NOR))
SELECT=.16

def write(path,rows):
    core.write_csv(path,rows)

def read(path):
    if not path.exists(): return []
    with path.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            try: r[k]=float(v)
            except ValueError: pass
    return rows

def tweedie(x,eta):
    # Equal empirical weights. Stable normal residual via posterior variance.
    logits=-cdist(x,P,'sqeuclidean')/(2*eta*eta)
    w=softmax(logits,axis=1)
    mean=w@P
    anchor=np.argmax(logits,axis=1)
    offset=mean-P[anchor]
    var=np.einsum('ij,ij->i',w,PAIR[anchor])-np.einsum('ij,ij->i',offset,offset)
    frac=np.clip(var/(R*R),0.,1.)
    radial=R*frac/(1+np.sqrt(1-frac))
    return mean,radial

def polygon(x):
    # Ray from the known centre to the chord between adjacent empirical atoms.
    # This is radial repair, NOT Euclidean nearest-point projection to polygon.
    theta=np.mod(np.arctan2(x[:,1],x[:,0]),2*np.pi)
    angles=np.mod(np.arctan2(P[:,1],P[:,0]),2*np.pi)
    order=np.argsort(angles); angles=angles[order]
    hi=np.searchsorted(angles,theta,side='right')
    lo=(hi-1)%N; hi=hi%N
    left=angles[lo]; gap=np.mod(angles[hi]-left,2*np.pi)
    offset=np.mod(theta-left,2*np.pi)-gap/2
    radius=R*np.cos(gap/2)/np.cos(offset)
    return radius[:,None]*np.column_stack([np.cos(theta),np.sin(theta)])

def solve(task):
    eta,power,count,seed,level=task
    start=time.perf_counter()
    x=core.noisy_nodes(eta,power,seed); ideal=core.project(x)
    tw,radial=tweedie(x,eta); poly=polygon(x); y=core.target_design(count)
    wt=core.exact_ot(tw,y); wtt=core.exact_ot(core.project(tw),y)
    wp=core.exact_ot(poly,y)
    wn_t=float(np.sqrt(np.mean(radial**2)))
    wn_p=float(np.sqrt(np.mean((np.linalg.norm(poly,axis=1)-R)**2)))
    if wn_t>np.sqrt(8)/R*eta**2+1e-9: raise RuntimeError('Normal bound check failed')
    if not wn_t<=wt[0]+1e-9<=wn_t+wtt[0]+2e-9: raise RuntimeError('Tweedie transport bounds failed')
    if np.max(abs(core.project(poly)-ideal))>1e-12: raise RuntimeError('Polygon angle preservation failed')
    return dict(eta=eta,t=float(core.t_of_eta(eta)),power=power,target_nodes=count,seed=seed,level=level,
                tweedie_w2=float(wt[0]),tweedie_wt=float(wtt[0]),tweedie_wn=wn_t,
                polygon_w2=float(wp[0]),polygon_wn=wn_p,
                tweedie_bias=float(np.sqrt(np.mean(np.sum((tw-ideal)**2,axis=1)))),
                polygon_bias=wn_p,
                max_duality_gap=max(wt[1],wtt[1],wp[1]),
                max_marginal_error=max(wt[2],wtt[2],wp[2]),
                max_dual_violation=max(wt[3],wtt[3],wp[3]),seconds=time.perf_counter()-start)

def batch(tasks,path,workers):
    rows=read(path)
    done={(round(r['eta'],12),int(r['power']),int(r['seed'])) for r in rows}
    todo=[t for t in tasks if (round(t[0],12),t[1],t[3]) not in done]
    print(f'{path.name}: {len(todo)} scales pending',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(solve,t) for t in todo]):
            r=fut.result(); rows.append(r); rows.sort(key=lambda r:(r['level'],r['t']))
            write(path,rows)
            if len(rows)%10==0 or abs(r['eta']-SELECT)<1e-10:
                print(f"{path.name}: {len(rows)} done; eta={r['eta']:.5g}; Tweedie W2={r['tweedie_w2']:.6g}",flush=True)
    return rows

def normal_table():
    rows=[]
    # Dedicated denser quadrature, not a subset chosen for a favourable slope.
    grid=np.unique(np.r_[np.geomspace(.1,1.2,34),SELECT,ETA_NOR])
    for eta in grid:
        result=[]
        for power,seed in [(12,SEED),(13,SEED),(12,SEED+1)]:
            x=core.noisy_nodes(eta,power,seed)
            _,r=tweedie(x,eta)
            result.append(float(np.sqrt(np.mean(r*r))))
        rows.append(dict(eta=eta,t=float(core.t_of_eta(eta)),raw_wn=core.normal_exact(eta),
                         tweedie_wn=result[1],coarse_wn=result[0],independent_wn=result[2],
                         sensitivity=max(abs(result[1]-result[0]),abs(result[1]-result[2])),
                         upper_bound=np.sqrt(8)/R*eta**2))
    rows.sort(key=lambda r:r['t']); write(DATA/'figure04_normal.csv',rows)
    return rows

def merge():
    bases=read(ROOT/'reference/data/distances.csv')
    repairs=read(DATA/'repairs.csv'); angular=read(DATA/'ideal_angular.csv')
    a={round(r['eta'],12):r for r in angular}
    b={round(r['eta'],12):r for r in bases}
    polar={round(r['eta'],12):r for r in read(DATA/'polar_moments.csv')}
    dense={round(r['eta'],12):r for r in read(DATA/'tweedie_angular_dense.csv')}
    rows=[]
    for original in repairs:
        r=dict(original)
        key=round(r['eta'],12)
        if key in polar:
            z=polar[key]
            for m in ['tweedie','polygon']:
                r[m+'_w2']=float(np.sqrt(max(0.,r[m+'_w2']**2-r[m+'_wn']**2+z[m+'_wn']**2)))
                r[m+'_wn']=z[m+'_wn']
            r['tweedie_bias']=z['tweedie_bias']; r['polygon_bias']=z['polygon_wn']
        if key in dense: r['tweedie_wt']=dense[key]['wt']
        key=round(r['eta'],12); base=b[key]; ideal=a[key]['wt']
        # Freeze the Figure 2 total/normal estimates. Ideal is integrated from
        # the analytic angular density to avoid source-QMC angular artefacts.
        z=dict(eta=r['eta'],t=r['t'],raw_w2=base['w2'],raw_wt=ideal,raw_wn=base['wn'],
               ideal_w2=ideal,ideal_wt=ideal,ideal_wn=0.,
               tweedie_w2=r['tweedie_w2'],tweedie_wt=r['tweedie_wt'],tweedie_wn=r['tweedie_wn'],
               polygon_w2=r['polygon_w2'],polygon_wt=ideal,polygon_wn=r['polygon_wn'],
               tweedie_bias=r['tweedie_bias'],polygon_bias=r['polygon_bias'],endpoint=SN)
        rows.append(z)
    rows.sort(key=lambda r:r['t']); write(DATA/'figure05_curves.csv',rows)
    selected=min(rows,key=lambda r:abs(r['eta']-SELECT))
    table=[]
    for m in METHODS:
        table.append(dict(method=m,eta=selected['eta'],t=selected['t'],
                          w2=selected[m+'_w2'],wt=selected[m+'_wt'],wn=selected[m+'_wn'],
                          gain_vs_raw_percent=100*(1-selected[m+'_w2']/selected['raw_w2']),
                          gain_vs_endpoint_percent=100*(1-selected[m+'_w2']/SN)))
    write(DATA/'paired_comparison_eta_0p16.csv',table)
    return rows

def compute(workers):
    DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    bases=read(ROOT/'reference/data/distances.csv')
    rows=batch([(r['eta'],7,2048,SEED,'main') for r in bases],DATA/'repairs.csv',workers)
    # Use cached exact-angular OT where available; fill only missing scales.
    cached=read(ROOT/'reference/data/angular_transport.csv')+read(DATA/'ideal_angular.csv')
    angular={round(r['eta'],12):r for r in cached if int(r['target_nodes'])==2048}
    tasks=[(r['eta'],2048) for r in bases if round(r['eta'],12) not in angular]
    print(f'Ideal angular transport: {len(tasks)} missing scales',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(core.angular_ot_worker,t) for t in tasks]):
            r=fut.result(); angular[round(r['eta'],12)]=r
            write(DATA/'ideal_angular.csv',list(angular.values()))
    write(DATA/'ideal_angular.csv',[angular[round(r['eta'],12)] for r in bases])
    grid=np.array([r['eta'] for r in bases])
    checks=np.unique([grid[np.argmin(abs(grid-e))] for e in [.008,.08,.12,.16,.2,.35,1.2]])
    tasks=[(float(e),8,4096,SEED,'refined') for e in checks]
    tasks += [(float(e),7,2048,SEED+1,'independent_scramble') for e in checks]
    audits=batch(tasks,DATA/'quadrature_checks.csv',workers)
    nr=normal_table(); curves=merge()
    deltas=[]
    for r in audits:
        base=min(rows,key=lambda z:abs(z['eta']-r['eta']))
        deltas.append(dict(eta=r['eta'],level=r['level'],**{k:r[k]-base[k] for k in ['tweedie_w2','tweedie_wt','polygon_w2','tweedie_wn']}))
    write(DATA/'quadrature_deltas.csv',deltas)
    manifest=dict(config=dict(n_train=N,radius=R,amplitude=core.AMP,seed=SEED,selected_eta=SELECT,
                             source_nodes=2048,target_nodes=2048,normal_source_nodes=131072),
                  endpoint=SN,eta_normal_sufficient=ETA_NOR,t_normal_sufficient=T_NOR,
                  coordinate='U = X / alpha(t); eta = cot(pi*t/2)',
                  dimension_panel='Exact orthogonal-padding identity; no independent high-dimensional run claimed.',
                  normal_sensitivity_max=max(r['sensitivity'] for r in nr),
                  normal_sensitivity_relative_max=max(r['sensitivity']/r['tweedie_wn'] for r in nr),
                  max_lp_duality_gap=max(r['max_duality_gap'] for r in rows+audits),
                  max_lp_marginal_error=max(r['max_marginal_error'] for r in rows+audits),
                  max_lp_dual_violation=max(r['max_dual_violation'] for r in rows+audits),
                  max_refinement_w2_change=max(abs(r[k]) for r in deltas for k in ['tweedie_w2','polygon_w2']),
                  caveat='Quadrature sensitivity checks are not statistical confidence intervals or rigorous source error bounds.',
                  training_sha256=hashlib.sha256((ROOT/'reference/data/training_points.npy').read_bytes()).hexdigest(),
                  software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,POT=ot.__version__))
    (DATA/'numerical_audit.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'mathtext.fontset':'dejavusans',
                         'axes.labelsize':8,'axes.titlesize':9,'axes.edgecolor':'#BCC5CC','axes.linewidth':.65,
                         'xtick.labelsize':7,'ytick.labelsize':7,'xtick.color':'#62717D','ytick.color':'#62717D',
                         'text.color':'#253340','axes.labelcolor':'#455561','pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})

def frame(ax,title,xlabel=r'Reverse time $t$'):
    ax.set_title(title,loc='left',pad=8,fontweight='bold')
    ax.set_xlabel(xlabel)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.grid(axis='y',color='#EDF0F2',lw=.55,zorder=0)
    ax.tick_params(length=3,width=.6)

def top_noise(ax):
    def inverse(t):
        return 1/np.tan(np.pi*np.clip(t,1e-8,1-1e-8)/2)
    sec=ax.secondary_xaxis('top',functions=(inverse,core.t_of_eta))
    sec.set_xticks([1.2,.35,.16,.08]); sec.set_xticklabels(['1.2','0.35','0.16','0.08'])
    sec.set_xlabel(r'Noise scale $\eta$',labelpad=5,fontsize=7)
    sec.tick_params(labelsize=6.5,length=2)

def save(fig,name):
    OUT.mkdir(exist_ok=True)
    for ext in ['pdf','png','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=350,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

def plot4():
    rows=read(DATA/'figure04_normal.csv')
    t=np.array([r['t'] for r in rows]); eta=np.array([r['eta'] for r in rows])
    raw=np.array([r['raw_wn'] for r in rows]); tw=np.array([r['tweedie_wn'] for r in rows])
    err=np.array([r['sensitivity'] for r in rows])
    bound=np.array([r['upper_bound'] for r in rows])
    fig=plt.figure(figsize=(6.75,5.0)); gs=fig.add_gridspec(2,2,height_ratios=[1.2,1],hspace=.62,wspace=.38)
    ax=fig.add_subplot(gs[0,:]); frame(ax,'(a) Normal thickness along the reverse process')
    ax.plot(t,raw,color=COL['raw'],lw=1.8,label=r'Raw $W_N(\mu_{N,\eta})$')
    ax.plot(t,tw,color=COL['tweedie'],lw=1.8,label=r'Tweedie $W_N(\nu_{N,\eta})$')
    ax.fill_between(t,np.maximum(tw-err,1e-10),tw+err,color=COL['tweedie'],alpha=.16,lw=0)
    ax.plot(t,eta,color=COL['raw'],ls=':',lw=1,label=r'First-order reference $\eta$')
    ax.plot(t,bound,color=COL['bound'],ls='--',lw=1.2,label=r'Uniform bound $\sqrt{8}\,\eta^2/\rho$')
    ax.set_yscale('log'); ax.set_ylabel(r'Normal RMS $W_N$'); ax.set_xlim(t.min(),t.max())
    ax.legend(frameon=False,ncol=2,fontsize=7,loc='lower left')
    ax.text(.985,.96,r'$N=16,\ R=\rho=2,\ D=k=2$',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    ax.text(.015,.37,r'Ideal projection: $W_N=0$',transform=ax.transAxes,ha='left',fontsize=7,color=COL['ideal'])
    top_noise(ax)
    ax=fig.add_subplot(gs[1,0]); frame(ax,'(b) Strict normal improvement')
    ax.axvspan(T_NOR,t.max(),color='#EAF3EF',zorder=0)
    ax.axhline(1,color=COL['bound'],ls='--',lw=1)
    ax.plot(t,tw/raw,color=COL['tweedie'],lw=1.8)
    ax.axvline(T_NOR,color='#688875',ls=':',lw=1)
    ax.set_xlim(t.min(),t.max()); ax.set_ylim(0,1.07)
    ax.set_ylabel(r'$W_N(\nu_{N,\eta})/W_N(\mu_{N,\eta})$')
    ax.text(.04,.88,'Green: sufficient theorem region',transform=ax.transAxes,fontsize=6.5)
    ax.text(.04,.76,rf'$t> {T_NOR:.3f}$ ($\eta<{ETA_NOR:.3f}$)',transform=ax.transAxes,fontsize=7)
    ax=fig.add_subplot(gs[1,1]); frame(ax,'(c) Ambient dimension',r'Ambient dimension $D$')
    dims=np.array([2,8,32,128]); raw0=core.normal_exact(SELECT)
    idx=np.argmin(abs(eta-SELECT)); tw0=tw[idx]
    rawd=np.sqrt(raw0**2+(dims-2)*SELECT**2)
    ax.plot(dims,rawd,'o-',color=COL['raw'],lw=1.5,ms=4,label='Raw (exact identity)')
    ax.plot(dims,np.full(4,tw0),'s-',color=COL['tweedie'],lw=1.5,ms=4,label='Tweedie (invariant)')
    ax.axhline(np.sqrt(8)/R*SELECT**2,color=COL['bound'],ls='--',lw=1,label='Affine-rank bound')
    ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.set_xticks(dims); ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel(r'Normal RMS $W_N$'); ax.set_ylim(.002,10); ax.legend(frameon=False,fontsize=6.5,loc='upper left')
    ax.text(.98,.04,rf'$\eta={SELECT},\ t={float(core.t_of_eta(SELECT)):.3f},\ k=2$',transform=ax.transAxes,ha='right',fontsize=6.5)
    save(fig,'figure_04_normal_reduction')

def plot5():
    rows=read(DATA/'figure05_curves.csv'); t=np.array([r['t'] for r in rows])
    def arr(key): return np.array([r[key] for r in rows])
    chosen=min(rows,key=lambda r:abs(r['eta']-SELECT)); ts=chosen['t']
    fig,axes=plt.subplots(3,2,figsize=(6.75,8.1)); fig.subplots_adjust(hspace=.58,wspace=.37,top=.90,bottom=.065,left=.095,right=.985)
    for ax in axes.flat:
        ax.axvline(ts,color='#ABB4BD',lw=.8,ls=':',zorder=0)
        ax.set_xlim(t.min(),1.)
    ax=axes[0,0]; frame(ax,'(a) Total distributional error')
    for m in METHODS:
        vals=np.r_[arr(m+'_w2')/SN,1.]
        ax.plot(np.r_[t,1.],vals,color=COL[m],lw=1.6,ls='--' if m=='ideal' else '-',zorder=4 if m=='ideal' else 3)
    ax.axhline(1,color=COL['bound'],ls=':',lw=1)
    ax.set_yscale('log'); ax.set_ylabel(r'$W_2(p,q)/S_N$'); ax.set_ylim(.006,14)
    
    ax=axes[0,1]; frame(ax,'(b) Tangential error')
    ax.plot(np.r_[t,1.],np.r_[arr('raw_wt')/SN,1.],color=COL['ideal'],lw=1.8,label='Raw = ideal = polygon')
    ax.plot(np.r_[t,1.],np.r_[arr('tweedie_wt')/SN,1.],color=COL['tweedie'],lw=1.8,label='Empirical Tweedie')
    ax.axhline(1,color=COL['bound'],ls=':',lw=1); ax.set_ylabel(r'$W_T(p,q)/S_N$'); ax.set_ylim(0,1.1)
    ax.legend(frameon=False,fontsize=6.5,loc='center left',bbox_to_anchor=(0,.76))
    ax=axes[1,0]; frame(ax,'(c) Normal residual')
    for m in ['raw','tweedie','polygon']:
        values=arr(m+'_wn')/SN; values=np.where(values>1e-7,values,np.nan)
        ax.plot(t,values,color=COL[m],lw=1.6)
    ax.set_yscale('log'); ax.set_ylim(1e-5,20); ax.set_ylabel(r'$W_N(q)/S_N$')
    ax.text(.04,.08,r'Ideal projection: $W_N=0$',transform=ax.transAxes,fontsize=7,color=COL['ideal'])
    ax=axes[1,1]; ax.clear(); frame(ax,'(d) Paired gains at a fixed scale','')
    pos=np.arange(4); w=.34
    rawgain=np.array([100*(1-chosen[m+'_w2']/chosen['raw_w2']) for m in METHODS])
    endgain=np.array([100*(1-chosen[m+'_w2']/SN) for m in METHODS])
    ax.bar(pos-w/2,rawgain,w,color=[COL[m] for m in METHODS],label='vs. same-time raw')
    ax.bar(pos+w/2,endgain,w,facecolor='white',edgecolor=[COL[m] for m in METHODS],hatch='///',label='vs. endpoint')
    ax.axhline(0,color=COL['bound'],lw=.8); ax.set_xticks(pos,['Raw','Ideal','Tweedie','Polygon'],fontsize=6.6)
    ax.set_ylabel('Relative reduction (%)'); ax.set_ylim(min(rawgain.min(),endgain.min(),0)-16,115)
    ax.legend(frameon=False,fontsize=6.2,loc='upper left')
    ax.text(.98,.96,rf'$\eta={SELECT}$, $t={ts:.3f}$',transform=ax.transAxes,ha='right',va='top',fontsize=6.7)
    for x,g in zip(pos-w/2,rawgain):
        ax.text(x,g+(3 if g>=0 else -3),f'{g:.0f}',ha='center',va='bottom' if g>=0 else 'top',fontsize=6)
    ax=axes[2,0]; frame(ax,'(e) Repair bias versus gain budget')
    budget=(SN-arr('ideal_w2'))/SN
    ax.plot(t,budget,color=COL['ideal'],lw=1.7,label=r'Budget $(S_N-W_T)/S_N$')
    for m in ['tweedie','polygon']:
        ax.plot(t,arr(m+'_bias')/SN,color=COL[m],lw=1.5,label=r'$\epsilon_R/S_N$: '+('Tweedie' if m=='tweedie' else 'polygon'))
    ax.set_ylabel('Normalized RMS / budget'); ax.set_xlim(.65,1.); ax.set_ylim(0,1.18)
    ax.text(.03,.91,'Green: available gain budget',color=COL['ideal'],transform=ax.transAxes,fontsize=6.8)
    
    ax=axes[2,1]; frame(ax,'(f) Total error near the endpoint')
    for m in METHODS:
        ax.plot(np.r_[t,1.],np.r_[arr(m+'_w2')/SN,1.],color=COL[m],lw=1.6,ls='--' if m=='ideal' else '-')
    ax.axhline(1,color=COL['bound'],ls=':',lw=1)
    ax.set_xlim(.85,1.); ax.set_ylim(0,1.15); ax.set_ylabel(r'$W_2(p,q)/S_N$')
    
    handles=[Line2D([],[],color=COL[m],lw=1.8,ls='--' if m=='ideal' else '-',label=LABEL[m]) for m in METHODS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.995),ncol=2,frameon=False,fontsize=7.4,columnspacing=2.5)
    save(fig,'figure_05_early_stop_and_repair')
    # Supplementary geometry diagnostic: same 512 deterministic quadrature nodes.
    x=core.noisy_nodes(SELECT,5,SEED); tw,_=tweedie(x,SELECT)
    outputs=[x,core.project(x),tw,polygon(x)]
    fig,axes=plt.subplots(1,4,figsize=(6.75,1.95))
    theta=np.linspace(0,2*np.pi,500)
    for ax,m,z in zip(axes,METHODS,outputs):
        ax.plot(R*np.cos(theta),R*np.sin(theta),color='#AAB3BA',lw=.7)
        ax.scatter(z[:,0],z[:,1],s=1.4,color=COL[m],alpha=.45,rasterized=True)
        ax.scatter(P[:,0],P[:,1],s=5,color='#253340',zorder=4)
        ax.set_aspect('equal'); ax.set_xlim(-2.65,2.65); ax.set_ylim(-2.65,2.65); ax.axis('off')
        ax.set_title(LABEL[m],fontsize=7)
    fig.suptitle(rf'Same input nodes at $\eta={SELECT}$ ($t={ts:.3f}$)',fontsize=8,y=.99)
    fig.tight_layout(pad=.4,rect=[0,0,1,.85])
    for ext in ['png','svg']: fig.savefig(OUT/f'shared_input_geometry.{ext}',dpi=350,bbox_inches='tight')
    plt.close(fig)

def plot():
    style(); plot4(); plot5()
    print('Saved final PDF, PNG and SVG figures.',flush=True)


def finalize_refinement(workers):
    from refine_integrals import run,dense_angular
    polar,angular=run(workers)
    indexed={round(r['eta'],12):r for r in polar}
    normals=read(DATA/'figure04_normal.csv')
    for r in normals:
        z=indexed[round(r['eta'],12)]
        r['qmc_wn']=r.get('qmc_wn',r['tweedie_wn'])
        r['tweedie_wn']=z['tweedie_wn']; r['sensitivity']=z['sensitivity']
        r['polar_coarse_wn']=z['coarse_wn']; r['polar_mass']=z['mass']
    write(DATA/'figure04_normal.csv',normals)
    curves=merge()
    angular_checks=read(DATA/'angular_refinement.csv')
    done={round(r['eta'],12) for r in angular_checks}
    for eta in [1.2,.35,.16,.08]:
        if round(eta,12) not in done:
            z=dense_angular(eta,order=96,bins=4096)
            base=min(angular,key=lambda r:abs(r['eta']-eta))
            z['change_from_main']=z['wt']-base['wt']
            angular_checks.append(z); write(DATA/'angular_refinement.csv',angular_checks)
    audits=read(DATA/'quadrature_checks.csv'); repairs=read(DATA/'repairs.csv')
    deltas=[]
    for r in audits:
        key=round(r['eta'],12); base=min(repairs,key=lambda z:abs(z['eta']-r['eta']))
        z=dict(eta=r['eta'],level=r['level'])
        for m in ['tweedie','polygon']:
            a,b=r[m+'_w2'],base[m+'_w2']
            if key in indexed:
                normal=indexed[key][m+'_wn']
                a=np.sqrt(max(0.,a*a-r[m+'_wn']**2+normal**2))
                b=np.sqrt(max(0.,b*b-base[m+'_wn']**2+normal**2))
            z[m+'_w2_change']=float(a-b)
        deltas.append(z)
    write(DATA/'final_transport_sensitivity.csv',deltas)
    audit=json.loads((DATA/'numerical_audit.json').read_text())
    for key in ['normal_sensitivity_max','normal_sensitivity_relative_max','max_refinement_w2_change']:
        audit['initial_qmc_'+key]=audit.pop(key,None)
    audit.update(normal_estimator='Split polar Gauss-Legendre, orders 32 and 64; radial cutoff R+9 eta',
                 normal_sensitivity_max=max(r['sensitivity'] for r in normals),
                 normal_sensitivity_relative_max=max(r['sensitivity']/r['tweedie_wn'] for r in normals),
                 polar_mass_error_max=max(abs(r['mass']-1) for r in polar),
                 total_estimator='Exact radial-cost control variate using polar normal moment for eta >= .06',
                 tangential_estimator='Polar pushforward + 2048 equal-mass angular quantiles for Tweedie at eta >= .08; QMC endpoint tail',
                 max_dense_angular_refinement_change=max(abs(r['change_from_main']) for r in angular_checks),
                 max_refinement_w2_change=max(abs(r[k]) for r in deltas for k in ['tweedie_w2_change','polygon_w2_change']))
    (DATA/'numerical_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Final refinement audit:',json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['compute','plot','all'],nargs='?',default='plot')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.action in ['compute','all']:
        compute(a.workers)
        finalize_refinement(a.workers)
    if a.action in ['plot','all']: plot()