"""Shared reverse-time topology and distribution figure.
Blue=W2, green=WT, orange=WN, purple=generated particles, black=training.
Run: python make_figure_02.py plot. No new distance estimates are substituted.
"""
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from scipy.integrate import solve_ivp
from scipy.special import softmax


def save(fig,stem):
    # Trim only the removed caption area; preserve the plot and numeric labels.
    bounds=Bbox.from_extents(0,.52,6.75,6.5)
    fig.savefig(stem.with_suffix('.pdf'),bbox_inches=bounds,pad_inches=0)
    fig.savefig(stem.with_suffix('.svg'),bbox_inches=bounds,pad_inches=0)
    fig.savefig(stem.with_suffix('.png'),dpi=360,bbox_inches=bounds,pad_inches=0)
    plt.close(fig)


def union_betti(m,noise):
    # Same equal-radius neighborhood union as Figure 1.
    th=np.sort(np.mod(np.arctan2(m.POINTS[:,1],m.POINTS[:,0]),2*np.pi))
    gaps=np.diff(np.r_[th,th[0]+2*np.pi])
    bridges=m.R*np.sin(gaps/2)
    radius=m.CONTEXT['geometry']['c']*float(noise)
    if radius>=m.R: return (1,0)
    unbridged=int(np.count_nonzero(bridges>radius))
    return max(1,unbridged),int(unbridged==0)


def prepare_states(m,chosen):
    initial_path=m.DATA/'timeline_initial_state.npz'
    src=np.load(initial_path)
    assert np.array_equal(src['training_points'],m.POINTS)
    noise=np.array([r['eta'] for r in chosen])
    sig=dict(source_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
             etas=noise.tolist(),method='DOP853-log-noise-v1')
    cache=m.DATA/'timeline_states.npz'; audit_path=m.DATA/'timeline_integration_audit.json'
    if cache.exists() and audit_path.exists():
        audit=json.loads(audit_path.read_text())
        if audit['signature']==sig: return np.load(cache)['clouds'],audit
    start=float(src['eta_start']); reference=float(src['reference_eta'])
    scales=np.sort(np.unique(np.r_[noise,reference]))[::-1]
    ss=np.log(start/scales)
    def rhs(s,y):
        x=y.reshape(-1,2); eta=start*np.exp(-s)
        d2=np.sum((x[:,None,:]-m.POINTS[None,:,:])**2,axis=2)
        mean=softmax(-d2/(2*eta**2),axis=1)@m.POINTS
        return (mean-x).ravel()
    def solve(rtol,atol,step):
        out=solve_ivp(rhs,(0,float(ss[-1])),src['initial'].ravel(),method='DOP853',
                      t_eval=ss,rtol=rtol,atol=atol,max_step=step)
        if not out.success: raise RuntimeError(out.message)
        return out.y.T.reshape(-1,len(src['initial']),2)
    first=solve(2e-9,2e-11,.055)
    refined=solve(4e-10,4e-12,.0275)
    error=float(np.linalg.norm(first-refined,axis=2).max())
    reference_error=float(np.linalg.norm(refined[np.argmin(abs(scales-reference))]-src['reference_cloud'],axis=1).max())
    assert error<1e-6 and reference_error<1e-6
    clouds=np.stack([refined[np.argmin(abs(scales-e))] for e in noise])
    np.savez_compressed(cache,clouds=clouds,etas=noise,times=[r['t'] for r in chosen],training_points=m.POINTS)
    audit=dict(signature=sig,max_refinement_error=error,figure_01_reference_error=reference_error,
               particles=len(src['initial']),same_initial_particles=True)
    audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return clouds,audit


def render(m):
    m.style()
    rows=sorted(m.read_csv(m.DATA/'distances.csv'),key=lambda r:r['t'])
    dist_audit=json.loads((m.DATA/'numerical_audit.json').read_text())
    sn=dist_audit['empirical_endpoint']
    chosen=[min(rows,key=lambda r:abs(r['t']-v)) for v in [.743,.840,.921]]
    clouds,integration=prepare_states(m,chosen)
    for r in chosen:
        assert m.RECOVERY[0]<r['t']<m.RECOVERY[1]
        assert union_betti(m,r['eta'])==(1,1)
    t=np.r_[[r['t'] for r in rows],1.]
    raw=np.r_[[r['w2'] for r in rows],sn]
    wt=np.r_[[r['wt'] for r in rows],sn]
    wn=np.r_[[r['wn'] for r in rows],0.]
    th=np.sort(np.mod(np.arctan2(m.POINTS[:,1],m.POINTS[:,0]),2*np.pi))
    gaps=np.diff(np.r_[th,th[0]+2*np.pi]); bridges=m.R*np.sin(gaps/2)
    critical=m.t_of_eta(np.r_[bridges,m.R]/m.CONTEXT['geometry']['c'])
    topo_t=np.unique(np.r_[np.linspace(.25,1,2401),critical-1e-9,critical+1e-9,[r['t'] for r in chosen]])
    topo_noise=1/np.tan(np.pi*topo_t/2)
    betti=np.array([union_betti(m,e) for e in topo_noise])
    # Independent persistence check, excluding roundoff-scale cocircular bars.
    topology_check='not run: optional gudhi unavailable'
    try:
        import gudhi
        st=gudhi.AlphaComplex(points=m.POINTS).create_simplex_tree()
        st.compute_persistence(homology_coeff_field=2)
        bars=[st.persistence_intervals_in_dimension(k) for k in [0,1]]
        bars=[v[(v[:,1]-v[:,0])>1e-10] for v in bars]
        verify_times=np.linspace(.251,.999,401)
        for tt in verify_times:
            e=1/np.tan(np.pi*tt/2); rr=(m.CONTEXT['geometry']['c']*e)**2
            measured=tuple(int(np.sum((v[:,0]<=rr)&(rr<v[:,1]))) for v in bars)
            assert measured==union_betti(m,e)
        topology_check='passed: AlphaComplex, F2, 401 independent times'
    except ImportError:
        pass
    np.savez_compressed(m.DATA/'timeline_betti.npz',times=topo_t,betti=betti)
    blue=m.COL['raw']; green=m.COL['tangent']; orange=m.COL['normal']
    purple='#7953A2'; ink='#253340'; grey='#78858E'
    fig=plt.figure(figsize=(6.75,6.5))
    m.panel_title(fig,.05,.968,'a','Topology and distributional error along reverse diffusion')
    b=fig.add_axes([.10,.528,.78,.345])
    m.decorate(b,time_axis=False,shade=True)
    b.set_xlim(.25,1)
    for r in chosen: b.axvline(r['t'],color='#9AA4AD',lw=.65,ls=(0,(2,3)),zorder=1)
    topology=b.twinx()
    topology.spines['top'].set_visible(False)
    topology.spines['left'].set_visible(False)
    topology.spines['bottom'].set_visible(False)
    topology.set_ylim(-.8,17.2)
    topology.set_yticks([0,1,4,8,16])
    topology.tick_params(axis='y',labelsize=6,length=2,pad=3,colors='#697782')
    topology.set_ylabel('Betti number',fontsize=7.5,color='#697782',labelpad=6)
    topology.step(topo_t,betti[:,0],where='post',lw=1.3,color='#7C8790',ls=(0,(5,3)),zorder=2)
    topology.step(topo_t,betti[:,1],where='post',lw=1.15,color='#36424D',ls=(0,(1,2)),zorder=3)
    for r in chosen:
        topology.scatter(r['t'],1,s=17,facecolors='white',edgecolors='#56636E',lw=.8,zorder=5)
    b.text(np.mean(m.RECOVERY),.985,'Topology recovery interval',transform=b.get_xaxis_transform(),
           ha='center',va='top',size=6.6,color=grey)
    for values,color,label in [(wn,orange,r'Normal $W_N$'),(wt,green,r'Tangential $W_T$'),(raw,blue,r'Total $W_2$')]:
        keep=values>0
        b.plot(t[keep],values[keep]/sn,color=color,lw=1.45 if color!=blue else 2,zorder=3 if color!=blue else 4,label=label)
    b.axhline(1,color='#6D7780',lw=.7,ls=(0,(4,3)))
    b.set_yscale('log'); b.set_ylim(.003,12)
    b.set_ylabel(r'Distance / $S_N$'); b.set_xlabel(r'Reverse diffusion time $t$',labelpad=3)
    b.set_xticks([.25,.5,.75,1],['0.25','0.50','0.75','1'])
    handles=[Line2D([],[],color=blue,lw=2),Line2D([],[],color=green,lw=1.45),Line2D([],[],color=orange,lw=1.45),
             Line2D([],[],color='#7C8790',lw=1.3,ls=(0,(5,3))),Line2D([],[],color='#36424D',lw=1.15,ls=(0,(1,2)))]
    fig.legend(handles,[r'Total $W_2$',r'Tangential $W_T$',r'Normal $W_N$',r'$\beta_0$',r'$\beta_1$'],
               loc='center',bbox_to_anchor=(.495,.918),frameon=False,ncol=5,fontsize=6.9,handlelength=2,columnspacing=1.2)
    for label,r in zip('ABC',chosen):
        b.scatter(r['t'],r['wt']/sn,s=31,color=green,edgecolor='white',lw=.7,zorder=7)
        b.scatter(r['t'],r['w2']/sn,s=18,color=blue,edgecolor='white',lw=.5,zorder=7)
        b.annotate(label,(r['t'],r['wt']/sn),xytext=(-8,-11),textcoords='offset points',size=7.0,weight='bold',color=green)
    b.text(.99,1.11,r'Endpoint $S_N$',ha='right',size=6,color='#6D7780')
    # Labels identify correspondence; fan-out lines connect the same exact times to snapshots.
    centers=[.18,.50,.82]
    for center,r in zip(centers,chosen):
        xpos=.10+.78*(r['t']-.25)/.75
        line=Line2D([xpos,center],[.470,.421],transform=fig.transFigure,color='#BBC2C8',lw=.7,zorder=0)
        fig.add_artist(line)
    m.panel_title(fig,.05,.448,'b','Phase II states')
    train=m.POINTS/m.R
    cloud_norm=clouds/m.R
    lim=max(1.75,float(np.ceil(np.max(abs(cloud_norm))*20)/20+.05))
    theta=np.linspace(0,2*np.pi,600)
    for i,(center,label,r) in enumerate(zip(centers,'ABC',chosen)):
        fig.text(center,.400,label+'   '+r'$t=%.3f$'%r['t'],ha='center',size=8.5,weight='bold')
        fig.text(center,.377,r'$(\beta_0,\beta_1)=(1,1)$',ha='center',size=7.2,color=grey)
        ax=fig.add_axes([center-.148,.137,.296,.225]); ax.set_aspect('equal'); ax.axis('off')
        ax.plot(np.cos(theta),np.sin(theta),color='#647380',lw=.85,zorder=0)
        cloud=cloud_norm[i]
        ax.scatter(cloud[:,0],cloud[:,1],s=3.5,color=purple,alpha=.48,linewidths=0,zorder=2)
        ax.scatter(train[:,0],train[:,1],s=8.5,color='#15191E',edgecolors='white',lw=.3,zorder=4)
        ax.set(xlim=(-lim,lim),ylim=(-lim,lim))
        fig.text(center,.114,r'$W_T/S_N = %.3f$'%(r['wt']/sn),ha='center',size=11,color=green,weight='bold')
    # Publish the selected metrics rather than replacing them with rounded labels.
    selected=[]
    for label,r in zip('ABC',chosen):
        selected.append(dict(label=label,t=r['t'],eta=r['eta'],beta0=1,beta1=1,w2=r['w2'],wt=r['wt'],wn=r['wn'],
                             w2_over_sn=r['w2']/sn,wt_over_sn=r['wt']/sn,wn_over_sn=r['wn']/sn))
    m.write_csv(m.DATA/'timeline_selected_states.csv',selected)
    report=dict(layout='one dual-axis plot: distances left, Betti counts right; one row of three matched Phase II states',
                topology_object='union of radius c*eta disks centered at the same training samples as Figure 1',
                topology_check=topology_check,recovery_interval=m.RECOVERY,selected=selected,
                field_of_view_R=[-lim,lim],integration=integration,
                interpretation='Across A-C WT rises while W2 still falls; the late full-process rise occurs later.')
    (m.DATA/'visual_layout_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save(fig,m.OUT/'figure_02_distribution_geometry')
    print(json.dumps(report,indent=2),flush=True)
