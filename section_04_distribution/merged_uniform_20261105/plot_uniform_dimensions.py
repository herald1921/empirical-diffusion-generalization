"""Compact Figure 2 layout (2026-11-05); unchanged cached data and 9.2 pt typography."""
from pathlib import Path
import csv,json,sys,shutil
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text
import compute_uniform as c
ROOT=Path(__file__).resolve().parent
def load(name):
    return [{k:v if k=='level' else float(v) for k,v in row.items()} for row in csv.DictReader((ROOT/name).open())]
byeta={round(r['eta'],12):r for r in load('curves_dense.csv')}
for file in ['dimension_checks.csv','minimum_search.csv','fine_checks.csv','late_stop_checks.csv','late_b_candidates.csv']:
    for row in load(file):
        key=round(row['eta'],12)
        if key not in byeta or row['power']>=byeta[key]['power']:
            if row['seed']==c.SEED:byeta[key]=row
rows=sorted(byeta.values(),key=lambda r:r['eta'])
eta=np.array([r['eta'] for r in rows]);x=eta/c.R
A=byeta[.35];B=byeta[.04]
raw2=np.array([r['raw_w2']**2 for r in rows]);rawfun=PchipInterpolator(eta,raw2)
Ds=[2,4,8,16];alpha=np.pi/c.N
def cdim(d):return np.sqrt(d)+np.sqrt(2*np.log(20))
def diversity(noise,d=2):
    radius=cdim(d)*np.asarray(noise)
    beta=np.arcsin(np.minimum(radius/c.R,1.))
    g=np.maximum(alpha-beta,0.)
    # Stable analytic expression near zero, with integral definition checked below.
    diff=np.where(g<1e-3,g**3/6-g**5/120+g**7/5040,g-np.sin(g))
    val=np.sqrt(2*c.R*c.R/alpha*np.maximum(diff,0.))
    return np.where(radius<c.R,val,np.nan)
dimension_data=[]
for d in Ds:
    y2=raw2+(d-2)*eta**2
    # Bracket the last nonzero crossing on the computed, densely refined grid.
    idx=np.flatnonzero((y2[:-1]<c.SN**2)&(y2[1:]>=c.SN**2))
    j=int(idx[-1]);cross=brentq(lambda e:float(rawfun(e))+(d-2)*e*e-c.SN*c.SN,eta[j],eta[j+1])
    lo,hi=np.sin(alpha)/cdim(d),1/cdim(d)
    overlap=[lo,min(hi,cross/c.R)] if lo<cross/c.R else None
    lowereta=c.R*lo;atlower=float(rawfun(lowereta))+(d-2)*lowereta**2
    dimension_data.append(dict(D=d,c=cdim(d),topology_interval_eta_over_R=[lo,hi],raw_below_benchmark_interval_eta_over_R=[0.,cross/c.R],overlap_eta_over_R=overlap,raw_over_SN_at_topology_lower=np.sqrt(atlower)/c.SN,A_raw_over_SN=np.sqrt(A['raw_w2']**2+(d-2)*A['eta']**2)/c.SN,B_raw_over_SN=np.sqrt(B['raw_w2']**2+(d-2)*B['eta']**2)/c.SN))
gchecks=[]
for rr in [0.,.1,.2,.35,c.R*np.sin(alpha),.8,1.8]:
    gap=max(alpha-np.arcsin(rr/c.R),0.)
    direct=quad(lambda t:2*c.R*c.R*(1-np.cos(t))/alpha,0,gap,epsabs=1e-14)[0]
    closed=float(diversity(rr/cdim(2)))**2
    gchecks.append(dict(r=rr,integral_squared=direct,formula_squared=closed,difference=abs(direct-closed)))

C=dict(raw='#0072B2',tw='#D55E00',tangent='#009E73',normal='#E69F00',diversity='#CC79A7',ink='#26343D',gray='#7E8B94',support='#B5BDC2',grid='#E7ECEF',shade='#EDF5F2')
SIZE=(6.,4.35)
YSHIFT=0.
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9.2,'axes.labelsize':9.2,'xtick.labelsize':9.2,'ytick.labelsize':9.2,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.color':C['ink'],'axes.labelcolor':C['ink'],'xtick.color':C['ink'],'ytick.color':C['ink'],'savefig.facecolor':'white'})
fig=plt.figure(figsize=SIZE)
def axes(xx,y,w,h):return fig.add_axes([xx/6,(y-YSHIFT)/SIZE[1],w/6,h/SIZE[1]])
def text(xx,y,s,**kw):return fig.text(xx/6,(y-YSHIFT)/SIZE[1],s,**kw)
def arrow(x0,y0,x1,y1):
    fig.add_artist(FancyArrowPatch((x0/6,(y0-YSHIFT)/SIZE[1]),(x1/6,(y1-YSHIFT)/SIZE[1]),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=6,lw=.65,color=C['gray']))
cloud=np.load(ROOT/'snapshot_data.npz');rawA=cloud['raw_A'];rawB=c.P[cloud['labels']]+B['eta']*cloud['base_noise'];twA=c.tweedie(rawA,A['eta']);twB=c.tweedie(rawB,B['eta'])
angle=np.linspace(0,2*np.pi,721);patch_counts={}
text(.05,1.53,'(c)',weight='bold')
for label,row,raw,tw,left,title in [('A',A,rawA,twA,.38,'Early stop'),('B',B,rawB,twB,2.35,'Late stop')]:
    center=left+.72
    text(center,1.53,label+'  '+title,ha='center',weight='bold')
    text(center,1.35,rf'$\eta/R={row["eta"]/c.R:.3g}$',ha='center')
    fig.add_artist(Line2D([left/6,(left+1.45)/6],[(1.26-YSHIFT)/SIZE[1]]*2,transform=fig.transFigure,c=C['grid'],lw=.75))
    local=raw-np.array([0.,c.R]);localtw=tw-np.array([0.,c.R]);ids=np.flatnonzero((abs(local[:,0])<1.25)&(local[:,1]>-.7)&(local[:,1]<.55));patch_counts[label]=len(ids)
    for off,points,lp,col,name in [(0.,raw,local,C['raw'],'Raw'),(.89,tw,localtw,C['tw'],'Tweedie')]:
        text(left+off+.28,1.10,name,ha='center',color=col)
        ax=axes(left+off-.04,.71,.64,.30);xx=np.linspace(-1.45,1.45,301)
        ax.plot(xx,np.sqrt(c.R*c.R-xx*xx)-c.R,c=C['support'],lw=.6)
        ax.scatter(lp[ids,0],lp[ids,1],c=col,s=2.6 if name=='Tweedie' else 1.4,alpha=.9 if name=='Tweedie' else .7,lw=0,zorder=3)
        tp=c.P-np.array([0.,c.R]);ax.scatter(tp[:,0],tp[:,1],c=C['gray'],marker='+',s=7,lw=.35,alpha=.65,zorder=2)
        ax.set(xlim=(-1.45,1.45),ylim=(-.82,.65),aspect='equal');ax.axis('off')
        ax=axes(left+off-.04,.05,.64,.64)
        ax.plot(np.cos(angle),np.sin(angle),c=C['support'],lw=.55)
        ax.scatter(points[:,0]/c.R,points[:,1]/c.R,c=col,s=1.7 if name=='Tweedie' else .8,alpha=.9 if name=='Tweedie' else .65,lw=0,zorder=3)
        ax.scatter(c.P[:,0]/c.R,c.P[:,1]/c.R,c=C['gray'],marker='+',s=5,lw=.3,alpha=.65,zorder=2)
        ax.set(xlim=(-1.65,1.65),ylim=(-1.65,1.65),aspect='equal');ax.axis('off')
    arrow(left+.62,.86,left+.80,.86);arrow(left+.62,.37,left+.80,.37)

# Main D=2 error curves and the geometric diversity deficit.
text(2.22,3.85,'(a) Wasserstein error',ha='center',weight='bold')
ax=axes(.48,2.03,3.48,1.62)
lo,hi=dimension_data[0]['topology_interval_eta_over_R']
ax.axvspan(lo,hi,color=C['shade'],ec='none',zorder=0)
ax.text((lo+hi)/2,3.34,'Circle topology',ha='center',color=C['tangent'])
ax.text(.39,2.85,r'$D=2$',ha='left',color=C['ink'])
ax.axhline(1,color=C['gray'],ls=(0,(4,3)),lw=.8,zorder=1)
ax.text(.005,1.055,r'$S_N$',ha='right',va='bottom',color=C['gray'])
handles=[]
for field,col,ls,lw,label in [('raw_w2','raw','-',1.6,r'$W_2$ raw'),('raw_wt','tangent','-',1.5,r'$W_T$'),('raw_wn','normal',(0,(5,2)),1.35,r'$W_N$'),('tweedie_w2','tw','-.',1.6,r'$W_2$ denoised')]:
    line,=ax.plot(x,[r[field]/c.SN for r in rows],c=C[col],ls=ls,lw=lw,label=label,zorder=3);handles.append(line)
gx=np.linspace(0,1/cdim(2),1000,endpoint=False)
line,=ax.plot(gx,diversity(c.R*gx)/c.SN,c=C['diversity'],ls=(0,(1.5,1.8)),lw=1.55,label=r'$\mathcal{G}(r)$',zorder=4);handles.append(line)
shared_legend=fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,(4.15-YSHIFT)/SIZE[1]),ncol=5,frameon=False,fontsize=9.2,handlelength=1.8,handletextpad=.5,columnspacing=1.15)
for label,row,field,col,offset in [('A',A,'raw_w2','raw',(-12,7)),("A$'$",A,'tweedie_w2','tw',(6,6)),('B',B,'raw_w2','raw',(-31,-1)),("B$'$",B,'tweedie_w2','tw',(-31,10))]:
    xx=row['eta']/c.R;yy=row[field]/c.SN
    ax.plot(xx,yy,'o',ms=3.4,mfc='white',mec=C[col],mew=1.,zorder=5)
    label_options=dict(arrowprops=dict(arrowstyle='-',lw=.5,color=C[col])) if label.startswith('B') else {}
    ax.annotate(label,(xx,yy),xytext=offset,textcoords='offset points',color=C[col],**label_options)
for row in [A,B]:ax.plot([row['eta']/c.R]*2,[row['raw_w2']/c.SN,row['tweedie_w2']/c.SN],c=C['gray'],ls=':',lw=.8,zorder=2)
ax.set(xlim=(.4,0),ylim=(-.035,3.6),xticks=[.4,.2,0],yticks=[0,1,2,3]);ax.set_xticklabels(['0.4','0.2','0'])
ax.set_ylabel(r'Wasserstein error / $S_N$',labelpad=5);ax.set_xlabel(r'Noise $\eta/R$ (decreasing $\rightarrow$)',labelpad=4)
ax.grid(axis='y',color=C['grid'],lw=.5);ax.set_axisbelow(True);ax.tick_params(length=2.5,width=.6,pad=2)

# Miniature copies of the main plot, one ambient dimension per panel.
fig.add_artist(Line2D([4.16/6]*2,[(.05-YSHIFT)/SIZE[1],(3.92-YSHIFT)/SIZE[1]],transform=fig.transFigure,c=C['grid'],lw=.8))
text(5.11,3.85,'(b) Ambient dimension',ha='center',weight='bold')
text(5.11,3.63,r'Same circle in $\mathbb{R}^D$',ha='center')
small_panels=[]
raw_wn2=np.array([r['raw_wn']**2 for r in rows])
wt=np.array([r['raw_wt'] for r in rows])/c.SN
den=np.array([r['tweedie_w2'] for r in rows])/c.SN
for i,dd in enumerate(dimension_data[1:]):
    d=dd['D'];bottom=2.64-1.12*i
    dax=axes(4.51,bottom,1.36,.88)
    lower,upper=dd['topology_interval_eta_over_R']
    dax.axvspan(lower,upper,color=C['shade'],ec='none',zorder=0)
    dax.axhline(1,c=C['gray'],ls=(0,(4,3)),lw=.7,zorder=1)
    raw=np.sqrt(raw2+(d-2)*eta**2)/c.SN
    wn=np.sqrt(raw_wn2+(d-2)*eta**2)/c.SN
    for yy,col,ls,lw in [(raw,'raw','-',1.3),(wt,'tangent','-',1.2),(wn,'normal',(0,(5,2)),1.15),(den,'tw','-.',1.3)]:
        dax.plot(x,yy,c=C[col],ls=ls,lw=lw,zorder=3)
    gx_d=np.linspace(0,1/cdim(d),1000,endpoint=False)
    dax.plot(gx_d,diversity(c.R*gx_d,d)/c.SN,c=C['diversity'],ls=(0,(1.5,1.8)),lw=1.2,zorder=4)
    top=[7.,10.,14.][i]
    assert max(raw.max(),wn.max(),wt.max(),den.max())<top
    dax.set(xlim=(.4,0),ylim=(-.025*top,top),xticks=[.4,.2,0],yticks=[0,top/2,top])
    dax.set_xticklabels(['0.4','0.2','0'])
    dax.set_yticklabels([f'{v:g}' for v in [0,top/2,top]])
    dax.text(.97,.84,rf'$D={d}$',transform=dax.transAxes,ha='right',va='top',color=C['ink'])
    dax.grid(axis='y',c=C['grid'],lw=.45);dax.set_axisbelow(True);dax.tick_params(length=2.2,width=.6,pad=2)
    if i==2:dax.set_xlabel(r'Noise $\eta/R$',labelpad=4)
    small_panels.append(dict(D=d,eta_over_R_limits=[.4,0.],y_limits=[-.025*top,top],y_ticks=[0,top/2,top],raw_max=float(raw.max()),no_curve_clipping=True))

fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
for txt in fig.findobj(Text):
    if txt.get_visible() and txt.get_text():
        bb=txt.get_window_extent(renderer)
        if bb.width and bb.height and (bb.x0<0 or bb.y0<0 or bb.x1>fig.bbox.width or bb.y1>fig.bbox.height):outside.append(txt.get_text())
assert not outside,outside
legend_text_boxes=[t.get_window_extent(renderer) for t in shared_legend.get_texts()]
assert max(bb.y0 for bb in legend_text_boxes)-min(bb.y0 for bb in legend_text_boxes)<4
for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'figure_02_uniform_dimension_small_multiples.{ext}',dpi=300)
result=dict(dimensions=dimension_data,G_formula='G(r)^2=(2 R^2/alpha) (g-sin g), alpha=pi/N, g=max(alpha-arcsin(r/R),0), 0<=r<R',G_integral_checks=gchecks,A={k:A[k]/c.SN for k in ['raw_w2','raw_wt','raw_wn','tweedie_w2','tweedie_wn']},B={k:B[k]/c.SN for k in ['raw_w2','raw_wt','raw_wn','tweedie_w2','tweedie_wn']},B_eta_over_R=B['eta']/c.R,common_eta_over_R=.175,D_identity='W2_D^2=W2_2^2+(D-2)eta^2; WN_D^2=WN_2^2+(D-2)eta^2; WT and empirical-Tweedie law independent of D',geometry='Fixed one-dimensional circle embedded in R^D; independent isotropic Gaussian in each added orthogonal coordinate',interval_method='Topology intervals exact; W2<SN crossing estimated by PCHIP of refined numerical transport, not a certified continuum inequality',font='STIX 9.2 pt at 6-inch width',patch_counts=patch_counts,outside_canvas_text=outside)
(ROOT/'dimension_comparison_audit.json').write_text(json.dumps(result,indent=2))
result['layout']=dict(size_inches=SIZE,right_column='Three small multiples of all five main-panel curves for D=4,8,16; D=2 appears only in the main panel',small_panels=small_panels,shared_legend_rows=1,legend_text_bounds=[list(bb.bounds) for bb in legend_text_boxes],old_raw_overlay_and_interval_bars_removed=True,ordinary_font_sizes_pt=sorted(set(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text())),minimum_ordinary_font_pt=9.2)
result['B_selection']=dict(eta=.04,eta_over_R=.02,original_role='Later stop: empirical Tweedie increases total error relative to the same-time raw law',additional_current_condition='The calibrated sample neighborhood has 16 disconnected components and G(r)>0',historical_algorithm=False,source_audit='B_selection_report.txt',G_over_SN=float(diversity(.04))/c.SN,denoised_benchmark_interpretation='Approximately S_N; the tiny numerical deficit below S_N is target quadrature bias and not evidence of a strict gain')
(ROOT/'dimension_comparison_audit.json').write_text(json.dumps(result,indent=2))
with (ROOT/'dimension_curves.csv').open('w',newline='') as f:
    fields=['D','eta','eta_over_R','r_over_R','raw_w2','raw_wn','raw_wt','denoised_w2','G','topology_recovered','raw_below_SN']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for d in Ds:
        for row in rows:
            e=row['eta'];radius=cdim(d)*e;raw=np.sqrt(row['raw_w2']**2+(d-2)*e*e)
            writer.writerow(dict(D=d,eta=e,eta_over_R=e/c.R,r_over_R=radius/c.R,raw_w2=raw,raw_wn=np.sqrt(row['raw_wn']**2+(d-2)*e*e),raw_wt=row['raw_wt'],denoised_w2=row['tweedie_w2'],G=float(diversity(e,d)),topology_recovered=int(c.R*np.sin(alpha)<=radius<c.R),raw_below_SN=int(raw<c.SN)))
print(json.dumps(result,indent=2))

for ext in ['pdf','png','svg']:
    shutil.copy2(ROOT/f'figure_02_uniform_dimension_small_multiples.{ext}',ROOT/f'figure_02_stopping_denoising_uniform.{ext}')
