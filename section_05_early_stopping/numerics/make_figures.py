"""One main curve axis per figure, with small data-linked geometric insets.
python make_figures.py
"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parent
D=np.load(ROOT/'data/inset_data.npz');P=D['training_points'];R=2.
A=json.loads((ROOT/'data/inset_audit.json').read_text())
C=dict(raw='#0072B2',tw='#D55E00',good='#009E73',ink='#253340',gray='#78858D',support='#B5BDC2',grid='#E9ECEE',frame='#D4DCE0')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white','savefig.dpi':350,'text.color':C['ink'],'axes.labelcolor':C['ink']})

def read(name):
    with (ROOT/'data'/name).open(newline='',encoding='utf-8') as f:r=list(csv.DictReader(f))
    return {k:np.array([float(v[k]) for v in r]) for k in r[0]}

def main_axis():
    fig=plt.figure(figsize=(6.75,4.2));ax=fig.add_axes([.12,.15,.85,.75])
    ax.grid(axis='y',lw=.6,color=C['grid']);ax.set_axisbelow(True)
    ax.tick_params(length=3,width=.7);ax.set_xlabel(r'Reverse time $t$')
    return fig,ax

def save(fig,name):
    for fmt in ('pdf','png','svg'):
        path=ROOT/'output'/fmt;path.mkdir(parents=True,exist_ok=True);fig.savefig(path/(name+'.'+fmt))
    plt.close(fig)

def card(ax,bounds,title,color=C['ink']):
    inset=ax.inset_axes(bounds,zorder=8);inset.set_facecolor('white');inset.set_xticks([]);inset.set_yticks([])
    for sp in inset.spines.values():sp.set_visible(True);sp.set_color(C['frame']);sp.set_linewidth(.6)
    inset.text(.5,.955,title,ha='center',va='top',transform=inset.transAxes,fontsize=7.7,color=color)
    return inset

def circle(ax,x,color,limit=3.2,endpoint=False,size=1.05):
    theta=np.linspace(0,2*np.pi,600);ax.plot(R*np.cos(theta),R*np.sin(theta),color=C['support'],lw=.55,zorder=1)
    if endpoint:ax.scatter(P[:,0],P[:,1],s=6,c=C['ink'],lw=0,zorder=3)
    else:
        ax.scatter(x[:,0],x[:,1],s=size,c=color,alpha=.66,lw=0,zorder=2)
        ax.scatter(P[:,0],P[:,1],s=5,marker='+',c=C['ink'],lw=.4,zorder=3)
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal');ax.axis('off')

def link(ax,box,origin,point,color=C['gray']):
    con=ConnectionPatch(xyA=origin,coordsA=box.transAxes,xyB=point,coordsB=ax.transData,color=color,lw=.85,arrowstyle='-',zorder=6)
    ax.add_artist(con)

def pair_cloud_card(ax,bounds,key,title,color):
    box=card(ax,bounds,title,color)
    box.text(.22,.78,'Before',ha='center',fontsize=6.2,color=C['raw']);box.text(.78,.78,'After',ha='center',fontsize=6.2,color=C['tw'])
    circle(box.inset_axes([.035,.04,.4,.68]),D['raw_'+key],C['raw'])
    circle(box.inset_axes([.565,.04,.4,.68]),D['tweedie_'+key],C['tw'])
    box.add_patch(FancyArrowPatch((.45,.37),(.55,.37),transform=box.transAxes,arrowstyle='-|>',mutation_scale=6,lw=.7,color=C['gray']))
    return box

def figure03(d):
    fig,ax=main_axis();s=d['endpoint'][0]
    ax.plot(np.r_[d['t'],1],np.r_[d['raw_wt'],s],color=C['good'],lw=2.2,zorder=3)
    ax.axhline(s,color=C['gray'],ls=(0,(4,3)),lw=.95,zorder=1)
    ax.set(xlim=(.55,1.055),ylim=(-.006,.32),ylabel=r'Tangential error $W_T$')
    ax.set_xticks([.6,.8,1]);ax.set_yticks([0,.1,.2])
    for key,bounds,title,color in [('noisy',[.015,.34,.20,.27],'Noisy cloud',C['raw']),('minimum',[.282,.34,.20,.27],'Early stop',C['good']),('endpoint',[.758,.73,.20,.27],'Empirical endpoint',C['ink'])]:
        box=card(ax,bounds,title,color)
        points=D['endpoint'] if key=='endpoint' else D['inset_'+key]
        circle(box.inset_axes([.035,.035,.93,.77]),points,color,limit=4.2,endpoint=key=='endpoint',size=.8)
        point=(1,s) if key=='endpoint' else (A[key]['t'],A[key]['wt'])
        ax.plot(*point,'o',color=color,ms=4,zorder=5)
        link(ax,box,(.5,0),point)
    save(fig,'figure_03_early_stopping_with_insets')

def figure04():
    fig,ax=main_axis();n=read('figure04_display_normal.csv')
    ax.plot(n['t'],n['raw_wn'],color=C['raw'],lw=2,label='Before Tweedie')
    ax.plot(n['t'],n['tweedie_wn'],color=C['tw'],lw=2,label='After Tweedie')
    ax.set_yscale('log');ax.set(xlim=(.43,.945),ylim=(1e-4,2),ylabel=r'Normal error $W_N$')
    ax.set_xticks([.45,.6,.75,.9]);ax.set_yticks([1e-4,1e-2,1]);ax.minorticks_off()
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,frameon=False,handlelength=2)
    box=card(ax,[.29,.13,.46,.28],'Normal contraction')
    ids=D['normal_patch_ids'];basis=D['local_basis']
    for rect,key,color,label in [([.035,.045,.40,.63],'raw_a',C['raw'],'Before'),([.565,.045,.40,.63],'tweedie_a',C['tw'],'After')]:
        local=D[key][ids]@basis.T-np.array([0,R]);points=P@basis.T-np.array([0,R])
        panel=box.inset_axes(rect);x=np.linspace(-1.45,1.45,300)
        panel.plot(x,np.sqrt(R*R-x*x)-R,lw=.65,color=C['support'])
        panel.scatter(local[:,0],local[:,1],s=1.7,c=color,alpha=.7,lw=0)
        panel.scatter(points[:,0],points[:,1],s=7,c=C['ink'],marker='+',lw=.4)
        panel.set(xlim=(-1.45,1.45),ylim=(-.82,.65),aspect='equal');panel.axis('off')
        box.text(rect[0]+rect[2]/2,.72,label,ha='center',fontsize=6.5,color=color)
    box.add_patch(FancyArrowPatch((.45,.35),(.55,.35),transform=box.transAxes,arrowstyle='-|>',mutation_scale=6,lw=.7,color=C['gray']))
    j=int(np.argmin(abs(n['eta']-.35)));t=n['t'][j];before=n['raw_wn'][j];after=n['tweedie_wn'][j]
    ax.plot([t,t],[after,before],color=C['gray'],ls=':',lw=.85,zorder=2)
    ax.plot(t,before,'o',color=C['raw'],ms=4,zorder=5);ax.plot(t,after,'o',color=C['tw'],ms=4,zorder=5)
    link(ax,box,(.97,1),(t,after))
    save(fig,'figure_04_normal_contraction_with_inset')

def figure05(d):
    fig,ax=main_axis();t=np.r_[d['t'],1.];endpoint=float(d['endpoint'][0])
    before=np.r_[d['raw_w2'],endpoint];after=np.r_[d['tweedie_w2'],endpoint]
    ax.plot(t,before,color=C['raw'],lw=2.2,label='Before Tweedie',zorder=3)
    ax.plot(t,after,color=C['tw'],lw=2.2,label='After Tweedie',zorder=3)
    ax.axhline(endpoint,color=C['gray'],ls=(0,(4,3)),lw=1.1,label='Empirical endpoint',zorder=1)
    ax.set(xlim=(.64,1.015),ylim=(0,.75),ylabel=r'Total error $W_2$')
    ax.set_xticks([.65,.8,1]);ax.set_yticks([0,.2,.4,.6])
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),ncol=3,frameon=False,handlelength=2,columnspacing=1.5)
    for key,eta,bounds,title,color,origin,anchor in [
        ('a',.35,[.56,.68,.415,.28],'Earlier stop: helps',C['good'],(0,.35),'before'),
        ('b',.16,[.56,.37,.415,.28],'Later stop: harms',C['tw'],(.33,0),'after')]:
        box=pair_cloud_card(ax,bounds,key,title,color)
        j=int(np.argmin(abs(d['eta']-eta)));x=d['t'][j];b=d['raw_w2'][j];a=d['tweedie_w2'][j]
        ax.plot([x,x],[a,b],color=C['gray'],ls=':',lw=.9,zorder=2)
        ax.plot(x,b,'o',color=C['raw'],ms=4,zorder=5)
        ax.plot(x,a,'o',color=C['tw'],ms=4,zorder=5)
        link(ax,box,origin,(x,b if anchor=='before' else a),C['gray'])
    save(fig,'figure_05_total_error_with_insets')

if __name__=='__main__':
    d=read('figure05_curves.csv');order=np.argsort(d['t']);d={k:v[order] for k,v in d.items()}
    figure03(d);figure04();figure05(d);print('Created final Figures 3, 4 and 5 in PDF, PNG and SVG.')