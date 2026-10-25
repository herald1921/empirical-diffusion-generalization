"""Six-panel MNIST figure at the final 6-inch publication width.

All 56 displayed images and every plotted statistic use the existing frozen
arrays. Only panel placement, labels and tick density are changed.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'appendix_d_images/figure09/frozen_data';O=ROOT/'publication/output';O.mkdir(exist_ok=True)
z=np.load(D/'plot_data.npz');field=list(csv.DictReader((D/'field_discrepancies.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','font.size':9.3,'axes.labelsize':9.3,'xtick.labelsize':9.3,'ytick.labelsize':9.3,'legend.fontsize':9.3,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
W,H=6,7.25;fig=plt.figure(figsize=(W,H));NS=[100,1000,10000];METHODS=['empirical','mlp','unet'];names=['Empirical','MLP','U-Net'];colors=['#0072B2','#7A8994','#008578'];nc=['#318AC3','#4F6CC1','#7352AD'];images=[];curves=[]
def txt(x,y,s,**kw):return fig.text(x/W,y/H,s,fontsize=9.3,va='center',**kw)
def ax(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
def title(x,y,s):txt(x,y,s,weight='bold')
def im(x,y,s,a):
 a=np.asarray(a);images.append(hashlib.sha256(a.tobytes()).hexdigest());p=ax(x,y,s,s);p.imshow(a.reshape(28,28).clip(0,1),cmap='gray',vmin=0,vmax=1,interpolation='nearest');p.axis('off')
def frame(p,lim):
 p.set(xlim=(0,1.025),ylim=lim);p.set_xticks([0,.5,1],['0','.5','1']);p.grid(axis='y',color='#e4e9ed',lw=.5);p.tick_params(length=2,pad=2)
def rule(y):fig.add_artist(Line2D([.02,.98],[y/H,y/H],transform=fig.transFigure,color='#dce3e7',lw=.5))
title(.08,7.09,'(a) Empirical readouts');title(3.13,7.09,'(b) Posterior ESS')
s=.36;xs=[.69,1.18,1.67,2.16]
for x,t in zip(xs,z['snapshot_tau']):txt(x+s/2,6.79,f'{t:.3f}' if t==.999 else f'{t:.2f}',ha='center',color='#D55E00' if t==.25 else '#263742')
for i,n in enumerate(NS):
 y=6.29-i*.4;txt(.59,y+s/2,f'{n:,}',ha='right')
 for j,x in enumerate(xs):im(x,y,s,z['panel_a'][i,j])
txt(.48,6.79,r'$N$',ha='right');txt(2.70,6.79,r'$\tau$',ha='center')
fig.add_artist(Rectangle(((xs[1]-.025)/W,(5.49-.025)/H),(.36+.05)/W,(1.16+.05)/H,transform=fig.transFigure,fill=False,ec='#D55E00',lw=.9))
p=ax(3.56,5.64,2.20,1.08)
for n,c,ls in zip(NS,nc,[':','--','-']):
 lo,md,hi=z[f'ess_quantiles_N{n}'];p.fill_between(z['tau'],lo,hi,color=c,alpha=.14,lw=0);p.plot(z['tau'],md,color=c,ls=ls,lw=1.1,label=f'{n:,}')
frame(p,(.85,16000));p.set_yscale('log');p.set_yticks([1,100,10000]);p.minorticks_off();p.axvline(.25,color='#D55E00',ls=':',lw=.8);p.set_ylabel('ESS');p.set_xlabel(r'FM time $\tau$',labelpad=2);p.legend(loc='upper right',frameon=False,handlelength=1.3,labelspacing=.15)
rule(5.25);title(.08,5.06,'(c) Training size and generation method')
starts=[.69,2.42,4.15];s=.35;offs=[0,.36,.78,1.14]
for j,kind in enumerate(METHODS):
 x=starts[j];txt(x+.75,4.78,names[j],ha='center');txt(x+.75,4.56,r'$\tau=.25$' if j==0 else r'$\tau=.999$',ha='center',color='#D55E00' if j==0 else '#62717d')
 txt(x+.35,4.34,r'$S_1$',ha='center');txt(x+1.13,4.34,r'$S_2$',ha='center')
 for k,label in enumerate(['NN','Out','Out','NN']):txt(x+offs[k]+s/2,4.13,label,ha='center')
 for i,n in enumerate(NS):
  y=3.72-i*.39
  if j==0:txt(.59,y+s/2,f'{n:,}',ha='right')
  for k,off in enumerate(offs):im(x+off,y,s,z['panel_c'][i,j,k])
rule(2.82);title(.08,2.63,'(d) Training/reference distances and gaps')
handles=[Line2D([],[],color=c,label=n) for c,n in zip(colors,names)]+[Line2D([],[],color='#62717d',label='Train'),Line2D([],[],color='#62717d',ls='--',label='Reference')]
fig.legend(handles=handles,ncol=5,loc='center',bbox_to_anchor=(.54,2.39/H),frameon=False,columnspacing=.8,handlelength=1.2,handletextpad=.3)
for j,n in enumerate(NS):
 x=.52+j*1.85;pd=ax(x,1.69,1.55,.44);pg=ax(x,1.12,1.55,.44);frame(pd,(0,.29));frame(pg,(-.012,.27));pd.tick_params(labelbottom=False);pd.set_title(f'N = {n:,}',fontsize=9.3,pad=3)
 for p in [pd,pg]:p.set_yticks([0,.2],['0','.2'])
 pg.axhline(0,color='#98a7b1',lw=.5)
 for kind,c in zip(METHODS,colors):
  v=z[f'{kind}_N{n}_means'];ci=z[f'{kind}_N{n}_ci95']
  for k,ls in [(0,'-'),(1,'--')]:pd.fill_between(z['tau'],ci[k,0],ci[k,1],color=c,alpha=.09,lw=0);pd.plot(z['tau'],v[k],color=c,ls=ls,lw=1)
  pg.fill_between(z['tau'],ci[2,0],ci[2,1],color=c,alpha=.12,lw=0);pg.plot(z['tau'],v[2],color=c,lw=1)
 if j==0:pd.set_ylabel('RMSE',labelpad=2);pg.set_ylabel('Gap',labelpad=2)
txt(3.13,.87,r'FM time $\tau$',ha='center')
# The remaining two panels are exported separately below the first block, at
# the same figure number, so their curves and labels retain readable sizes.
for p in fig.axes:
 for l in p.lines:
  if len(l.get_xdata())>2:curves.append([np.asarray(l.get_xdata()).tolist(),np.asarray(l.get_ydata()).tolist()])
fig.savefig(O/'figure_09_mnist_comparison.pdf',bbox_inches=matplotlib.transforms.Bbox.from_bounds(0,.65,6,6.60));fig.savefig(O/'figure_09_mnist_comparison.png',dpi=180,bbox_inches=matplotlib.transforms.Bbox.from_bounds(0,.65,6,6.60));plt.close(fig)
fig=plt.figure(figsize=(6,2.45));W,H=6,2.45
title(.08,2.27,'(e) Shared-noise generations');title(3.15,2.27,'(f) Common-input fields')
txt(1.47,2.02,'Each pair: MLP / U-Net',ha='center')
s=.45
for seed in range(4):
 x=.20+(seed%2)*1.31;y=1.30-(seed//2)*.56
 for j in range(2):im(x+j*.47,y,s,z['panel_e'][j,seed])
p=ax(3.55,.65,2.20,1.27)
for pair,label,c,ls in [('mlp_unet','MLP vs U-Net','#008578','-'),('empirical_mlp','Empirical vs MLP','#82929F','--'),('empirical_unet','Empirical vs U-Net','#C17E47','-.')]:
 rows=[r for r in field if r['pair']==pair];p.plot([float(r['tau']) for r in rows],[float(r['scaled_field_rmse']) for r in rows],color=c,ls=ls,marker='o',ms=2,lw=1,label=label)
frame(p,(0,.175));p.set_yticks([0,.05,.10,.15]);p.set_xlabel(r'FM time $\tau$',labelpad=2);p.set_ylabel('Scaled RMSE',labelpad=2)
fig.legend(*p.get_legend_handles_labels(),ncol=3,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False,handlelength=1.5,columnspacing=1.1,handletextpad=.4)
fig.savefig(O/'figure_09_mnist_comparison_continued.pdf');fig.savefig(O/'figure_09_mnist_comparison_continued.png',dpi=180);plt.close(fig)
assert len(images)==56
(O/'figure09_data_audit.json').write_text(json.dumps({'image_count':56,'image_array_hashes':images,'input_sha256':hashlib.sha256((D/'plot_data.npz').read_bytes()).hexdigest(),'field_csv_sha256':hashlib.sha256((D/'field_discrepancies.csv').read_bytes()).hexdigest(),'new_simulation':False,'panels':'a-d; e-f continuation','curves':curves},indent=2))
