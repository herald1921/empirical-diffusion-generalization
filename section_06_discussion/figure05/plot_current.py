"""Current two-panel Figure 5, using all validated paired seeds and times."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'validated_data';OUT=ROOT/'output';OUT.mkdir(parents=True,exist_ok=True)
meta=json.loads((DATA/'provenance.json').read_text());path=DATA/'figure5_inputs.npz'
assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
z=np.load(path);tau=z['tau'];ns=z['n_values'];assert len(ns)==6 and len(tau)==32
assert z['d_train'].shape==z['d_ref'].shape==(2,6,2,32,256)
assert z['images'].shape==z['nearest_train'].shape==(2,6,2,6400)
assert np.array_equal(z['seeds'],np.arange(256))
colors=['#21a477','#12a3b5','#2e8bd0','#4b69cb','#7551bd','#933ca5']
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,
 'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.6,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
 'text.color':'#263b49','axes.labelcolor':'#263b49','xtick.color':'#263b49','ytick.color':'#263b49'})
fig=plt.figure(figsize=(7.2,4.65))
bootstrap=np.random.default_rng(22092027).integers(0,256,(2000,256));stats=[]
for mi,(name,subtitle,left) in enumerate([('Empirical',r'Early stop $\tau=0.15$',.105),('U-Net',r'Terminal $\tau=0.999$',.565)]):
 width=.407;cell=width/6;height=cell*7.2/4.65
 fig.text(left+width/2,.966,f'({chr(97+mi)}) {name}',ha='center',va='top',fontsize=13,fontweight='bold')
 fig.text(left+width/2,.909,subtitle,ha='center',va='top',fontsize=10.5)
 for ni,n in enumerate(ns):
  x=left+ni*cell
  fig.text(x+cell/2,.849,f'{n:,}',ha='center',va='bottom',fontsize=9,color=colors[ni])
  for ri,(key,gi) in enumerate([('nearest_train',0),('images',0),('images',1),('nearest_train',1)]):
   ax=fig.add_axes([x,.835-(ri+1)*height,cell*.98,height*.98])
   ax.imshow(z[key][mi,ni,gi].reshape(80,80).clip(0,1),cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_axis_off()
   if mi==0 and ni==0:fig.text(left-.012,.835-(ri+.5)*height,[r'Nearest $S_1$',r'Output $S_1$',r'Output $S_2$',r'Nearest $S_2$'][ri],ha='right',va='center',fontsize=9)
 dist=fig.add_axes([left,.235,width,.13]);gapax=fig.add_axes([left,.078,width,.115])
 for ni,n in enumerate(ns):
  train=z['d_train'][mi,ni].mean(0);ref=z['d_ref'][mi,ni].mean(0);gap=ref-train
  ci=np.quantile(gap[:,bootstrap].mean(-1),[.025,.975],axis=1)
  dist.plot(tau,train.mean(-1),color=colors[ni],lw=1.35)
  dist.plot(tau,ref.mean(-1),'--',color=colors[ni],lw=1.35)
  gapax.fill_between(tau,ci[0],ci[1],color=colors[ni],alpha=.15,lw=0)
  gapax.plot(tau,gap.mean(-1),color=colors[ni],lw=1.35)
  for ax,values in [(dist,train.mean(-1)),(dist,ref.mean(-1)),(gapax,gap.mean(-1))]:ax.plot(tau[-1],values[-1],'o',ms=2.7,color=colors[ni])
  stats.append({'method':name,'n':int(n),'gap_mean':gap.mean(-1).tolist(),'gap_ci95':ci.tolist()})
 for ax in [dist,gapax]:
  ax.set_xlim(0,1);ax.set_ylim(-.004,.35);ax.set_yticks([0,.1,.2,.3]);ax.grid(axis='y',color='#dce3e7',lw=.6);ax.set_axisbelow(True);ax.tick_params(length=3,width=.6)
  if mi==0:ax.axvline(.15,color='#e58330',ls='--',lw=.8)
  else:ax.set_yticklabels([])
 dist.set_xticks([0,.25,.5,.75,1],[])
 gapax.set_xticks([0,.25,.5,.75,1],['0','.25','.5','.75','1']);gapax.axhline(0,color='#83939b',ls=':',lw=.65)
 if mi==0:dist.set_ylabel('Pixel RMSE');gapax.set_ylabel('Gap')
fig.text(.05,.858,r'$N$',fontsize=11)
fig.text(.55,.009,r'Linear FM time $\tau$',ha='center',fontsize=11)
fig.legend(handles=[Line2D([],[],color='#263b49',lw=1.2,label=r'$d_{\rm train}$'),Line2D([],[],color='#263b49',lw=1.2,ls='--',label=r'$d_{\rm ref}$'),Line2D([],[],color='#e58330',lw=1.2,ls='--',label='Empirical stop')],loc='center',bbox_to_anchor=(.55,.39),frameon=False,ncol=3,fontsize=9.5,handlelength=2.3)
fig.savefig(OUT/'figure_05_celeba_empirical_unet.pdf');fig.savefig(OUT/'figure_05_celeba_empirical_unet.png',dpi=220);plt.close(fig)
(OUT/'statistics.json').write_text(json.dumps({'bootstrap_unit':'Gaussian seed after averaging the two directions','resamples':2000,'seed':22092027,'curves':stats},indent=2))
print('Current Figure 5 reproduced from exhaustive-float64 validated readouts.')
