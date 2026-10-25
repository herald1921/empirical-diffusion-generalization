"""Readable six-panel layout, from the unchanged published frozen arrays."""
from pathlib import Path
import argparse,csv,sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from publication_style import apply,BLUE,ORANGE,INK,GRAY,LIGHT,METHOD_COLORS,N_COLORS,FIELD_PAIRS
ROOT=Path(__file__).resolve().parent
NS=[100,1000,10000];METHODS=['empirical','mlp','unet']
NAMES={'empirical':'Empirical','mlp':'MLP','unet':'U-Net'}
W,H=7.2,8.65

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'revised');args=ap.parse_args()
 source=ROOT/'frozen_data';z=dict(np.load(source/'plot_data.npz'))
 field=list(csv.DictReader((source/'field_discrepancies.csv').open()))
 apply(plt);fig=plt.figure(figsize=(W,H));quantitative=[]
 def text(x,y,s,fs=7.5,**kw):return fig.text(x/W,y/H,s,fontsize=fs,va='center',**kw)
 def axis(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
 def title(x,y,s):text(x,y,s,8.5,fontweight='bold')
 def image(x,y,s,a):
  ax=axis(x,y,s,s);ax.imshow(a.reshape(28,28).clip(0,1),cmap='gray',vmin=0,vmax=1,
    interpolation='nearest',rasterized=True);ax.set_axis_off();return ax
 def frame(ax,lim,labels=True,stop=True):
  quantitative.append(ax);ax.set(xlim=(0,1.025),ylim=lim)
  ax.set_xticks([0,.25,.5,.75,.999],['0','.25','.50','.75','.999'])
  if not labels:ax.tick_params(labelbottom=False)
  ax.grid(axis='y',color=LIGHT,lw=.5);ax.tick_params(length=2.5,width=.5,pad=2)
  if stop:
   ax.axvline(.25,color=ORANGE,lw=.9,ls=':');ax.axvline(.999,color=GRAY,lw=.65,ls=':')
 def rule(y):fig.add_artist(Line2D([.015,.985],[y/H,y/H],transform=fig.transFigure,color=LIGHT,lw=.6))
 # Panel letters now follow reading order. No examples are reselected.
 title(.07,8.44,'(a) Empirical posterior readouts')
 s=.40;xs=[.76,1.36,1.96,2.56]
 for x,t in zip(xs,z['snapshot_tau']):
  text(x+s/2,8.13,rf'$\tau={t:.3f}$' if t==.999 else rf'$\tau={t:.2f}$',6.7,
    ha='center',color=ORANGE if t==.25 else INK)
 for i,n in enumerate(NS):
  y=7.52-i*.46;text(.64,y+s/2,rf'$N={n:,}$',7,ha='right')
  for j,x in enumerate(xs):image(x,y,s,z['panel_a'][i,j])
 fig.add_artist(Rectangle(((xs[1]-.03)/W,(6.60-.03)/H),(.40+.06)/W,(1.32+.06)/H,
   transform=fig.transFigure,fill=False,edgecolor=ORANGE,lw=.9))
 title(3.68,8.44,'(b) Posterior effective sample size')
 ax=axis(4.02,6.69,2.99,1.36)
 for n,ls in zip(NS,[':','--','-']):
  low,med,high=z[f'ess_quantiles_N{n}']
  ax.fill_between(z['tau'],low,high,color=N_COLORS[n],alpha=.14,lw=0)
  ax.plot(z['tau'],med,color=N_COLORS[n],ls=ls,lw=1.2,label=rf'$N={n:,}$')
 frame(ax,(.85,16000),stop=False);ax.axvline(.25,color=ORANGE,lw=.9,ls=':')
 ax.set_yscale('log');ax.set_yticks([1,10,100,1000,10000]);ax.minorticks_off()
 ax.set_xticks([0,.25,.5,.75,1],['0','.25','.50','.75','1']);ax.set_xlabel(r'Linear FM time $\tau$');ax.set_ylabel('ESS')
 ax.legend(loc='upper right',frameon=False,fontsize=6.7,labelspacing=.2)
 rule(6.34)
 title(.07,6.13,'(c) Training size and generation method')
 s=.39;offsets=[0,.405,.88,1.285];starts=[.79,2.95,5.11]
 for j,kind in enumerate(METHODS):
  x=starts[j];text(x+.8375,5.86,NAMES[kind],8,ha='center')
  text(x+.8375,5.64,r'$\tau=0.25$' if kind=='empirical' else r'$\tau=0.999$',6.8,ha='center',color=ORANGE if kind=='empirical' else GRAY)
  for xx,g in [(x+.4,r'$S_1$'),(x+1.28,r'$S_2$')]:text(xx,5.42,g,7,ha='center')
  for k,label in enumerate(['Nearest','Output','Output','Nearest']):text(x+offsets[k]+s/2,5.23,label,6.4,ha='center')
  for i,n in enumerate(NS):
   y=4.71-i*.44
   if j==0:text(.67,y+s/2,rf'$N={n:,}$',7,ha='right')
   for k,off in enumerate(offsets):image(x+off,y,s,z['panel_c'][i,j,k])
 rule(3.70)
 title(.07,3.49,'(d) Training/reference distances and signed gaps')
 handles=[Line2D([],[],color=METHOD_COLORS[k],lw=1.3,label=NAMES[k]) for k in METHODS]
 handles += [Line2D([],[],color=GRAY,lw=1.2,label='Training'),Line2D([],[],color=GRAY,lw=1.2,ls='--',label='Reference')]
 fig.legend(handles=handles,ncol=5,loc='center',bbox_to_anchor=(.52,3.22/H),frameon=False,
  columnspacing=1.1,handlelength=1.5,handletextpad=.4,fontsize=6.8)
 for j,n in enumerate(NS):
  x=.48+j*2.28;ad=axis(x,2.37,1.98,.56);ag=axis(x,1.70,1.98,.56)
  frame(ad,(0,.29),labels=False);frame(ag,(-.012,.27))
  ad.set_title(rf'$N={n:,}$',fontsize=7.5,pad=5)
  for a in [ad,ag]:a.set_yticks([0,.1,.2],['0','.1','.2'])
  ag.axhline(0,color='#98A7B1',lw=.5)
  for kind in METHODS:
   vals=z[f'{kind}_N{n}_means'];ci=z[f'{kind}_N{n}_ci95'];c=METHOD_COLORS[kind]
   for mi,ls in [(0,'-'),(1,'--')]:
    ad.fill_between(z['tau'],ci[mi,0],ci[mi,1],color=c,alpha=.09,lw=0)
    ad.plot(z['tau'],vals[mi],color=c,lw=1.05,ls=ls)
   ag.fill_between(z['tau'],ci[2,0],ci[2,1],color=c,alpha=.12,lw=0)
   ag.plot(z['tau'],vals[2],color=c,lw=1.1)
  if j==0:ad.set_ylabel('Pixel RMSE',labelpad=4);ag.set_ylabel(r'$d_{\rm ref}-d_{\rm train}$',labelpad=4)
 text(3.72,1.40,r'Linear FM time $\tau$',7.5,ha='center');rule(1.22)
 title(.07,1.05,'(e) Shared-noise generations')
 s=.32;pairx=[.18,.90,1.62,2.34]
 for x in pairx:
  for j,k in enumerate(['mlp','unet']):text(x+j*.33+s/2,.79,NAMES[k],6.5,ha='center')
 for seed,x in enumerate(pairx):
  for j in range(2):image(x+j*.33,.29,s,z['panel_e'][j,seed])
 title(3.57,1.05,'(f) Field differences on common inputs')
 ax=axis(4.04,.40,2.97,.48);quantitative.append(ax)
 for pair,label,c,ls in FIELD_PAIRS:
  rs=[r for r in field if r['pair']==pair]
  ax.plot([float(r['tau']) for r in rs],[float(r['scaled_field_rmse']) for r in rs],
   color=c,ls=ls,marker='o',ms=2,lw=.9,label=label)
 ax.set(xlim=(0,1),ylim=(0,.175));ax.set_xticks([0,.2,.4,.6,.8,1]);ax.set_yticks([0,.05,.1,.15])
 ax.set_xlabel(r'Linear FM time $\tau$',labelpad=2);ax.set_ylabel('Scaled RMSE',labelpad=4)
 ax.grid(axis='y',color=LIGHT,lw=.5)
 # Put a compact legend above the curves; it does not obscure any values.
 ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.01),ncol=3,frameon=False,fontsize=5.2,
  columnspacing=.5,handlelength=1.2,handletextpad=.3,borderaxespad=0)
 args.output.mkdir(parents=True,exist_ok=True)
 fig.savefig(args.output/'mnist_appendix_composite.pdf',metadata={'Title':'MNIST empirical concentration and learned generation','Author':''})
 fig.savefig(args.output/'mnist_appendix_composite.png',dpi=240)
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 outside=[]
 for t in fig.findobj(matplotlib.text.Text):
  if t.get_visible() and t.get_text():
   b=t.get_window_extent(renderer)
   if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:outside.append(t.get_text())
 (args.output/'layout_validation.json').write_text(json.dumps({'all_text_inside_canvas':not outside,'outside':outside,
  'canvas_inches':[W,H],'plot_data_sha256':hashlib.sha256((source/'plot_data.npz').read_bytes()).hexdigest()},indent=2))
 assert not outside,outside
 plt.close(fig)
if __name__=='__main__':main()
