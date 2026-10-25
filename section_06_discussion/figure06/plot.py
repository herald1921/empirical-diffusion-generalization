"""Recovered Figure 6 renderer using four fixed, pre-existing outputs and probes.

The numerical/image inputs are original; the page layout is reconstructed.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R=Path(__file__).resolve().parent;D=R/'frozen_data';O=R/'output';O.mkdir(exist_ok=True)
z=np.load(D/'plot_inputs.npz');report=json.loads((D/'common_input_probes.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(7.2,4.6));grid=fig.add_gridspec(2,2,height_ratios=[1.2,1],left=.10,right=.98,top=.92,bottom=.12,hspace=.55,wspace=.35)
for col,n in enumerate([10000,98023]):
 ims=grid[0,col].subgridspec(2,4,wspace=.02,hspace=.05)
 for row,kind in enumerate(['mlp','unet']):
  assert z[f'{kind}_N{n}'].shape==(4,6400)
  for seed in range(4):
   ax=fig.add_subplot(ims[row,seed]);ax.imshow(z[f'{kind}_N{n}'][seed].reshape(80,80).clip(0,1),cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_visible(False)
   if seed==0:ax.set_ylabel(kind.upper() if kind=='mlp' else 'U-Net')
   if row==0:ax.set_title(f'Seed {seed}',fontsize=8)
 box=grid[0,col].get_position(fig);fig.text((box.x0+box.x1)/2,.98,f'(a) N = {n:,}',ha='center',va='top',weight='bold')
 ax=fig.add_subplot(grid[1,col]);rows=report['fields'][str(n)]['rows'];tau=np.array([x['tau'] for x in rows])
 for key,label,color,style in [('mlp_to_unet','MLP vs U-Net','#009E73','-'),('mlp_to_empirical','MLP vs empirical','#0072B2','--'),('unet_to_empirical','U-Net vs empirical','#D55E00',':')]:
  values=np.array([x[key] for x in rows]);assert np.isfinite(values).all()
  ax.plot(tau,values,style,color=color,marker='o',ms=3,label=label)
 ax.set(xlabel=r'Linear FM time $\tau$',ylabel='Scaled field RMSE',title=f'(b) Common inputs, N = {n:,}');ax.grid(alpha=.15)
fig.legend(handles=[Line2D([],[],color=c,ls=s,label=l) for c,s,l in [('#009E73','-','MLP vs U-Net'),('#0072B2','--','MLP vs empirical'),('#D55E00',':','U-Net vs empirical')]],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.55,-.005))
fig.savefig(O/'figure06.pdf');fig.savefig(O/'figure06.png',dpi=180);plt.close(fig)
print('Recovered Figure 6 layout; original images and common-input readout RMSE unchanged.')
