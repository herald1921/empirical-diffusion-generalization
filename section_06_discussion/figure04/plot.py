"""Figure 4 measurements in current linear FM coordinates; no inference."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent;D=R/'frozen_data';O=R/'output';O.mkdir(exist_ok=True)
z=np.load(D/'main_figure_inputs.npz')
m=json.loads((D/'main_original_plot_manifest.json').read_text())
records=[np.load(R/'validated_data'/f'empirical_N98023_{g}_baseline200k.npz') for g in ['s1','s2']]
tau=records[0]['tau'];assert np.array_equal(tau,records[1]['tau'])
k=m['main_common_index'];assert np.isclose(tau[k],.15)
take=m['figure1_snapshot_indices'];dense=np.asarray(m['figure1_curve_indices'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(7.2,4.6));gs=fig.add_gridspec(3,1,height_ratios=[1,1.2,1.2],left=.12,right=.98,bottom=.12,top=.94,hspace=.5)
im=gs[0].subgridspec(1,len(take),wspace=.08)
for j,idx in enumerate(take):
 ax=fig.add_subplot(im[j]);ax.imshow(z['figure6_readouts'][j].reshape(80,80).clip(0,1),cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(rf'$\tau={tau[idx]:.3f}$',fontsize=9)
 for s in ax.spines.values():s.set_visible(idx==k);s.set_color('#D55E00');s.set_linewidth(1.6)
ax=fig.add_subplot(gs[1]);ess=z['ess'];assert ess.shape[0]==len(tau)
for seed in range(1,8):ax.plot(tau[dense],ess[dense,seed],color='#BABFC4',lw=.7)
ax.plot(tau[dense],ess[dense,0],color='#0072B2',lw=1.5);ax.set_yscale('log');ax.set_ylim(.8,150000);ax.set_ylabel('(b) Posterior ESS');ax.axvline(.15,color='#D55E00',ls=':');ax.set_xlim(0,1)
ax=fig.add_subplot(gs[2])
for key,style,label in [('readout_train','-','Training'),('readout_reference','--','Reference')]:
 values=np.mean([x[key] for x in records],axis=(0,2))
 assert values.shape==tau.shape
 ax.plot(tau[dense],values[dense],style,color='#0072B2',label=label)
ax.axvline(.15,color='#D55E00',ls=':');ax.set(xlim=(0,1),ylabel='(c) Pixel RMSE',xlabel=r'Linear FM time $\tau$');ax.legend(frameon=False,ncol=2)
fig.savefig(O/'figure04.pdf');fig.savefig(O/'figure04.png',dpi=180);plt.close(fig)
print('Replotted stored Figure 4 measurements in linear FM time; no new model evaluation.')
