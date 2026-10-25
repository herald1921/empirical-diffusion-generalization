"""Six-inch paper panels; generated only when all six N values are available."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',required=True);R=Path(p.parse_args().root);O=R/'output'
plt.rcParams.update({'font.size':9.5})
ns=[1,10,100,1000,10000,98023]
for method,title in [('empirical','Empirical ODE'),('unet','Time-conditioned U-Net EMA'),('mlp','Time-conditioned MLP EMA')]:
    paths=[R/'data'/method/f'N{n}' for n in ns]
    if not all((p/'complete.json').exists() for p in paths):continue
    for seed in range(4):
        fig,axs=plt.subplots(4,6,figsize=(6,4.15))
        for col,(n,d) in enumerate(zip(ns,paths)):
            z=dict(np.load(d/'samples.npz'))
            for row,(key,label) in enumerate([('nearest_s1','Nearest in $S_1$'),('output_s1','Output from $S_1$'),('output_s2','Output from $S_2$'),('nearest_s2','Nearest in $S_2$')]):
                ax=axs[row,col];ax.imshow(z[key][seed].reshape(80,80),cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                if col==0:ax.set_ylabel(label,rotation=0,ha='right',va='center',labelpad=8,fontsize=9.5)
                if row==0:ax.set_title(f'{n:,}',fontsize=9.5)
        fig.suptitle(f'{title} | source seed {seed}',fontsize=10)
        fig.text(.60,.90,'Training images per group $N$',ha='center',fontsize=9.5)
        fig.subplots_adjust(left=.23,right=.995,bottom=.02,top=.82,hspace=.035,wspace=.05)
        for ext in ['png','svg']:fig.savefig(O/f'{method}_four_rows_paper_seed{seed}.{ext}',dpi=300)
        plt.close(fig)
print('PAPER_FOUR_ROWS_UPDATED',flush=True)
