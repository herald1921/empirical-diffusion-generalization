from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();R=Path(a.root);O=R/'output';O.mkdir(exist_ok=True)
try:
 import fcntl
 plot_lock=(R/'plot.lock').open('w');fcntl.flock(plot_lock,fcntl.LOCK_EX)
except ImportError:pass
render_cases=sorted(str(p.parent.relative_to(R/'data')) for p in (R/'data').glob('*/*/complete.json'))
labels=['Nearest training\nimage in S1','Generated\nfrom S1','Generated\nfrom S2','Nearest training\nimage in S2']
keys=['nearest_s1','output_s1','output_s2','nearest_s2']
for method,title in [('empirical','Empirical flow'),('unet','Time-conditioned U-Net, EMA'),('mlp','Time-conditioned MLP, EMA')]:
 cases=[(n,R/'data'/method/f'N{n}'/'samples.npz') for n in [1,10,100,1000,10000,98023] if (R/'data'/method/f'N{n}'/'complete.json').exists()]
 if not cases:continue
 for seed in range(4):
  fig,axs=plt.subplots(4,len(cases),figsize=(2.3+1.4*len(cases),6),squeeze=False)
  for col,(n,path) in enumerate(cases):
   with np.load(path) as z:
    for row,key in enumerate(keys):
     ax=axs[row,col];ax.imshow(z[key][seed].reshape(80,80),cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
     for s in ax.spines.values():s.set_visible(False)
     if col==0:ax.set_ylabel(labels[row],fontsize=9,rotation=0,ha="right",va="center",labelpad=12)
     if row==0:ax.set_title(f'N = {n:,}',fontsize=10)
  fig.suptitle(f'{title} | source seed {seed}',fontsize=12)
  fig.subplots_adjust(left=.22,right=.99,bottom=.025,top=.91,wspace=.04,hspace=.07)
  fig.savefig(O/f'{method}_four_rows_seed{seed}.png',dpi=180);fig.savefig(O/f'{method}_four_rows_seed{seed}.svg');plt.close(fig)
# Audit preview: final legacy EMA and explicitly labelled short MLP pilots.
f=R/'qa/audit_arrays.npz'
if f.exists():
 with np.load(f) as z:
  rows=[('reused_unet_output','Reused U-Net EMA: 200k steps'),('mlp_pilot_N1','MLP pilot only: N=1'),('mlp_pilot_N10000','MLP pilot only: N=10,000')]
  fig,axs=plt.subplots(3,4,figsize=(8,6))
  for r,(key,label) in enumerate(rows):
   for c in range(4):
    ax=axs[r,c];ax.imshow(z[key][c].reshape(80,80),cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
    if c==0:ax.set_ylabel(label,fontsize=9)
    if r==0:ax.set_title(f'Source seed {c}',fontsize=10)
  fig.tight_layout();fig.savefig(O/'preflight_preview.png',dpi=130);plt.close(fig)


import subprocess,sys
subprocess.run([sys.executable,str(Path(__file__).with_name('plot_paper.py')),'--root',str(R)],check=True)
(O/'render_complete.json').write_text(json.dumps({'cases':render_cases,'finished':__import__('time').time()}))
