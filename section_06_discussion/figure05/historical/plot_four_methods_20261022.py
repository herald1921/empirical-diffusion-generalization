from pathlib import Path
import hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
DESIGN=ROOT/'frozen_data'
z=np.load(DESIGN/'figure5_inputs.npz')
assert hashlib.sha256((DESIGN/'figure5_inputs.npz').read_bytes()).hexdigest()==json.loads((DESIGN/'provenance.json').read_text())['export_sha256']
OUT=ROOT/'output/figure5_matched_with_training.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
NS=z['n_values']; COLORS=['#D98228','#70777D','#399BBC','#184D75']
NC=['#777F84','#83ADC0','#6598B5','#4384AB','#236E9E','#163E6B']
LABELS=['Empirical stop','Empirical endpoint','MLP endpoint','U-Net endpoint']
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':8,
 'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
 'pdf.fonttype':42,'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,
 'text.color':'#233A49','axes.labelcolor':'#233A49','xtick.color':'#233A49','ytick.color':'#233A49'})

# A shared display row is used only after checking the rendered grayscale pixels.
rgba=plt.get_cmap('gray')
assert np.array_equal(rgba(z['images'][1].clip(0,1)),rgba(z['nearest_train'][1].clip(0,1)))
assert float(z['d_train'][1,...,0].max())<1e-6
fig=plt.figure(figsize=(7.2,5.5)); H=5.5
fig.text(.018,5.46/H,'(a) Outputs and nearest training images',fontsize=10,weight='bold',va='top')
fig.text(.99,5.46/H,'One shared initialization (seed 0)',fontsize=8.5,ha='right',va='top')
left=.18; gw=.815/6; iw=gw*.41; ih=iw*7.2/H
rows={1:[4.78],2:[4.32,3.81],3:[3.35,2.84],0:[2.38,1.87]}
for ni,n in enumerate(NS):
    fig.text(left+(ni+.5)*gw,5.20/H,f'$N={n:,}$',ha='center',fontsize=10,color=NC[ni])
    for gi,g in enumerate(['$S_1$','$S_2$']):
        x=left+ni*gw+(gw*.035 if gi==0 else gw*.545)
        fig.text(x+iw/2,5.04/H,g,ha='center',fontsize=7)
        for ci,ys in rows.items():
            for ri,y in enumerate(ys):
                ax=fig.add_axes([x,y/H-ih/2,iw,ih])
                v=z['images'][ci,ni,gi] if ri==0 else z['nearest_train'][ci,ni,gi]
                ax.imshow(v.reshape(80,80).clip(0,1),cmap='gray',vmin=0,vmax=1,interpolation='nearest'); ax.set_axis_off()
            if ci!=1:
                d=float(z['d_train'][ci,ni,gi,0])
                val=r'$<10^{-6}$' if d<1e-6 else f'{d:.4f}'
                fig.text(x+iw/2,(ys[0]-.255)/H,val,ha='center',va='center',fontsize=6)
for ci,name in [(1,'Empirical\nendpoint /\nnearest train'),(2,'MLP\nendpoint'),(3,'U-Net\nendpoint'),(0,'Empirical\nearly stop')]:
    ys=rows[ci]; y=sum(ys)/len(ys)
    if ci==1:
        fig.text(.168,y/H,name,ha='right',va='center',fontsize=8,color=COLORS[ci],linespacing=1.05)
    else:
        fig.text(.11,y/H,name,ha='right',va='center',fontsize=8,color=COLORS[ci],linespacing=1.1)
        for ri,yy in enumerate(ys):fig.text(.174,yy/H,['Output','Nearest'][ri],ha='right',va='center',fontsize=6.5)
for yy in [4.55,3.58,2.61]:
    fig.add_artist(Line2D([.02,.995],[yy/H,yy/H],transform=fig.transFigure,color='#D7DFE3',lw=.5))
handles=[Line2D([],[],color=COLORS[i],marker='o',markersize=2.6,lw=1.4,label=LABELS[i]) for i in [1,2,3,0]]
fig.legend(handles=handles,ncol=4,frameon=False,loc='center',bbox_to_anchor=(.53,1.56/H),
           columnspacing=1.2,handlelength=1.5,handletextpad=.4,fontsize=8)
axes=[fig.add_axes([x,.42/H,w,.85/H]) for x,w in [(.075,.25),(.415,.255),(.75,.235)]]
tau=z['tau']; ess=z['ess'].transpose(0,2,1,3).reshape(6,len(tau),16)
q=np.quantile(ess,[.1,.5,.9],axis=-1)
ax=axes[0]
for ni,n in enumerate(NS):
    ax.fill_between(tau,q[0,ni],q[2,ni],color=NC[ni],alpha=.12,lw=0)
    ax.plot(tau,q[1,ni],color=NC[ni],lw=1.1,label=f'{n:,}',ls='--' if n==1 else '-')
ax.axvline(.15,color=COLORS[0],ls=':',lw=1)
ax.set_yscale('log'); ax.set_ylim(.8,150000); ax.set_xlim(0,.5)
ax.set_xticks([0,.15,.3,.5],['0','.15','.3','.5']); ax.set_yticks([1,100,10000],['1',r'$10^2$',r'$10^4$']); ax.minorticks_off()
ax.set_ylabel('Posterior ESS'); ax.set_xlabel(r'Linear FM time $\tau$')
ax.set_title('(b) Empirical concentration',loc='left',pad=5,weight='bold')
ax.legend(ncol=2,frameon=False,loc='upper right',fontsize=5.8,handlelength=1.1,columnspacing=.55,labelspacing=.1,borderaxespad=.1)
train=z['d_train'].astype(float); ref=z['d_ref'].astype(float); gap=ref-train
mt=train.mean((2,3)); mr=ref.mean((2,3)); paired=gap.mean(-2); mg=paired.mean(-1)
ids=np.random.default_rng(22092027).integers(0,256,(2000,256))
ci=np.quantile(paired[...,ids].mean(-1),[.025,.975],axis=-1)
for j,c in enumerate(COLORS):
    axes[1].plot(NS,mt[j],color=c,marker='o',ms=2.5,lw=1.2)
    axes[1].plot(NS,mr[j],color=c,marker='o',mfc='white',ms=2.5,lw=1.1,ls='--')
    axes[2].fill_between(NS,ci[0,j],ci[1,j],color=c,alpha=.13,lw=0)
    axes[2].plot(NS,mg[j],color=c,marker='o',ms=2.5,lw=1.2)
for ax in axes[1:]:
    ax.set_xscale('log'); ax.set_xticks(NS,['1','10','100','1k','10k','98k']); ax.minorticks_off()
    ax.set_xlabel('Training/reference size $N$'); ax.set_xlim(.8,130000); ax.set_ylim(-.012,.34)
axes[1].set_ylabel('Pixel RMSE'); axes[1].set_title('(c) Distance at readout',loc='left',pad=5,weight='bold')
axes[1].legend(handles=[Line2D([],[],color='#344754',lw=1.1,label=r'$d_{\rm train}$'),Line2D([],[],color='#344754',lw=1.1,ls='--',label=r'$d_{\rm ref}$')],
               ncol=2,frameon=False,loc='upper right',fontsize=6.5,handlelength=1.5,columnspacing=.7)
axes[2].set_ylabel(r'Gap $d_{\rm ref}-d_{\rm train}$'); axes[2].set_title('(d) Training-set preference',loc='left',pad=5,weight='bold')
axes[2].axhline(0,color='#8997A0',lw=.6,ls=':')
for ax in axes: ax.grid(axis='y',color='#E1E8EC',lw=.5); ax.set_axisbelow(True); ax.tick_params(length=2.5,width=.6)
fig.savefig(OUT); fig.savefig(ROOT/'output/main_with_training.png',dpi=240)
(ROOT/'shared_row_validation.json').write_text(json.dumps({'empirical_endpoint_and_nearest_rendered_pixels_equal':True,'maximum_seed0_distance':float(z['d_train'][1,...,0].max()),'displayed_unique_images':84,'represented_output_reference_pairs':48},indent=2))
print('Rendered 48 output/reference pairs with 84 distinct image panels.')
