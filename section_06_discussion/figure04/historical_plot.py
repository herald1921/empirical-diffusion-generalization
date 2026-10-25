"""JMLR-width layouts of the approved CelebA figures; data and examples unchanged.

--freeze extracts the exact approved readouts and nearest images once.
Subsequent builds read only the self-contained frozen data directory.
"""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[3];WORK=ROOT.parent;DATA=ROOT/'experiments/celeba/data'
OUT=ROOT/'paper/figures';PREVIEW=ROOT/'output/figures'
NS=[100,1000,10000,98023];METHODS=[('empirical','Empirical ODE'),('mlp','MLP'),('unet','U-Net')]
COLORS={100:'#A6CEE3',1000:'#5AA6CF',10000:'#2474B5',98023:'#173F7A'}

def aggregate(kind,n):
    z=[np.load(DATA/'metrics'/f'{kind}_N{n}_{g}_baseline200k.npz') for g in ['s1','s2']]
    tr=np.mean([a['readout_train'] for a in z],axis=0);ref=np.mean([a['readout_reference'] for a in z],axis=0);gap=ref-tr
    ids=np.random.default_rng(22092027).integers(0,256,(2000,256))
    return dict(train=tr.mean(1),reference=ref.mean(1),gap=gap.mean(1),ci=np.quantile(gap[:,ids].mean(2),[.025,.975],axis=1),t=z[0]['reverse_t'])

def freeze():
    report=json.loads((WORK/'figure_06_07_final/output_distances/plot_manifest.json').read_text(encoding='utf8'))
    k=report['main_common_index'];take=report['figure1_snapshot_indices'];arrays={}
    ess=np.load(WORK/'figure_06_07_final/data/ess.npz');arrays['ess']=ess['N98023_s1']
    paths=[]
    for kind,_ in METHODS:
        for n in NS:
            imgs=[]
            for g,key in [('s1','nearest_train'),('s1','readout'),('s2','readout'),('s2','nearest_train')]:
                path=WORK/'figure_08_gap/data'/f'{kind}_N{n}_{g}_baseline200k/gallery.npz'
                z=np.load(path);idx=k if kind=='empirical' else -1;imgs.append(z[key][idx,0])
                if kind=='empirical' and n==98023 and g=='s1' and key=='readout':arrays['figure6_readouts']=z[key][take,0]
                paths.append(path)
            arrays[f'{kind}_N{n}']=np.array(imgs)
    np.savez_compressed(DATA/'main_figure_inputs.npz',**arrays)
    (DATA/'main_original_plot_manifest.json').write_text(json.dumps(report,indent=2))
    manifest={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(paths)}
    manifest['figure_06_07_final/data/ess.npz']=hashlib.sha256((WORK/'figure_06_07_final/data/ess.npz').read_bytes()).hexdigest()
    (DATA/'main_figure_source_hashes.json').write_text(json.dumps(manifest,indent=2))

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':10.5,
        'xtick.labelsize':9.5,'ytick.labelsize':9.5,'legend.fontsize':9.5,'pdf.fonttype':42,'axes.linewidth':.7,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B2BBC2','text.color':'#253340'})

def face(ax,p):
    ax.imshow(p.reshape(80,80).clip(0,1),cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.03)
    fig.savefig(PREVIEW/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.03);plt.close(fig)

def plot():
    report=json.loads((DATA/'main_original_plot_manifest.json').read_text());z=np.load(DATA/'main_figure_inputs.npz')
    cases={(m,n):aggregate(m,n) for m,_ in METHODS for n in NS};t=cases['empirical',98023]['t']
    k=report['main_common_index'];stop=report['main_common_stop'];assert abs(t[k]-stop)<1e-14
    limit=report['shared_distance_ylim'][1];dense=np.array(report['figure1_curve_indices']);take=report['figure1_snapshot_indices']
    fig=plt.figure(figsize=(6.0,4.85));gs=fig.add_gridspec(3,1,height_ratios=[.88,1.15,1.15],hspace=.40,left=.14,right=.985,top=.94,bottom=.105)
    images=gs[0].subgridspec(1,len(take),wspace=.07)
    for j,idx in enumerate(take):
        ax=fig.add_subplot(images[0,j]);face(ax,z['figure6_readouts'][j]);ax.set_title(f'$t={t[idx]:.3f}$',fontsize=9.5,pad=5)
        if idx==k:
            for sp in ax.spines.values():sp.set_visible(True);sp.set_color('#8B5DA8');sp.set_linewidth(1.5)
    ax=fig.add_subplot(gs[1]);e=z['ess']
    for seed in range(1,8):ax.plot(t[dense],e[dense,seed],color='#BBC0C5',lw=.7,alpha=.6)
    ax.semilogy(t[dense],e[dense,0],color='#8B5DA8',lw=1.8,label='Displayed trajectory')
    ax.set_ylim(.8,150000);ax.set_yticks([1,100,10000],['1','100','10,000']);ax.minorticks_off();ax.set_ylabel('Posterior ESS')
    ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xticklabels([]);ax.axvline(stop,color='#69757E',lw=.8,ls=':');ax.grid(axis='y',color='#EDF0F2');ax.legend(frameon=False,loc='upper right')
    ax=fig.add_subplot(gs[2]);c=cases['empirical',98023]
    ax.plot(t[dense],c['train'][dense],color='#8B5DA8',lw=1.8,label=r'$d_{\rm train}$')
    ax.plot(t[dense],c['reference'][dense],color='#8B5DA8',lw=1.8,ls='--',label=r'$d_{\rm ref}$')
    ax.set_xlim(0,1);ax.set_ylim(0,limit);ax.set_yticks([0,.1,.2],['0','0.1','0.2']);ax.set_xticks([0,.25,.5,.75,1],['0','.25','.5','.75','1'])
    ax.set_xlabel('Reverse time $t$');ax.set_ylabel('Pixel RMSE');ax.axvline(stop,color='#69757E',lw=.8,ls=':');ax.grid(axis='y',color='#EDF0F2')
    ax.legend(frameon=False,ncol=2,loc='upper right');ax.text(.99,.55,'Shared early stop',ha='right',transform=ax.transAxes,fontsize=9.5)
    save(fig,'06_empirical_readout_ESS_distances')

    # Transpose the image grid: N is now in rows, so all labels fit at 9.5 pt.
    fig=plt.figure(figsize=(6.0,6.25));gs=fig.add_gridspec(3,3,height_ratios=[1.8,1.15,1.15],left=.13,right=.985,top=.86,bottom=.085,wspace=.18,hspace=.39)
    for j,(kind,title) in enumerate(METHODS):
        ig=gs[0,j].subgridspec(4,4,wspace=.025,hspace=.06)
        for row,n in enumerate(NS):
            for col,label in enumerate(['Train\n$S_1$','From\n$S_1$','From\n$S_2$','Train\n$S_2$']):
                ax=fig.add_subplot(ig[row,col]);face(ax,z[f'{kind}_N{n}'][col])
                if row==0:ax.set_title(label,fontsize=9.5,pad=5)
                if col==0 and j==0:ax.set_ylabel(f'{n:,}',rotation=0,ha='right',va='center',labelpad=6,fontsize=9.5)
        box=gs[0,j].get_position(fig);cx=(box.x0+box.x1)/2
        fig.text(cx,.985,title,ha='center',va='top',fontsize=10.5)
        fig.text(cx,.951,'Early stop' if kind=='empirical' else 'Terminal EMA',ha='center',va='top',fontsize=9.5)
        axd=fig.add_subplot(gs[1,j]);axg=fig.add_subplot(gs[2,j])
        for n in NS:
            c=cases[kind,n];col=COLORS[n];idx=k if kind=='empirical' else -1
            axd.plot(t,c['train'],color=col,lw=1.6,ls=(0,(5,2.2)))
            axd.plot(t,c['reference'],color=col,lw=1.35,ls=(0,(1.5,1.8)),alpha=.63)
            axd.plot(t[idx],c['train'][idx],'o',color=col,ms=2.7);axd.plot(t[idx],c['reference'][idx],'o',color=col,mfc='white',ms=2.7)
            axg.fill_between(t,c['ci'][0],c['ci'][1],color=col,alpha=.10,lw=0);axg.plot(t,c['gap'],color=col,lw=1.6)
            axg.plot(t[idx],c['gap'][idx],'o',color=col,ms=2.7)
        for ax in [axd,axg]:
            ax.set_xlim(0,1);ax.set_xticks([0,.5,1],['0','.5','1']);ax.set_yticks([0,.1,.2],['0','.1','.2']);ax.grid(axis='y',color='#EDF0F2',lw=.65)
            if kind=='empirical':ax.axvline(stop,color='#69757E',lw=.8,ls=':')
            if j:ax.tick_params(labelleft=False)
        axd.set_ylim(0,limit);axd.set_xticklabels([]);axg.set_ylim(*report['shared_gap_ylim']);axg.set_xlabel('Reverse time $t$');axg.axhline(0,color='#AAB3BB',lw=.6)
        if j==0:axd.set_ylabel('Pixel RMSE');axg.set_ylabel('Gap')
    legend_y=(gs[0,0].get_position(fig).y0+gs[1,0].get_position(fig).y1)/2
    handles=[Line2D([],[],color=COLORS[n],lw=1.6) for n in NS]
    fig.legend(handles,['100','1000','10,000','98,023'],ncol=4,loc='center',bbox_to_anchor=(.565,legend_y+.018),frameon=False,fontsize=9.5,handlelength=1.2,columnspacing=1.0,borderpad=0)
    fig.legend([Line2D([],[],color='#333',ls=(0,(5,2.2)),lw=1.6),Line2D([],[],color='#777',ls=(0,(1.5,1.8)),lw=1.35)],
               [r'$d_{\rm train}$',r'$d_{\rm ref}$'],ncol=2,loc='center',bbox_to_anchor=(.565,legend_y-.018),frameon=False,fontsize=9.5,borderpad=0)
    save(fig,'07_three_methods_four_rows_distances')
    audit=dict(same_selected_examples=True,same_noise_seed=0,same_stopping_index=k,same_stopping_t=stop,same_metrics=True,
       same_bootstrap_seed=22092027,same_bootstrap_resamples=2000,minimum_nominal_font_points=9.5,nominal_width_inches=6,
       changes='Publication layout only; Figure 7 image grid transposed so sample sizes are rows. Original workspace entries remain unchanged.')
    (ROOT/'audit/main_celeba_publication_layout.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    if args.freeze:freeze()
    style();plot()

if __name__=='__main__':main()
