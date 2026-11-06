"""Figure 7: paired error curves; unchanged geometric snapshots and Betti data."""
from pathlib import Path
import argparse,json,csv,sys,hashlib
from compute_curves import module
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
try: import fitz
except ImportError:
    for p in Path(__file__).resolve().parents:
        if (p/'.tools/pdfdeps').is_dir():sys.path.insert(0,str(p/'.tools/pdfdeps'));break
    import fitz

ap=argparse.ArgumentParser();ap.add_argument('--repository',required=True,type=Path);ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'output');a=ap.parse_args()
m,_=module(a.repository);here=Path(__file__).resolve().parent;out=a.output;out.mkdir(parents=True,exist_ok=True)
data=json.loads((here/'curves.json').read_text())
old=a.repository/'appendix_c_geometries/synthetic_diversity_20261105'
template=old/'inputs/synthetic_geometry_template.pdf'
grows=list(csv.DictReader((old/'diversity_curves.csv').open()))
C=dict(raw='#0072B2',tw='#D55E00',tangent='#009E73',normal='#E69F00',diversity='#CC79A7',ink='#26343D',gray='#7E8B94',grid='#E7ECEF',shade='#EDF5F2',betti='#71869A')
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9.2,'axes.labelsize':9.2,'xtick.labelsize':9.2,'ytick.labelsize':9.2,'legend.fontsize':9.2,'pdf.fonttype':42,'text.color':C['ink'],'axes.labelcolor':C['ink'],'xtick.color':C['ink'],'ytick.color':C['ink']})
W,H=440,366;fig=plt.figure(figsize=(W/72,H/72));centers=[80,144,205,264];rowcenters=[84,150,216,282]
def txt(x,y,s,**kw):return fig.text(x/W,1-y/H,s,fontsize=9.2,va='center',**kw)
for x,lab in zip([70,143,204,268],['Support + observations','High noise','Intermediate noise','Low noise']):txt(x,39,lab,ha='center',weight='bold')
txt(365,39,'Wasserstein error',ha='center',weight='bold')
styles=[('raw_w2','raw','-',r'$W_2$ raw'),('raw_wt','tangent','-',r'$W_T$ raw'),('raw_wn','normal',(0,(5,2)),r'$W_N$ raw'),('denoised_w2','tw','-.',r'$W_2$ denoised'),('denoised_wn','normal','-',r'$W_N$ denoised')]
audit={}
for i,(shape,cy) in enumerate(zip(['ellipse','sphere','segment','eight'],rowcenters)):
    rec=data[shape];sn=rec['endpoint_SN'];rows=rec['distribution'];eta=np.array([r['eta'] for r in rows]);topo=m.topology_record(shape)
    txt(2,cy,{'ellipse':'Ellipse','sphere':'Sphere','segment':'Segment','eight':'Figure eight'}[shape],ha='left',weight='bold')
    for j,v in enumerate(rec['stages_eta']):txt(centers[j+1],cy+30,rf'$\eta={v:g}$',ha='center')
    ax=fig.add_axes([322/W,1-(cy+25)/H,86/W,50/H]);mid=min(rows,key=lambda r:abs(r['eta']-rec['stages_eta'][1]))
    for lo,hi in topo['spans_eta']:
        lo=max(lo,eta.min());hi=min(hi,eta.max())
        if hi>lo:ax.axvspan(lo,hi,color=C['shade'],lw=0,zorder=0)
    for field,col,ls,label in styles:
        values=np.array([r[field]/sn for r in rows]);values=np.where(values>1e-14,values,np.nan)
        ax.plot(eta,values,c=C[col],ls=ls,lw=1.15,zorder=3)
        if mid[field]/sn>=1e-3:ax.plot(mid['eta'],mid[field]/sn,'o',ms=2.4,mfc='white',mec=C[col],mew=.65,zorder=5)
    if shape!='eight':
        gr=sorted([r for r in grows if r['shape']==shape],key=lambda r:float(r['eta']))
        gx=np.array([float(r['eta']) for r in gr]);gy=np.array([float(r['G_over_SN']) for r in gr]);gy=np.where((gy>0)&np.isfinite(gy),gy,np.nan)
        ax.plot(gx,gy,c=C['diversity'],ls=(0,(1.5,1.8)),lw=1.25,zorder=4)
    ax.axhline(1,c=C['gray'],ls=(0,(4,3)),lw=.7,zorder=1)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(2,.008);ax.set_ylim(1e-3,100)
    ax.set_xticks([1,.1,.01],['1','0.1','0.01']);ax.set_yticks([1e-3,1e-1,10]);ax.minorticks_off();ax.grid(axis='y',c=C['grid'],lw=.45)
    ax.tick_params(labelsize=9.2,pad=1,length=2);ax.set_ylabel(r'Error / $S_N$',labelpad=1)
    ax.spines['top'].set_visible(False)
    if shape=='segment':ax.text(.03,.03,r'$W_N$ denoised $=0$',transform=ax.transAxes,fontsize=9.2,color=C['normal'],ha='left',va='bottom')
    beta=np.asarray(topo['betti_dense']);ax2=ax.twinx();k=2 if shape=='sphere' else 1
    ax2.step(topo['eta_dense'],beta[:,0],where='mid',c=C['betti'],ls=(0,(4,2)),lw=.65,alpha=.9)
    ax2.step(topo['eta_dense'],beta[:,k],where='mid',c=C['betti'],ls=(0,(1.2,1.6)),lw=.65,alpha=.9)
    ax2.set_ylim(-.3,max(3.2,beta[:,0].max()+.5));ax2.set_yticks([0,8,16]);ax2.tick_params(labelsize=9.2,pad=1,length=2,colors=C['betti']);ax2.set_ylabel('Betti number',fontsize=9.2,labelpad=1,color=C['betti']);ax2.spines['top'].set_visible(False)
    audit[shape]={'beta_sha256':hashlib.sha256(beta.tobytes()).hexdigest(),'topology_spans':topo['spans_eta'],'max_normalized_error':max(r[f]/sn for r in rows for f,*_ in styles),'error_curves':5,'diversity_curve':shape!='eight'}
    if i<3:fig.add_artist(Line2D([.005,.995],[1-(cy+40)/H]*2,transform=fig.transFigure,color='#dce5eb',lw=.5))
txt(365,325,r'Noise $\eta$ (decreasing $\rightarrow$)',ha='center')
handles=[Line2D([],[],c=C[col],ls=ls,lw=1.3,label=label) for _,col,ls,label in styles]
handles.append(Line2D([],[],c=C['diversity'],ls=(0,(1.5,1.8)),label=r'$\mathcal{G}(r)$'))
leg=fig.legend(handles=handles,ncol=6,loc='center',bbox_to_anchor=(.5,1-340/H),frameon=False,handlelength=1.4,columnspacing=.7,handletextpad=.35,borderpad=0)
aux=[Line2D([],[],c=C['gray'],ls=(0,(4,3)),label='1'),Line2D([],[],c=C['betti'],ls=(0,(4,2)),label=r'$\beta_0$'),Line2D([],[],c=C['betti'],ls=(0,(1.2,1.6)),label=r'$\beta_k$')]
fig.legend(handles=aux,ncol=3,loc='center',bbox_to_anchor=(.5,1-356/H),frameon=False,handlelength=1.4,columnspacing=1.5,handletextpad=.4,borderpad=0)
fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
for t in fig.findobj(Text):
    if t.get_visible() and t.get_text():
        bb=t.get_window_extent(renderer)
        if bb.width and bb.height and (bb.x0<0 or bb.x1>fig.bbox.width or bb.y0<0 or bb.y1>fig.bbox.height):outside.append(t.get_text())
assert not outside,outside
temp=out/'curves_labels.pdf';fig.savefig(temp,transparent=True);plt.close(fig)
original=fitz.open(template);doc=fitz.open(temp);page=doc[0]
for b in original[0].get_text('dict')['blocks']:
    for line in b.get('lines',[]):
        if ''.join(s['text'] for s in line['spans']).startswith('Distance / '):
            original[0].add_redact_annot(fitz.Rect(line['bbox']),fill=None,cross_out=False)
original[0].apply_redactions(images=0,graphics=0,text=0)
for cy,sourcecy in zip(rowcenters,[84,158,232,306]):
    for cx in centers:
        page.show_pdf_page(fitz.Rect(cx-30,cy-20.25,cx+30,cy+20.25),original,0,clip=fitz.Rect(cx-30,sourcecy-20.25-28,cx+30,sourcecy+20.25-28),keep_proportion=True,overlay=False)
page.set_cropbox(fitz.Rect(0,28,W,H))
dest=out/'figure_07_synthetic_geometries.pdf';doc.save(dest,deflate=True);page.get_pixmap(matrix=fitz.Matrix(2,2)).save(dest.with_suffix('.png'))
(out/'plot_audit.json').write_text(json.dumps(dict(rows=audit,colors=C,font='STIXGeneral',font_pt=9.2,outside_canvas_text=outside,geometry_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),G_sha256=hashlib.sha256((old/'diversity_curves.csv').read_bytes()).hexdigest()),indent=2))
print(dest)
