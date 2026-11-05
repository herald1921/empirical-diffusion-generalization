"""Add the projected-neighborhood diversity deficit to the frozen synthetic figure."""
from pathlib import Path
import argparse, importlib.util, json, hashlib, csv, sys

ap=argparse.ArgumentParser()
ap.add_argument('--repository',type=Path,required=True)
ap.add_argument('--input',type=Path,required=True)
ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'output')
ap.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'diversity_curves.csv')
a=ap.parse_args();O=a.output;O.mkdir(exist_ok=True,parents=True)
try:
    import fitz
except ImportError:
    for parent in Path(__file__).resolve().parents:
        if (parent/'.tools/pdfdeps').is_dir():
            sys.path.insert(0,str(parent/'.tools/pdfdeps'));break
    import fitz
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

path=a.repository/'appendix_c_geometries/figure8_appendixC_20261019/generate_figure8.py'
spec=importlib.util.spec_from_file_location('geometry_render',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
records=json.loads((path.parent/'output/figure8_audit.json').read_text())['supports']
grows=list(csv.DictReader(a.data.open()))
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9.2,'axes.labelsize':9.2,'xtick.labelsize':9.2,'ytick.labelsize':9.2,'legend.fontsize':9.2,'pdf.fonttype':42})
W,H=440,366
fig=plt.figure(figsize=(W/72,H/72));centers=[80,144,205,264];rowcenters=[84,158,232,306]
def txt(x,y,s,**kw):return fig.text(x/W,1-y/H,s,fontsize=9.2,va='center',**kw)
for x,lab in zip([70,143,204,268],['Support + observations','High noise','Intermediate noise','Low noise']):txt(x,39,lab,ha='center',weight='bold')
txt(365,39,'Distributional accuracy',ha='center',weight='bold')
for i,(shape,cy) in enumerate(zip(['ellipse','sphere','segment','eight'],rowcenters)):
    rec=records[shape];name={'ellipse':'Ellipse','sphere':'Sphere','segment':'Segment','eight':'Figure eight'}[shape]
    txt(2,cy,name,ha='left',weight='bold')
    for j,eta in enumerate(rec['stages_eta']):txt(centers[j+1],cy+30,rf'$\eta={eta:g}$',ha='center')
    ax=fig.add_axes([322/W,1-(cy+25)/H,86/W,50/H])
    topo=m.topology_record(shape)
    m.draw_distribution(ax,shape,rec['distribution'],rec['endpoint_SN'],topo,True)
    ax2=fig.axes[-1]
    mid=min(rec['distribution'],key=lambda r:abs(r['eta']-rec['stages_eta'][1]))
    for key,col in [('w2',m.BLUE),('wt',m.GREEN),('wn',m.ORANGE)]:ax.plot(mid['eta'],max(mid[key]/rec['endpoint_SN'],1e-4),marker='o',ms=2.4,mfc='white',mec=col,mew=.65,zorder=5)
    if shape!='eight':
        rows=sorted([r for r in grows if r['shape']==shape],key=lambda r:float(r['eta']))
        eta=np.array([float(r['eta']) for r in rows])
        vals=np.array([float(r['G_over_SN']) for r in rows])
        # The logarithmic panel cannot display zero; NaN also suppresses out-of-domain radii.
        vals=np.where((vals>0)&np.isfinite(vals),vals,np.nan)
        ax.plot(eta,vals,color='#CC79A7',ls=(0,(1.5,1.8)),lw=1.25,zorder=6)
    ax.set_xlabel('');ax.set_ylabel(r'Distance / $S_N$',fontsize=9.2,labelpad=1)
    ax.set_yticks([1e-3,1e-1,10]);ax.tick_params(labelsize=9.2,pad=1)
    ax2.set_yticks([0,8,16]);ax2.set_ylabel('Betti number',fontsize=9.2,labelpad=1);ax2.tick_params(labelsize=9.2,pad=1)
    for t in ax.texts:t.set_fontsize(9.2)
    if i<3:fig.add_artist(Line2D([.005,.995],[1-(cy+44)/H]*2,transform=fig.transFigure,color='#dce5eb',lw=.5))
h=[Line2D([],[],color=m.BLUE,label=r'$W_2$'),Line2D([],[],color=m.GREEN,label=r'$W_T$'),Line2D([],[],color=m.ORANGE,label=r'$W_N$'),Line2D([],[],color='#CC79A7',ls=(0,(1.5,1.8)),label=r'$\mathcal{G}(r)$'),Line2D([],[],color=m.GRAY,ls='--',label=r'$S_N$'),Line2D([],[],color='#71869A',ls='--',label=r'$\beta_0$'),Line2D([],[],color='#71869A',ls=':',label=r'$\beta_k$')]
legend=fig.legend(handles=h,ncol=7,loc='center',bbox_to_anchor=(137/W,1-354/H),frameon=False,handlelength=1.2,columnspacing=.8,handletextpad=.32,borderpad=0)
noise_label=txt(366,354,r'Normalized noise $\eta$ (decreasing $\longrightarrow$)',ha='center')
fig.canvas.draw();renderer=fig.canvas.get_renderer()
assert legend.get_window_extent(renderer).x1 < noise_label.get_window_extent(renderer).x0
outside=[]
for t in fig.findobj(Text):
    if t.get_visible() and t.get_text():
        bb=t.get_window_extent(renderer)
        if bb.width and bb.height and (bb.x0<0 or bb.x1>fig.bbox.width or bb.y0<0 or bb.y1>fig.bbox.height):outside.append(t.get_text())
assert not outside,outside
temp=O/'synthetic_labels_curves.pdf';fig.savefig(temp,transparent=True);plt.close(fig)
original=fitz.open(a.input);doc=fitz.open(temp);page=doc[0]
assert abs(original[0].rect.width-440)<.01 and abs(original[0].rect.height-338)<.01
for cy in rowcenters:
    for cx in centers:
        clip=fitz.Rect(cx-30,cy-20.25-28,cx+30,cy+20.25-28)
        rect=fitz.Rect(cx-30,cy-20.25,cx+30,cy+20.25)
        page.show_pdf_page(rect,original,0,clip=clip,keep_proportion=True,overlay=False)
page.set_cropbox(fitz.Rect(0,28,W,H))
out=O/'figure_07_synthetic_geometries.pdf'
doc.save(out,deflate=True);page.get_pixmap(matrix=fitz.Matrix(2,2)).save(out.with_suffix('.png'))
(O/'plot_audit.json').write_text(json.dumps({'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'frozen_curves_sha256':hashlib.sha256((path.parent/'output/figure8_audit.json').read_bytes()).hexdigest(),'G_data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'G_color':'#CC79A7','G_omitted':['eight'],'ordinary_font_pt':9.2,'original_geometry_preserved':True,'original_transport_curves_preserved':True,'outside_canvas_text':outside,'shared_legend_rows':1},indent=2),encoding='utf8')
print(out)
