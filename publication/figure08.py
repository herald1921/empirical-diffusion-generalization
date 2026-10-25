"""Figure 8: original text/geometry relationships with readable 9.2 pt labels.

Preserve the original vector geometry and frozen numerical curve records.
The four row names stay to the left, eta labels below the geometry, and each
curve panel shares a bottom normalized-noise label beside the common legend.
"""
from pathlib import Path
import argparse, importlib.util, json, hashlib
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,default=ROOT/'publication/output');a=ap.parse_args();O=a.output;O.mkdir(exist_ok=True,parents=True)
path=ROOT/'appendix_c_geometries/figure8_appendixC_20261019/generate_figure8.py'
spec=importlib.util.spec_from_file_location('geometry_render',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
records=json.loads((path.parent/'output/figure8_audit.json').read_text())['supports']
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9.2,'axes.labelsize':9.2,'xtick.labelsize':9.2,'ytick.labelsize':9.2,'legend.fontsize':9.2,'pdf.fonttype':42})
W,H=440,366;fig=plt.figure(figsize=(W/72,H/72));centers=[80,144,205,264];rowcenters=[84,158,232,306]
def txt(x,y,s,**kw):return fig.text(x/W,1-y/H,s,fontsize=9.2,va='center',**kw)
txt(W/2,10,r'Generation direction: normalized noise $\eta$ decreases',ha='center')
fig.add_artist(FancyArrowPatch((76/W,1-23/H),(310/W,1-23/H),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=8,lw=.7,color='#71869A'))
for x,lab in zip([70,143,204,268],['Support + observations','High noise','Intermediate noise','Low noise']):txt(x,39,lab,ha='center',weight='bold',linespacing=1.05)
txt(365,39,'Distributional accuracy',ha='center',weight='bold')
for i,(shape,cy) in enumerate(zip(['ellipse','sphere','segment','eight'],rowcenters)):
 rec=records[shape];name={'ellipse':'Ellipse','sphere':'Sphere','segment':'Segment','eight':'Figure eight'}[shape]
 txt(2,cy,name,ha='left',weight='bold')
 for j,eta in enumerate(rec['stages_eta']):txt(centers[j+1],cy+30,rf'$\eta={eta:g}$',ha='center')
 ax=fig.add_axes([322/W,1-(cy+25)/H,86/W,50/H]);topo=m.topology_record(shape)
 m.draw_distribution(ax,shape,rec['distribution'],rec['endpoint_SN'],topo,True)
 mid=min(rec['distribution'],key=lambda r:abs(r['eta']-rec['stages_eta'][1]))
 for key,c in [('w2',m.BLUE),('wt',m.GREEN),('wn',m.ORANGE)]:ax.plot(mid['eta'],max(mid[key]/rec['endpoint_SN'],1e-4),marker='o',ms=2.4,mfc='white',mec=c,mew=.65,zorder=5)
 ax.set_xlabel('');ax.set_ylabel(r'Distance / $S_N$',fontsize=9.2,labelpad=1)
 ax.set_yticks([1e-3,1e-1,10]);ax.tick_params(labelsize=9.2,pad=1)
 ax2=fig.axes[-1];ax2.set_yticks([0,8,16]);ax2.set_ylabel('Betti number',fontsize=9.2,labelpad=1);ax2.tick_params(labelsize=9.2,pad=1)
 for t in ax.texts:t.set_fontsize(9.2)
 if i<3:fig.add_artist(Line2D([.005,.995],[1-(cy+44)/H]*2,transform=fig.transFigure,color='#dce5eb',lw=.5))
# The common legend and normalized-noise label share the final baseline.
h=[Line2D([],[],color=m.BLUE,label=r'$W_2$'),Line2D([],[],color=m.GREEN,label=r'$W_T$'),Line2D([],[],color=m.ORANGE,label=r'$W_N$'),Line2D([],[],color=m.GRAY,ls='--',label=r'$S_N$'),Line2D([],[],color='#71869A',ls='--',label=r'$\beta_0$'),Line2D([],[],color='#71869A',ls=':',label=r'$\beta_k$')]
fig.legend(handles=h,ncol=6,loc='center',bbox_to_anchor=(120/W,1-354/H),frameon=False,handlelength=1.3,columnspacing=1.1,handletextpad=.35,borderpad=0)
txt(366,354,r'Normalized noise $\eta$ (decreasing $\longrightarrow$)',ha='center')
temp=O/'figure08_labels_curves.pdf';fig.savefig(temp,transparent=True);plt.close(fig)
original=fitz.open(a.input);g=original[0]
for b in g.get_text('dict')['blocks']:
 for l in b.get('lines',[]):g.add_redact_annot(l['bbox'],fill=None,cross_out=False)
g.apply_redactions(images=0,graphics=0,text=0)
# The row separators are redrawn at the new row boundaries.
for stroke in g.get_drawings():
 r=stroke['rect']
 if r.width>400 and r.height<.01 and r.y0>90:
  for item in stroke['items']:
   if item[0]=='l':g.draw_line(item[1],item[2],color=(1,1,1),width=stroke['width']+1)
doc=fitz.open(temp);page=doc[0]
for i,cy in enumerate(rowcenters):
 for j,sx in enumerate([86.53,168.54,255.70,339.17]):
  clip=fitz.Rect(sx-40,39+i*66.96,sx+40,93+i*66.96)
  # Each original geometry panel is scaled uniformly, then aligned with its header.
  rect=fitz.Rect(centers[j]-30,cy-20.25,centers[j]+30,cy+20.25)
  page.show_pdf_page(rect,original,0,clip=clip,keep_proportion=True,overlay=False)
out=O/'figure_08_synthetic_geometries.pdf';doc.save(out,deflate=True);page.get_pixmap(matrix=fitz.Matrix(2,2)).save(out.with_suffix('.png'))
(O/'figure08_layout_audit.json').write_text(json.dumps({'layout':'original four-row five-column layout; row names on left; noise labels below geometry; single-line column headers; larger geometry panels; reduced row spacing; shared legend beside the common bottom noise-axis label','geometry_source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'curve_records_sha256':hashlib.sha256((path.parent/'output/figure8_audit.json').read_bytes()).hexdigest(),'noise_stages':{k:v['stages_eta'] for k,v in records.items()},'new_simulation':False},indent=2))
