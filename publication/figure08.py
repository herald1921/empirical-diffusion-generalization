"""Preserve Figure 8's four-row/five-column arrangement and geometric drawings.

Increase row spacing for 9.2 pt labels. The original vector geometry panels
are retained; the right-column curves reuse the archived numerical records.
"""
from pathlib import Path
import argparse, importlib.util, json, sys, hashlib
import numpy as np
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1];O=ROOT/'publication/output';O.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);a=ap.parse_args()
path=ROOT/'appendix_c_geometries/figure8_appendixC_20261019/generate_figure8.py'
spec=importlib.util.spec_from_file_location('geometry_render',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
records=json.loads((path.parent/'output/figure8_audit.json').read_text())['supports']
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','font.size':9.2,'axes.labelsize':9.2,'xtick.labelsize':9.2,'ytick.labelsize':9.2,'legend.fontsize':9.2,'pdf.fonttype':42})
W,H=6,6.4;fig=plt.figure(figsize=(W,H));tops=[.86,.66,.46,.26]
centers=[.74,1.58,2.42,3.26]
for x,lab in zip(centers,['Support +\nobservations','High noise','Intermediate\nnoise','Low noise']):fig.text(x/W,.945,lab,ha='center',va='center',fontsize=9.2)
fig.text(.81,.945,'Distributional\naccuracy',ha='center',va='center',fontsize=9.2)
fig.text(.5,.99,r'Generation direction: normalized noise $\eta$ decreases $\longrightarrow$',ha='center',va='top',fontsize=9.2)
for i,(shape,top) in enumerate(zip(['ellipse','sphere','segment','eight'],tops)):
 rec=records[shape];name={'ellipse':'Ellipse','sphere':'Sphere','segment':'Segment','eight':'Figure-eight'}[shape]
 fig.text(centers[0]/W,top+.006,name,ha='center',va='bottom',weight='bold',fontsize=9.2)
 for j,eta in enumerate(rec['stages_eta']):fig.text(centers[j+1]/W,top-.123,rf'$\eta={eta:g}$',ha='center',va='top',fontsize=9.2)
 ax=fig.add_axes([.733,top-.145,.203,.145]);topo=m.topology_record(shape)
 m.draw_distribution(ax,shape,rec['distribution'],rec['endpoint_SN'],topo,True)
 mid=min(rec['distribution'],key=lambda r:abs(r['eta']-rec['stages_eta'][1]))
 for key,c in [('w2',m.BLUE),('wt',m.GREEN),('wn',m.ORANGE)]:ax.plot(mid['eta'],max(mid[key]/rec['endpoint_SN'],1e-4),marker='o',ms=2.4,mfc='white',mec=c,mew=.65,zorder=5)
 ax.set_xlabel(r'Noise $\eta$ (decreasing)',fontsize=9.2,labelpad=2);ax.set_ylabel(r'Distance / $S_N$',fontsize=9.2,labelpad=1)
 ax.set_yticks([1e-3,1e-1,10]);ax.tick_params(labelsize=9.2,pad=1)
 ax2=fig.axes[-1];ax2.set_yticks([0,8,16]);ax2.set_ylabel('Betti number',fontsize=9.2,labelpad=1);ax2.tick_params(labelsize=9.2,pad=1)
 # A single shared legend avoids repeating six tiny labels in every row.
 if shape=='segment':
  for t in ax.texts:t.set_fontsize(9.2)
handles=[Line2D([],[],color=c,label=lab) for c,lab in [(m.BLUE,r'$W_2$'),(m.GREEN,r'$W_T$'),(m.ORANGE,r'$W_N$')]]
handles += [Line2D([],[],color=m.GRAY,ls='--',label=r'$S_N$'),Line2D([],[],color='#71869A',ls='--',label=r'$\beta_0$'),Line2D([],[],color='#71869A',ls=':',label=r'$\beta_k$')]
fig.legend(handles=handles,ncol=6,loc='lower center',bbox_to_anchor=(.5,.006),frameon=False,handlelength=1.4,columnspacing=1.2)
temp=O/'figure08_labels_curves.pdf';fig.savefig(temp);plt.close(fig)
original=fitz.open(a.input);g=original[0]
for b in g.get_text('dict')['blocks']:
 for l in b.get('lines',[]):g.add_redact_annot(l['bbox'],fill=None,cross_out=False)
g.apply_redactions(images=0,graphics=0,text=0)
doc=fitz.open(temp);page=doc[0]
for i,top in enumerate(tops):
 # Crop only the unchanged geometry drawings, preserving their aspect ratio.
 clip=fitz.Rect(45,39+i*66.96,372,93+i*66.96)
 rect=fitz.Rect(.39*72,(1-top)*H*72,3.72*72,((1-top)*H+.62)*72)
 page.show_pdf_page(rect,original,0,clip=clip,keep_proportion=True,overlay=True)
out=O/'figure_08_synthetic_geometries.pdf';doc.save(out,deflate=True)
page.get_pixmap(matrix=fitz.Matrix(2,2)).save(out.with_suffix('.png'))
(O/'figure08_layout_audit.json').write_text(json.dumps({'layout':'same four geometry rows and five columns; increased row spacing','geometry_source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'curve_records_sha256':hashlib.sha256((path.parent/'output/figure8_audit.json').read_bytes()).hexdigest(),'noise_stages':{k:v['stages_eta'] for k,v in records.items()},'new_simulation':False},indent=2))
