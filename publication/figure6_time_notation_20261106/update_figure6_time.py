"""Versioned vector-label update: Replacement start a -> Replacement start tau_a.
Usage: python update_figure6_time.py INPUT.pdf OUTPUT.pdf
"""
from pathlib import Path
import argparse,sys,json,hashlib
try:import fitz
except ImportError:
    sys.path.insert(0,str(Path.cwd()/'.tools/pdfdeps'));import fitz
import numpy as np
import matplotlib
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
matplotlib.rcParams['mathtext.fontset']='stix'
fitz.TOOLS.set_small_glyph_heights(True)

def process(src,dst):
    d=fitz.open(src);p=d[0];old=p.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False);edits=[]
    for b in p.get_text('dict')['blocks']:
        for l in b.get('lines',[]):
            if ''.join(s['text'] for s in l['spans'])!='Replacement start a':continue
            r=fitz.Rect(l['bbox']);baseline=l['spans'][0]['origin'][1]
            # Intersect label glyphs through their middle, clear of tick-label boxes.
            p.add_redact_annot(fitz.Rect(r.x0,baseline-4,r.x1,baseline+.5),fill=None,cross_out=False);edits.append((l['spans'][0],r))
    assert len(edits)==2
    p.apply_redactions(images=0,graphics=0,text=0);regions=[]
    for s,r in edits:
        layout=MathTextParser('path').parse(r'Replacement start $\tau_a$',dpi=72,prop=FontProperties(family='STIXGeneral',size=s['size']))
        x=(r.x0+r.x1-layout.width)/2;y=s['origin'][1];base=layout.glyphs[0][4]
        for font,size,code,ox,oy in layout.glyphs:
            alias='Time'+hashlib.sha256(font.fname.encode()).hexdigest()[:10]
            p.insert_font(fontname=alias,fontfile=font.fname)
            p.insert_text((x+ox,y+base-oy),chr(code),fontname=alias,fontsize=size,color=(0,0,0))
        regions.append([min(x,r.x0)-2,r.y0-2,max(x+layout.width,r.x1)+2,max(r.y1,y+layout.depth)+2])
    dst.parent.mkdir(parents=True,exist_ok=True);d.save(dst,garbage=4,deflate=True)
    checked=fitz.open(dst);pix=checked[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
    a=np.frombuffer(old.samples,np.uint8).reshape(old.height,old.width,3);b=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
    change=np.any(a!=b,axis=2);mask=np.zeros(change.shape,bool)
    for x0,y0,x1,y1 in regions:mask[int(y0*2):int(y1*2)+1,int(x0*2):int(x1*2)+1]=True
    outside=int(np.sum(change&~mask))
    if outside:
        ys,xs=np.where(change&~mask);print('outside bbox',xs.min()/2,ys.min()/2,xs.max()/2,ys.max()/2,'regions',regions)
    assert outside==0,outside
    pix.save(str(dst.with_suffix('.png')))
    report={'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'edits':2,'outside_label_changed_pixels':outside,'font':'STIXGeneral','font_size':9.2,'data_and_geometry':'unchanged'}
    dst.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();process(a.input,a.output)
