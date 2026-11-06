"""Update approved vector-PDF notation only; preserve plots and raster images.

Usage: python update_figure_notation.py INPUT_FIGURES OUTPUT_FIGURES
Requires PyMuPDF, matplotlib and numpy. Original PDFs are supplied separately.
"""
from pathlib import Path
import argparse, hashlib, json, sys
try:
    import fitz
except ImportError:
    sys.path.insert(0,str(Path.cwd()/'.tools/pdfdeps'))
    import fitz
import matplotlib
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
import numpy as np

matplotlib.rcParams['mathtext.fontset']='stix'
parser=MathTextParser('path')
counts={2:1,4:4,7:1,8:6,9:1}

def process(src,dst):
    doc=fitz.open(src);page=doc[0];number=int(src.name.split('_')[1]);edits=[]
    original=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            spans=line['spans'];text=''.join(s['text'] for s in spans)
            if number in (2,7):
                if text!='SN':continue
                selected=spans;expression='$1$'
            else:
                indices=[i for i,s in enumerate(spans) if s['text']=='S']
                if not indices:continue
                assert len(indices)==1
                i=indices[0]
                if i+1>=len(spans) or spans[i+1]['text'] not in ('1','2'):continue
                selected=spans[i:i+2];expression=r'$\mathcal{S}_'+spans[i+1]['text']+'$'
            rect=fitz.Rect(selected[0]['bbox'])
            for s in selected[1:]:rect|=fitz.Rect(s['bbox'])
            # A small inset avoids deleting an adjacent prefix-space glyph.
            cut=fitz.Rect(rect.x0+.04,rect.y0+.04,rect.x1-.04,rect.y1-.04)
            page.add_redact_annot(cut,fill=None,cross_out=False)
            edits.append((selected[0],expression,rect))
    assert len(edits)==counts[number],(src,len(edits))
    page.apply_redactions(images=0,graphics=0,text=0)
    regions=[]
    for span,expression,rect in edits:
        layout=parser.parse(expression,dpi=72,prop=FontProperties(size=span['size']))
        first_y=layout.glyphs[0][4];x,y=span['origin'];color=span['color']
        rgb=((color>>16&255)/255,(color>>8&255)/255,(color&255)/255)
        for font,size,code,ox,oy in layout.glyphs:
            alias='Notation'+hashlib.sha256(font.fname.encode()).hexdigest()[:10]
            page.insert_font(fontname=alias,fontfile=font.fname)
            page.insert_text((x+ox,y+first_y-oy),chr(code),fontname=alias,fontsize=size,color=rgb)
        regions.append([rect.x0-2,rect.y0-2,max(rect.x1,x+layout.width)+2,rect.y1+2])
    dst.parent.mkdir(parents=True,exist_ok=True);doc.save(dst,garbage=4,deflate=True)
    verified=fitz.open(dst);page=verified[0]
    updated=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
    a=np.frombuffer(original.samples,np.uint8).reshape(original.height,original.width,3)
    b=np.frombuffer(updated.samples,np.uint8).reshape(updated.height,updated.width,3)
    changed=np.any(a!=b,axis=2);mask=np.zeros(changed.shape,bool)
    for x0,y0,x1,y1 in regions:mask[max(0,int(y0*2)):int(y1*2)+1,max(0,int(x0*2)):int(x1*2)+1]=True
    assert not np.any(changed&~mask),(src,'changes outside notation regions')
    updated.save(str(dst.with_suffix('.png')))
    return {'file':src.name,'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'edits':len(edits),'outside_region_changed_pixels':int(np.sum(changed&~mask)),'regions':regions,'page_size':list(page.rect)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    records=[]
    for n in counts:
        matches=list(args.input.glob(f'figure_{n:02d}_*.pdf'));assert len(matches)==1
        records.append(process(matches[0],args.output/matches[0].name))
    (args.output/'notation_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('file','edits','outside_region_changed_pixels')} for r in records],indent=2))
