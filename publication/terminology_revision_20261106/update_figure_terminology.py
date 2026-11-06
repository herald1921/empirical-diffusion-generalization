"""Apply terminology-only edits to existing vector figures, retaining typography.

Usage: python update_figure_terminology.py INPUT_FIGURES OUTPUT_FIGURES
Requires PyMuPDF, matplotlib and numpy; original PDFs are supplied separately.
"""
from pathlib import Path
import argparse,hashlib,json,sys
try:
    import fitz
except ImportError:
    sys.path.insert(0,str(Path.cwd()/'.tools/pdfdeps'))
    import fitz
import matplotlib
import numpy as np
fitz.TOOLS.set_small_glyph_heights(True)

fontpath=Path(matplotlib.get_data_path())/'fonts/ttf/STIXGeneral.ttf'
font=fitz.Font(fontfile=str(fontpath))
rules={2:{'Tweedie':'Denoised'},6:{'Pure neural':'Pure learned','Empirical → neural':'Empirical → learned','Neural → empirical':'Learned → empirical'}}

def process(src,dst):
    number=int(src.name.split('_')[1]);doc=fitz.open(src);page=doc[0]
    original=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False);edits=[]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                if span['text'] not in rules[number]:continue
                assert span['font']=='STIXGeneral-Regular',span
                rect=fitz.Rect(span['bbox'])
                page.add_redact_annot(fitz.Rect(rect.x0+.04,rect.y0+.04,rect.x1-.04,rect.y1-.04),fill=None,cross_out=False)
                edits.append((span,rules[number][span['text']],rect))
    assert len(edits)=={2:2,6:6}[number],(src,len(edits))
    page.apply_redactions(images=0,graphics=0,text=0)
    page.insert_font(fontname='TerminologySTIX',fontfile=str(fontpath));regions=[]
    for span,label,rect in edits:
        x,y=span['origin'];width=font.text_length(label,fontsize=span['size'])
        if number==2:x=(rect.x0+rect.x1-width)/2
        c=span['color'];rgb=((c>>16&255)/255,(c>>8&255)/255,(c&255)/255)
        page.insert_text((x,y),label,fontname='TerminologySTIX',fontsize=span['size'],color=rgb)
        regions.append([min(rect.x0,x)-2,rect.y0-2,max(rect.x1,x+width)+2,rect.y1+2])
    dst.parent.mkdir(parents=True,exist_ok=True);doc.save(dst,garbage=4,deflate=True)
    checked=fitz.open(dst);updated=checked[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
    a=np.frombuffer(original.samples,np.uint8).reshape(original.height,original.width,3)
    b=np.frombuffer(updated.samples,np.uint8).reshape(updated.height,updated.width,3)
    changed=np.any(a!=b,axis=2);mask=np.zeros(changed.shape,bool)
    for x0,y0,x1,y1 in regions:mask[max(0,int(y0*2)):int(y1*2)+1,max(0,int(x0*2)):int(x1*2)+1]=True
    outside=int(np.sum(changed&~mask));assert outside==0,(src,outside)
    updated.save(str(dst.with_suffix('.png')))
    return {'file':src.name,'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'edits':[{'from':s['text'],'to':t,'font_size':s['size']} for s,t,r in edits],'outside_label_changed_pixels':outside,'page_size':list(checked[0].rect)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args();records=[]
    for n in rules:
        matches=list(args.input.glob(f'figure_{n:02d}_*.pdf'));assert len(matches)==1
        records.append(process(matches[0],args.output/matches[0].name))
    (args.output/'terminology_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(records,ensure_ascii=True,indent=2))
