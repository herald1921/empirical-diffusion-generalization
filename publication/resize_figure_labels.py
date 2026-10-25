"""Resize labels in vector PDF figures without changing plots or image samples.

Requires PyMuPDF. Inputs are the original manuscript figure PDFs, supplied
separately; no image data are distributed with this script.
"""
from pathlib import Path
import argparse, hashlib, json, re
import fitz
import matplotlib

def recolor(c):return ((c>>16&255)/255,(c>>8&255)/255,(c&255)/255)

def process(src,dst):
 old=fitz.open(src);scale=432/old[0].rect.width
 doc=fitz.open();p=doc.new_page(width=432,height=old[0].rect.height*scale)
 p.show_pdf_page(p.rect,old,0)
 lines=[l for b in p.get_text('dict')['blocks'] for l in b.get('lines',[])]
 fonts={}
 for f in p.get_fonts(full=True):
  data=doc.extract_font(f[0]);name=f[3].split('+')[-1]
  if 'TimesNewRoman' in name or 'STIXGeneral' in name:
   style='BolIta' if ('BoldItalic' in name or 'BoldIt' in name) else 'Bol' if 'Bold' in name else 'Italic' if 'Italic' in name else ''
   buf=(Path(matplotlib.get_data_path())/'fonts/ttf'/f'STIXGeneral{style}.ttf').read_bytes()
   fonts[name]=(f'LabelFont{len(fonts)}',buf)
  elif data[3] and name not in fonts:fonts[name]=(f'LabelFont{len(fonts)}',data[3])
 for l in lines:p.add_redact_annot(l['bbox'],fill=None,cross_out=False)
 p.apply_redactions(images=0,graphics=0,text=0)
 for name,(alias,buf) in fonts.items():p.insert_font(fontname=alias,fontbuffer=buf)
 records=[]
 for l in lines:
  spans=l['spans'];base=max(s['size'] for s in spans);factor=max(1,9.2/base)
  text=''.join(s['text'] for s in spans)
  number=int(src.name.split('_')[1])
  replacements={1:{'Sample neighborhood':'Neighborhood','Training observations':'Observations'},3:{'Empirical Tweedie':'Denoised'},6:{'Empirical / MLP':'Emp. / MLP','Empirical / U-Net':'Emp. / U-Net'},7:{'Nearest train':'Nearest','(a) Early replacement':'(a) Early','(b) Late replacement':'(b) Late','(c) Late replacement':'(c) Late','(d) Change in final output':'(d) Output change','(e) Proximity to training images':'(e) Training-image proximity'},10:{'main':'','final':''}}
  replacement=replacements.get(number,{}).get(text)
  if replacement is not None:
   if not replacement:continue
   s=dict(spans[0]);s['text']=replacement;spans=[s]
  box=fitz.Rect(l['bbox']);dx,dy=l['dir'];horizontal=abs(dx-1)<.01
  if horizontal:
   anchor=(box.x0+box.x1)/2;origin=spans[0]['origin'];newx=anchor+(origin[0]-anchor)*factor;newy=origin[1]
   if replacement is not None:
    alias,buf=fonts[spans[0]['font']];fontobj=fitz.Font(fontbuffer=buf) if buf else fitz.Font(alias)
    newx=anchor-fontobj.text_length(replacement,fontsize=base*factor)/2
   width=box.width*factor
   if newx<2:newx=2
   if newx+width>430 and replacement is None:newx=max(2,430-width)
  else:
   origin=spans[0]['origin'];newx,newy=origin
  for s in spans:
   font=s['font'];alias,buf=fonts[font]
   if horizontal:
    x=newx+(s['origin'][0]-origin[0])*factor;y=newy+(s['origin'][1]-origin[1])*factor
   else:x=newx+(s['origin'][0]-origin[0])*factor;y=newy+(s['origin'][1]-origin[1])*factor
   p.insert_text((x,y),s['text'].replace('\x00',''),fontname=alias,fontsize=s['size']*factor,color=recolor(s['color']),rotate=0 if horizontal else 90 if dy<0 else 270)
  records.append({'text':''.join(s['text'] for s in spans),'old_base_size':base,'new_base_size':base*factor})
 doc.save(dst,garbage=0,deflate=True)
 return {'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'placement_inches':6,'labels':records}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 report={}
 for p in sorted(a.input.glob('figure_*.pdf')):
  if p.name.startswith(('figure_08_','figure_09_')):continue
  report[p.name]=process(p,a.output/p.name);print(p.name,flush=True)
 (a.output/'label_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
