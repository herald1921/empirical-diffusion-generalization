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
 number=int(src.name.split('_')[1])
 if number==7:
  # Reserve a wider label column; uniformly scale each image row, preserving
  # face aspect ratios and every image sample. Header rules only move in x.
  original_graphics=doc;doc=fitz.open();p=doc.new_page(width=432,height=original_graphics[0].rect.height)
  p.show_pdf_page(p.rect,original_graphics,0)
  p.draw_rect(fitz.Rect(80,0,432,147),color=None,fill=(1,1,1))
  p.show_pdf_page(fitz.Rect(100,0,428.46,24),original_graphics,0,clip=fitz.Rect(86.76,0,428.46,24),keep_proportion=False)
  for top in [26.28,67.24,108.19]:
   p.show_pdf_page(fitz.Rect(100,top,428.46,top+38.03),original_graphics,0,clip=fitz.Rect(86.76,top,428.46,top+38.03),keep_proportion=True)
 if number==5:
  # Replace the two interior y ticks by their midpoint on both columns.
  groups={};frames=[]
  for g in p.get_drawings():
   r=g['rect'];items=g['items']
   if len(items)!=1 or items[0][0]!='l':continue
   if r.width<.01 and 30<r.height<40 and r.y0>175:frames.append(g)
   if abs(r.height)<.01 and abs(r.width-2.5)<.01 and any(abs(r.y0-y)<.01 for y in [202.77968,192.53397,247.812,238.74847]):
    _,a,b=items[0];key=(round(r.x0,2),r.y0>220);groups.setdefault(key,[]).append(g)
    p.draw_line(a,b,color=(1,1,1),width=g['width']+1)
  for gs in groups.values():
   assert len(gs)==2
   _,a,b=gs[0]['items'][0];y=sum(g['rect'].y0 for g in gs)/2
   p.draw_line((a.x,y),(b.x,y),color=gs[0]['color'],width=gs[0]['width'])
  for g in frames:
   _,a,b=g['items'][0];p.draw_line(a,b,color=g['color'],width=g['width'])
 fig1_axes=[];fig1_bars=[];fig1_links=[]
 if number==1:
  # Local clearance adjustments requested in the Figure 1 review. Erase
  # only the frame/annotation strokes being relocated, never data paths.
  for g in p.get_drawings():
   r=g['rect'];items=g['items']
   if len(items)!=1:continue
   item=items[0]
   if item[0]=='l' and r.x0>160 and abs(r.y0-209.6475)<.01:
    fig1_axes.append(g);p.draw_line(item[1],item[2],color=(1,1,1),width=g['width']+1)
   elif item[0]=='l' and abs(r.y0-232.518)<.01 and r.width>90:
    fig1_bars.append(g);p.draw_line(item[1],item[2],color=(1,1,1),width=g['width']+1)
   elif item[0]=='c' and abs(r.y0-236.3297)<.01 and abs(r.y1-260.4708)<.01:
    fig1_links.append(g);p.draw_bezier(*item[1:],color=(1,1,1),width=g['width']+1)
 # Add space only between existing rows; plot coordinates and image streams
 # retain their original scale. Text is placed after these translations.
 bands={1:[(35,4),(49,4),(64,5),(211,8)],3:[(167,2)],4:[(24,5)],5:[(22,4),(35,4),(165,4)],7:[(147,4),(161,2),(175,10)]}.get(number,[])
 def shift(y):return 6+sum(gap for edge,gap in bands if y>=edge)
 stripped=doc
 doc=fitz.open();p=doc.new_page(width=432,height=stripped[0].rect.height+14+sum(g for _,g in bands))
 boundaries=[0]+[edge for edge,_ in bands]+[stripped[0].rect.height]
 for y0,y1 in zip(boundaries,boundaries[1:]):
  off=shift((y0+y1)/2)
  p.show_pdf_page(fitz.Rect(0,y0+off,432,y1+off),stripped,0,clip=fitz.Rect(0,y0,432,y1),keep_proportion=False)
 if number==1:
  for g in fig1_axes:
   _,a,b=g['items'][0];off=shift(209.6475)+5
   p.draw_line((a.x,a.y+off),(b.x,b.y+off),color=g['color'],width=g['width'])
   if abs(a.y-b.y)<.01:
    p.draw_line((a.x,a.y+shift(a.y)),(a.x,a.y+off),color=g['color'],width=g['width'])
  for g in fig1_bars:
   _,a,b=g['items'][0];y=a.y+shift(a.y)+5;cx=(a.x+b.x)/2
   p.draw_line((a.x,y),(cx-5,y),color=g['color'],width=g['width'])
   p.draw_line((cx+5,y),(b.x,y),color=g['color'],width=g['width'])
  for g in fig1_links:
   item=g['items'][0];end=item[1];start=item[4]
   p.draw_line((start.x,start.y+shift(start.y)+9),(end.x,end.y+shift(end.y)),color=g['color'],width=g['width'])
 for name,(alias,buf) in fonts.items():p.insert_font(fontname=alias,fontbuffer=buf)
 records=[]
 for l in lines:
  spans=l['spans'];base=max(s['size'] for s in spans);factor=max(1,9.2/base)
  text=''.join(s['text'] for s in spans)
  by=(l['bbox'][1]+l['bbox'][3])/2
  if (number==1 and text in ['Sample neighborhood','Training observations','Target circle','Particle paths']) or (number==3 and by<16) or (number==4 and text in ['Seed 0','Other seeds']) or (number==6 and 150<by<170) or (number==7 and 147<by<162):continue
  number=int(src.name.split('_')[1])
  replacements={1:{'Sample neighborhood':'Neighborhood','Training observations':'Observations'},3:{'Empirical Tweedie':'Denoised'},6:{'Empirical / MLP':'Emp. / MLP','Empirical / U-Net':'Emp. / U-Net'},7:{'Nearest train':'Nearest','(a) Early replacement':'(a) Early','(b) Late replacement':'(b) Late','(c) Late replacement':'(c) Late','(d) Change in final output':'(d) Output change','(e) Proximity to training images':'(e) Training-image proximity','RMSE from original':'Output RMSE','Nearest-training LPIPS':'Nearest LPIPS'},10:{'main':'','final':''}}
  replacement=replacements.get(number,{}).get(text)
  if number==10:replacement=None
  if number==7:
   replacement='Nearest' if text=='Nearest train' else None
   if l['bbox'][0]<75 and 25<by<145:continue
  fig5_ytick=number==5 and l['dir']==(1.,0.) and l['bbox'][0]<42 and l['bbox'][1]>175 and text in ['0.0','0.1','0.2','0.3']
  if fig5_ytick:
   if text=='0.2':continue
   replacement={'0.0':'0','0.1':'0.15','0.3':'0.30'}[text]
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
   # Keep vertical labels centered rather than extending their top edge.
   anchor=(box.y0+box.y1)/2
   newy=anchor+(origin[1]-anchor)*factor
   if replacement is not None:
    alias,buf=fonts[spans[0]['font']];fo=fitz.Font(fontbuffer=buf)
    newy=anchor+fo.text_length(replacement,fontsize=base*factor)/2
  newy+=shift((box.y0+box.y1)/2)
  if number==10 and text in ['main','final']:newy+=2
  if number==1 and horizontal:
   if 'u1/R' in text.replace(' ','') or 'Linear FM' in text:newy+=3
   if text in ['Target circle','Particle paths']:newx+=5
   if text in ['Sample neighborhood','Training observations']:newx+=3
   if 202<box.y0<205 and box.x0>190:newy+=1.5
   if text in ['I','II','III'] and box.y0>230:newy+=5
   if text in ['−0.1','0.0','0.1'] and 160<box.y0<201 and (185<box.x0<198 or 321<box.x0<334):newx-=3.5
  if number==1 and not horizontal and '(u1' in text:newx-=7
  if number==3:
   if 'Noisier cloud' in text or 'Tangential minimum' in text or 'Empirical endpoint' in text:newy-=2
   if text.startswith('η') and box.y0>180:newy+=3
   if horizontal and box.x0<140 and 185<box.y0<280:newx=66
  if number==4 and text.startswith('τ'):newy+=2
  if number==5 and horizontal and 'Linear FM' in text:newy+=4
  if fig5_ytick:
   alias,buf=fonts[spans[0]['font']];newx=41-fitz.Font(fontbuffer=buf).text_length(replacement,fontsize=base*factor)
   if text=='0.1':newy-=5.12286 if box.y0<220 else 4.53176
  if number==5 and not horizontal and text in ['Pixel RMSE','Gap']:newx-=7
  if number==6 and text in ['MLP / U-Net','Empirical / MLP','Empirical / U-Net']:newx+=4
  if number==6:
   if horizontal and text.startswith('N=') and box.y0>170:newx+=5;newy+=3
   if horizontal and text in ['0.00','0.05','0.10','0.15','0.20'] and box.x0<42 and box.y0>170:newx-=3
   if not horizontal and text=='Scaled field RMSE':newx-=8
  if number==7:
   if text in ['0.60','0.6']:continue
   if text in ['Pure neural','Pure empirical']:newx+=5
   if horizontal and box.y0<25 and box.x0>80:
    cx=(box.x0+box.x1)/2;newx+=100+(cx-86.76)*(328.46/341.70)-cx
   if text.startswith(('(a)','(b)','(c)')):newx=2;newy-=3
   if '→' in text and box.y0<144:newx=2
   if text.startswith('['):newx=2;newy+=3
   if 'Replacement endpoint' in text:newy+=5
  for s in spans:
   font=s['font'];alias,buf=fonts[font]
   if horizontal:
    x=newx+(s['origin'][0]-origin[0])*factor;y=newy+(s['origin'][1]-origin[1])*factor
   else:x=newx+(s['origin'][0]-origin[0])*factor;y=newy+(s['origin'][1]-origin[1])*factor
   p.insert_text((x,y),s['text'].replace('\x00',''),fontname=alias,fontsize=s['size']*factor,color=recolor(s['color']),rotate=0 if horizontal else 90 if dy<0 else 270)
  records.append({'text':''.join(s['text'] for s in spans),'old_base_size':base,'new_base_size':base*factor})
 # Rebuild only legend keys, outside all data paths, with explicit key/text gaps.
 fontbuf=(Path(matplotlib.get_data_path())/'fonts/ttf/STIXGeneral.ttf').read_bytes()
 p.insert_font(fontname='Legend',fontbuffer=fontbuf)
 def label(x,y,s):p.insert_text((x,y),s,fontname='Legend',fontsize=9.2,color=(.12,.18,.22))
 def key(x,y,c,dash=None):p.draw_line((x,y-3),(x+14,y-3),color=c,width=1,dashes=dash)
 def clear(x0,y0,x1,y1):p.draw_rect(fitz.Rect(x0,y0+shift(y0),x1,y1+shift(y1)),color=None,fill=(1,1,1),overlay=True)
 blue=(0,.45,.7);gray=(.51,.58,.63);orange=(.83,.37,0)
 if number==1:
  clear(19,201,164,230)
  y1=214+shift(214)-4;y2=225+shift(225)-4
  p.draw_rect(fitz.Rect(16,y1-5,22,y1-1),color=(.55,.67,.74),fill=(.93,.96,.97),width=.4)
  label(28,y1,'Neighborhood');key(94,y1,(.25,.3,.33),'[2 2]');label(112,y1,'Target circle')
  p.draw_circle((19,y2-3),1.3,color=None,fill=(.12,.18,.22));label(28,y2,'Observations')
  key(94,y2,(.1,.4,.7));p.draw_circle((101,y2-3),1.4,color=None,fill=(.1,.4,.7));label(112,y2,'Particle paths')
 if number==3:
  clear(105,0,350,16);y=12+shift(12)
  key(110,y,blue);label(128,y,'Raw');key(160,y,orange);label(178,y,'Denoised');key(240,y,gray,'[3 2]');label(258,y,'Empirical benchmark')
  label(341,y,'S');p.insert_text((346,y+2),'N',fontname='Legend',fontsize=6.44,color=(.12,.18,.22))
 if number==4:
  clear(154,102,211,129)
  for y,c,s in [(110,blue,'Seed 0'),(123,gray,'Other seeds')]:
   yy=y+shift(y);key(156,yy,c);label(174,yy,s)
 if number==6:
  clear(125,153,350,171);y=166+shift(166)
  for x,c,d,s in [(105,(0,.52,.47),None,'MLP / U-Net'),(190,gray,'[3 2]','Emp. / MLP'),(280,orange,'[4 2 1 2]','Emp. / U-Net')]:key(x,y,c,d);label(x+19,y,s)
 if number==7:
  p.insert_font(fontname='RowBold',fontbuffer=(Path(matplotlib.get_data_path())/'fonts/ttf/STIXGeneralBol.ttf').read_bytes())
  for top,head,direction,interval in [(26.28,'(a) Early replacement','Empirical → neural','[0.10, 0.40)'),(67.24,'(b) Late replacement','Empirical → neural','[0.69, 0.99)'),(108.19,'(c) Late replacement','Neural → empirical','[0.69, 0.99)')]:
   base=top+shift(top)+7
   p.insert_text((2,base),head,fontname='RowBold',fontsize=9.2,color=(.12,.18,.22))
   label(2,base+12,direction);label(2,base+24,interval)
  clear(145,148,324,162);y=158+shift(158)
  for x,c,d,s in [(135,blue,None,'Empirical → neural'),(260,orange,'[3 2]','Neural → empirical')]:key(x,y,c,d);label(x+19,y,s)
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
