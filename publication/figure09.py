"""Three-region Figure 9 preview, from unchanged frozen numerical inputs."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ap=argparse.ArgumentParser()
ap.add_argument('--data',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args(); D=args.data; O=args.output; O.mkdir(parents=True,exist_ok=True)
z=np.load(D/'plot_data.npz')
field=list(csv.DictReader((D/'field_discrepancies.csv').open()))
FS=9.2; W,H=432,259
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':FS,
    'axes.labelsize':FS,'xtick.labelsize':FS,'ytick.labelsize':FS,'legend.fontsize':FS,
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.55})
fig=plt.figure(figsize=(W/72,H/72))
NS=[100,1000,10000]; methods=['empirical','mlp','unet']
names=['Emp.','MLP','U-Net']; colors=['#0072B2','#7A8994','#008578']; nc=['#318AC3','#4F6CC1','#7352AD']
image_hashes=[]; curve_data=[]; texts=[]
def short(n):return {100:'100',1000:'1k',10000:'10k'}[n]
def txt(x,y,s,**kw):
    t=fig.text(x/W,1-y/H,s,fontsize=FS,va='center',**kw);texts.append(t);return t
def ax(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def im(x,y,s,a):
    a=np.asarray(a);image_hashes.append(hashlib.sha256(a.tobytes()).hexdigest())
    p=ax(x,y,s,s);p.imshow(a.reshape(28,28).clip(0,1),cmap='gray',vmin=0,vmax=1,interpolation='nearest');p.axis('off')
def frame(p,ylim):
    p.set(xlim=(0,1.025),ylim=ylim);p.set_xticks([0,.5,1],['0','.5','1'])
    p.grid(axis='y',color='#e4e9ed',lw=.45);p.tick_params(length=2,pad=2)
def legend(handles,x,y,**kw):
    kw.setdefault('handlelength',1.1)
    return fig.legend(handles=handles,loc='center',bbox_to_anchor=(x/W,1-y/H),frameon=False,handletextpad=.35,**kw)

# Region (a): original upper image matrix and distance/gap panels.
txt(3,10,'(a)',weight='bold')
s=16; starts=[29,100,171]; offsets=[0,16.8,35.8,52.6]
for j,kind in enumerate(methods):
    x=starts[j];center=x+34.3
    txt(center,25,names[j],ha='center')
    txt(center,38,r'$\tau=0.25$' if j==0 else r'$\tau=0.999$',ha='center',color='#D55E00' if j==0 else '#62717d')
    txt(x+16.4,52,r'$S_1$',ha='center');txt(x+52.2,52,r'$S_2$',ha='center')
    for k,label in enumerate(['N','O','O','N']):txt(x+offsets[k]+s/2,65,label,ha='center')
    for i,n in enumerate(NS):
        y=77+i*24
        if j==0:txt(25,y+s/2,short(n),ha='right')
        for k,off in enumerate(offsets):im(x+off,y,s,z['panel_c'][i,j,k])
txt(15,65,r'$N$',ha='center')
a_handles=[Line2D([],[],color=c,label=n,lw=1) for c,n in zip(colors,names)]
a_handles += [Line2D([],[],color='#657782',label='Train',lw=1),Line2D([],[],color='#657782',ls='--',label='Ref.',lw=1)]
legend(a_handles,331,10,ncol=5,columnspacing=.55,handlelength=1)
for j,n in enumerate(NS):
    x=272+j*55
    pd=ax(x,39,43,42);pg=ax(x,87,43,42)
    frame(pd,(0,.29));frame(pg,(-.012,.27));pd.tick_params(labelbottom=False)
    txt(x+21.5,27,rf'$N={short(n)}$',ha='center')
    for p in [pd,pg]:
        p.set_yticks([0,.1,.2],['0','.1','.2']);p.axvline(.25,color='#D55E00',ls=':',lw=.65);p.axvline(.999,color='#7A8994',ls=':',lw=.5)
    if j>0:pd.tick_params(labelleft=False);pg.tick_params(labelleft=False)
    pg.axhline(0,color='#98a7b1',lw=.45)
    for kind,c in zip(methods,colors):
        v=z[f'{kind}_N{n}_means'];ci=z[f'{kind}_N{n}_ci95']
        for k,ls in [(0,'-'),(1,'--')]:
            pd.fill_between(z['tau'],ci[k,0],ci[k,1],color=c,alpha=.09,lw=0);pd.plot(z['tau'],v[k],color=c,ls=ls,lw=.9)
        pg.fill_between(z['tau'],ci[2,0],ci[2,1],color=c,alpha=.12,lw=0);pg.plot(z['tau'],v[2],color=c,lw=.9)
    if j==0:pd.set_ylabel('Pixel RMSE',labelpad=2);pg.set_ylabel('Gap',labelpad=2)

# Fine separators identify the three regions without adding heavy frames.
fig.add_artist(Line2D([3/W,429/W],[1-148/H]*2,transform=fig.transFigure,color='#dce3e7',lw=.5))
fig.add_artist(Line2D([218/W]*2,[1-153/H,1-253/H],transform=fig.transFigure,color='#dce3e7',lw=.5))

# Region (b): original empirical snapshots and posterior ESS.
txt(3,159,'(b)',weight='bold')
s=17;xs=[29,48,67,86]
txt(15,174,r'$N$',ha='center')
for x,t in zip(xs,z['snapshot_tau']):
    txt(x+s/2,174,('.999' if t==.999 else f'{t:.2f}'.lstrip('0')),ha='center',color='#D55E00' if t==.25 else '#263742')
txt(65,161,r'$\tau$',ha='center')
for i,n in enumerate(NS):
    y=185+i*24;txt(25,y+s/2,short(n),ha='right')
    for j,x in enumerate(xs):im(x,y,s,z['panel_a'][i,j])
fig.add_artist(Rectangle(((xs[1]-1)/W,1-251/H),19/W,67/H,transform=fig.transFigure,fill=False,ec='#D55E00',lw=.7))
p=ax(132,162,76,76)
for n,c,ls in zip(NS,nc,[':','--','-']):
    lo,md,hi=z[f'ess_quantiles_N{n}'];p.fill_between(z['tau'],lo,hi,color=c,alpha=.14,lw=0);p.plot(z['tau'],md,color=c,ls=ls,lw=1,label=short(n))
frame(p,(.85,16000));p.set_yscale('log');p.set_yticks([1,100,10000]);p.minorticks_off();p.axvline(.25,color='#D55E00',ls=':',lw=.65)
p.set_ylabel('ESS',labelpad=1)
p.legend(loc='upper right',frameon=False,handlelength=1.0,handletextpad=.35,borderpad=.1,labelspacing=.1)

# Region (c): four shared-noise neural pairs and field differences.
txt(225,159,'(c)',weight='bold')
txt(273,174,'MLP / U-Net',ha='center')
for seed in range(4):
    x=232+(seed%2)*44;y=185+(seed//2)*46
    for j in range(2):im(x+j*20,y,19,z['panel_e'][j,seed])
p=ax(349,172,77,66)
for pair,label,c,ls in [('mlp_unet','MLP / U-Net','#008578','-'),('empirical_mlp','Emp. / MLP','#82929F','--'),('empirical_unet','Emp. / U-Net','#C17E47','-.')]:
    rows=[r for r in field if r['pair']==pair]
    p.plot([float(r['tau']) for r in rows],[float(r['scaled_field_rmse']) for r in rows],color=c,ls=ls,marker='o',ms=1.8,lw=.9,label=label)
frame(p,(0,.175));p.set_yticks([0,.05,.1,.15],['0','.05','.10','.15'])
p.set_ylabel('Scaled field RMSE',labelpad=2)
legend(p.get_legend_handles_labels()[0],335,159,ncol=3,columnspacing=.55,handlelength=.8)

for p in fig.axes:
    for line in p.lines:
        if len(line.get_xdata())>2:curve_data.append([np.asarray(line.get_xdata()).tolist(),np.asarray(line.get_ydata()).tolist()])
fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
for t in fig.findobj(matplotlib.text.Text):
    if t.get_visible() and t.get_text():
        b=t.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:outside.append(t.get_text())
audit={'image_count':len(image_hashes),'image_array_hashes':image_hashes,'curves':curve_data,
       'input_sha256':hashlib.sha256((D/'plot_data.npz').read_bytes()).hexdigest(),
       'field_csv_sha256':hashlib.sha256((D/'field_discrepancies.csv').read_bytes()).hexdigest(),
       'new_simulation':False,'ordinary_font_pt':FS,'font':'STIXGeneral','canvas_pt':[W,H],
       'regions':{'a':'original c,d','b':'original a,b','c':'original e,f'},'outside_text':outside,
       'abbreviations':{'Emp.':'Empirical','N':'nearest training image (roman image-column label)','O':'output','Ref.':'reference','k':'1000','Gap':'d_ref - d_train','Scaled field RMSE':'(1-tau) RMSE(v_A,v_B)'},
       'curve_horizontal_axis':'Linear FM time tau, defined in the caption'}
(O/'audit.json').write_text(json.dumps(audit,indent=2))
assert len(image_hashes)==56
assert not outside,outside
fig.savefig(O/'figure_09_mnist_comparison.pdf')
fig.savefig(O/'figure_09_mnist_comparison.png',dpi=240)
print(json.dumps({'outside':outside,'image_count':len(image_hashes),'curve_count':len(curve_data)}))
