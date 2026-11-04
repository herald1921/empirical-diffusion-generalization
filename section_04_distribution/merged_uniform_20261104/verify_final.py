"""Independent sensitivity and publication-artifact checks."""
import csv,json,re
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import ndtr
import compute_uniform as c

def load(name):
    return [{k:v if k=='level' else float(v) for k,v in r.items()} for r in csv.DictReader((c.ROOT/name).open())]

rows=load('curves_dense.csv');fine=load('fine_checks.csv')
main={round(r['eta'],9):r for r in rows}
sensitivity=[]
for row in fine:
    a=main[round(row['eta'],9)]
    sensitivity.append(dict(eta=row['eta'],power=int(row['power']),seed=int(row['seed']),raw_difference_over_SN=(row['raw_w2']-a['raw_w2'])/c.SN,tweedie_difference_over_SN=(row['tweedie_w2']-a['tweedie_w2'])/c.SN))

angular=[]
for eta in [.111,.35]:
    n=2048
    theta=np.linspace(-np.pi/c.N,np.pi/c.N,65537)
    delta=theta[:,None]-c.THETA
    a=c.R*np.cos(delta)/eta
    density=(np.exp(-c.R*c.R/(2*eta*eta))/(2*np.pi)+a/np.sqrt(2*np.pi)*np.exp(-c.R*c.R*np.sin(delta)**2/(2*eta*eta))*ndtr(a)).sum(axis=1)
    cdf=np.r_[0.,cumulative_trapezoid(density,theta)];cdf/=cdf[-1]
    u=(np.arange(n)+.5)/n
    source_angles=np.interp(u,cdf,theta);target_angles=(2*u-1)*np.pi/c.N
    xx=c.R*np.column_stack([np.cos(source_angles),np.sin(source_angles)])
    yy=c.R*np.column_stack([np.cos(target_angles),np.sin(target_angles)])
    val,check=c.symmetry_ot(xx,yy)
    quant=float(np.mean(np.sum((xx-yy)**2,axis=1)))
    coarse,_=c.angular_wt(eta,8192);refined,_=c.angular_wt(eta,16384)
    angular.append(dict(eta=eta,quotient_ot_squared=val,quantile_squared=quant,squared_difference=abs(val-quant),quantile_refinement_over_SN=abs(coarse-refined)/c.SN,solver=check))

pdf=(c.ROOT/'figure_02_stopping_denoising_uniform.pdf').read_bytes()
font_info=sorted(set(f.decode() for f in re.findall(rb'/BaseFont /([^\s/]+)',pdf)))
out=dict(source_quadrature_sensitivity=sensitivity,angular_validation=angular,curve_count=len(rows),maximum_polar_mass_error=max(abs(r['polar_mass']-1) for r in rows),maximum_angular_mass_error=max(abs(r['angular_mass']-1) for r in rows),maximum_polar_normal_refinement_over_SN=max(r['normal_tw_check'] for r in rows)/c.SN,pdf=dict(pages=len(re.findall(rb'/Type /Page\s',pdf)),raster_images=pdf.count(b'/Subtype /Image'),fonts=font_info,media_box=re.findall(rb'/MediaBox\s*\[([^\]]+)\]',pdf)[0].decode()))
(c.ROOT/'verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
