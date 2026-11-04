"""Audit later-stop candidates against the original Figure 3 semantics."""
import csv,json
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
import compute_uniform as c
if __name__=='__main__':
    es=[.008,.016,.025,.04,.08,.11,.16]
    rows=c.run([(e,14,8192,c.SEED,'late_candidate_refined') for e in es],'late_b_candidates.csv',3)
    result=[];alpha=np.pi/c.N;cc=np.sqrt(2)+np.sqrt(2*np.log(20))
    for r in rows:
        e=r['eta'];xx=c.noisy_nodes(e,16,c.SEED)[:65536]
        w=softmax(-cdist(xx,c.P,'sqeuclidean')/(2*e*e),axis=1)
        gap=max(alpha-np.arcsin(cc*e/c.R),0.)
        G=np.sqrt(2*c.R*c.R/alpha*(gap-np.sin(gap)))
        out=dict(eta=e,eta_over_R=e/c.R,raw_over_SN=r['raw_w2']/c.SN,WT_over_SN=r['raw_wt']/c.SN,WN_over_SN=r['raw_wn']/c.SN,denoised_over_SN=r['tweedie_w2']/c.SN,G_over_SN=G/c.SN,r_over_R=cc*e/c.R,topology='circle' if cc*e>=c.R*np.sin(alpha) else '16 components',mean_ESS=float(np.mean(1/np.sum(w*w,axis=1))),fraction_max_weight_above_99percent=float(np.mean(w.max(1)>.99)),posterior_mean_displacement_from_nearest_site_RMS=float(np.sqrt(np.mean(np.sum((w@c.P-c.P[np.argmax(w,axis=1)])**2,axis=1)))),den_minus_raw_over_SN=(r['tweedie_w2']-r['raw_w2'])/c.SN)
        result.append(out)
    (c.ROOT/'late_b_selection_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
