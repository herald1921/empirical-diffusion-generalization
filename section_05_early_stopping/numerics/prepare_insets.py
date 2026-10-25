"""Derive exact-time inset readouts from the archived common trajectories."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.special import softmax
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parent
D=ROOT/'data'

def main():
    z=np.load(D/'figure_01_data.npz');v=np.load(D/'visual_data.npz')
    arrays={k:v[k] for k in v.files};points=v['training_points'];grid=z['eta'];paths=z['raw_paths']
    with (D/'figure05_curves.csv').open(newline='',encoding='utf-8') as f:rows=[{k:float(q) for k,q in r.items()} for r in csv.DictReader(f)]
    minimum=min(rows,key=lambda r:r['raw_wt'])
    # Pick an exact plotted readout near eta=.7, and the exact displayed grid minimum.
    noisy=min(rows,key=lambda r:abs(r['eta']-.7))
    def posterior(x,e):
        log=-np.sum((x[:,None,:]-points[None,:,:])**2,axis=2)/(2*e*e)
        return softmax(log,axis=1)@points
    def readout(eta,index):
        if abs(grid[index]-eta)<1e-13:return paths[index].copy()
        def rhs(e,x):
            p=x.reshape(-1,2);return ((p-posterior(p,e))/e).ravel()
        sol=solve_ivp(rhs,(float(grid[index]),eta),paths[index].ravel(),method='DOP853',rtol=2e-11,atol=2e-13)
        if not sol.success:raise RuntimeError(sol.message)
        return sol.y[:,-1].reshape(-1,2)
    audit={}
    for key,row in [('noisy',noisy),('minimum',minimum)]:
        eta=row['eta'];near=np.argsort(abs(grid-eta))[:2]
        first=readout(eta,int(near[0]));second=readout(eta,int(near[1]))
        delta=float(np.max(np.linalg.norm(first-second,axis=1)))
        assert delta<1e-7
        arrays['inset_'+key]=first
        audit[key]=dict(eta=eta,t=row['t'],wt=row['raw_wt'],readout_independent_max_difference=delta)
    local=v['raw_a']@v['local_basis'].T-np.array([0.,2.])
    ids=np.flatnonzero((abs(local[:,0])<1.25)&(local[:,1]>-.7)&(local[:,1]<.55))
    arrays['normal_patch_ids']=ids
    np.savez_compressed(D/'inset_data.npz',**arrays)
    audit['figure3_metric']='Tangential Euclidean W2; minimum is over the evaluated time grid, not total-W2 optimum.'
    audit['normal_inset']=dict(eta=.35,t=2/np.pi*np.arctan2(1,.35),particles=len(ids),selection='Same fixed geometric window on input, same IDs retained after Tweedie.')
    audit['figure5_metric']='Total Euclidean W2 before and after Tweedie, at the same stopping time, with empirical endpoint reference.'
    with (D/'figure04_normal.csv').open(newline='',encoding='utf-8') as f:
        normal=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    keys=['eta','t','raw_wn','tweedie_wn']
    normal=[{k:r[k] for k in keys} for r in normal]
    exact=min(rows,key=lambda r:abs(r['eta']-.35))
    assert abs(exact['eta']-.35)<1e-12
    if not any(abs(r['eta']-.35)<1e-12 for r in normal):normal.append({k:exact[k] for k in keys})
    normal.sort(key=lambda r:r['t'])
    with (D/'figure04_display_normal.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(normal)
    audit['normal_display_grid']='Original audited normal grid plus the audited eta=.35 polar readout from figure05_curves.csv; no nearest-time substitution.'
    (D/'inset_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()