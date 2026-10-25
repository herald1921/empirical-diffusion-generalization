"""Recompute the deterministic sample-size scan with Euclidean, unregularized OT.

python compute_sample_scan.py --workers 2
The main-paper target and sample design are unchanged; only N varies.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys, csv, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import ndtr
from scipy.integrate import cumulative_trapezoid
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'numerics/reference'))
import figure02_core as core
DATA=ROOT/'data'

def calculate(task):
    n,eta,count=task
    sample=core.target_design(n)
    if eta==0:
        result=core.exact_ot(sample,core.target_design(count)); mass=1.
    else:
        angles=np.linspace(0,2*np.pi,131073)
        training_theta=core.target_quantile((np.arange(n)+.5)/n)
        density=np.zeros_like(angles)
        for start in range(0,len(angles),4096):
            delta=angles[start:start+4096,None]-training_theta
            a=core.R*np.cos(delta)/eta
            terms=np.exp(-core.R**2/(2*eta**2))/(2*np.pi)+a/np.sqrt(2*np.pi)*np.exp(-core.R**2*np.sin(delta)**2/(2*eta**2))*ndtr(a)
            density[start:start+4096]=np.maximum(terms.mean(axis=1),0.)
        cumulative=np.r_[0.,cumulative_trapezoid(density,angles)]
        mass=float(cumulative[-1])
        if abs(mass-1)>1e-8: raise ValueError(f'Angular mass {mass}')
        cumulative/=mass
        theta=np.interp((np.arange(count)+.5)/count,cumulative,angles)
        source=core.R*np.column_stack((np.cos(theta),np.sin(theta)))
        result=core.exact_ot(source,core.target_design(count))
    return dict(N=n,eta=eta,t=float(core.t_of_eta(eta)),nodes=count,wt=float(result[0]),duality_gap=float(result[1]),marginal_error=float(result[2]),dual_violation=float(result[3]),angular_mass=mass)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args(); path=DATA/'figure03_sample_scan.csv'
    rows=[]
    if path.exists():
        with path.open(newline='',encoding='utf-8') as f:
            rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    done={(int(r['N']),r['eta'],int(r['nodes'])) for r in rows}
    tasks=[(n,eta,c) for n in (8,16,32,64) for eta in (.35,.16,.008) for c in (2048,4096)]
    tasks += [(n,0.,c) for n in (8,16,32,64) for c in (4096,8192)]
    tasks=[t for t in tasks if t not in done]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(calculate,t) for t in tasks]):
            row=future.result(); rows.append(row)
            rows.sort(key=lambda r:(r['N'],r['eta'],r['nodes']))
            with path.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            print(json.dumps(row),flush=True)
    lookup={(int(r['N']),r['eta'],int(r['nodes'])):r for r in rows}
    diffs=[abs(lookup[n,e,4096]['wt']-lookup[n,e,2048]['wt']) for n in (8,16,32,64) for e in (.35,.16,.008)]
    audit=dict(training='deterministic probability midpoints; no iid repetitions',N=[8,16,32,64],eta=[.35,.16,.008],main_nodes=4096,coarse_nodes=2048,endpoint_nodes=8192,max_transport_refinement_change=max(diffs),max_endpoint_refinement_change=max(abs(lookup[n,0.,8192]['wt']-lookup[n,0.,4096]['wt']) for n in (8,16,32,64)),max_angular_mass_error=max(abs(r['angular_mass']-1) for r in rows),max_duality_gap=max(r['duality_gap'] for r in rows),max_marginal_error=max(r['marginal_error'] for r in rows),max_dual_violation=max(r['dual_violation'] for r in rows))
    (DATA/'sample_scan_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit),flush=True)
if __name__=='__main__':main()