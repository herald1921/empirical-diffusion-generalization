"""Validate data-to-inset correspondence and paired posterior outputs."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent

def rows(name):
    with (ROOT/'data'/name).open(newline='',encoding='utf-8') as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]

def main():
    d=np.load(ROOT/'data/inset_data.npz');a=json.loads((ROOT/'data/inset_audit.json').read_text());r=rows('figure05_curves.csv');n=rows('figure04_display_normal.csv')
    minimum=min(r,key=lambda z:z['raw_wt']);assert abs(a['minimum']['eta']-minimum['eta'])<1e-12
    times=[]
    for key in ['noisy','minimum']:
        match=min(r,key=lambda z:abs(z['eta']-a[key]['eta']))
        assert abs(match['t']-a[key]['t'])<1e-12 and abs(match['raw_wt']-a[key]['wt'])<1e-12
        assert a[key]['readout_independent_max_difference']<1e-7
        assert np.max(abs(d['inset_'+key]))<4.2
        times.append(match['t'])
    assert times[0]<times[1]<1
    p=d['training_points'];diffs=[]
    for key,eta in [('a',.35),('b',.16)]:
        x=d['raw_'+key];scores=-np.sum((x[:,None,:]-p[None,:,:])**2,axis=2)/(2*eta**2);scores-=scores.max(axis=1,keepdims=True)
        weights=np.exp(scores);weights/=weights.sum(axis=1,keepdims=True)
        delta=float(abs(weights@p-d['tweedie_'+key]).max());assert delta<1e-12;diffs.append(delta)
        assert max(np.max(abs(x)),np.max(abs(d['tweedie_'+key])))<3.2
    normal=min(n,key=lambda z:abs(z['eta']-a['normal_inset']['eta']))
    assert abs(normal['eta']-.35)<1e-12 and abs(normal['t']-a['normal_inset']['t'])<1e-12
    basis=d['local_basis'];local=d['raw_a']@basis.T-np.array([0,2]);ids=np.flatnonzero((abs(local[:,0])<1.25)&(local[:,1]>-.7)&(local[:,1]<.55));assert np.array_equal(ids,d['normal_patch_ids'])
    for key in ('raw_a','tweedie_a'):
        x=d[key][ids]@basis.T-np.array([0,2]);assert max(abs(x[:,0]))<1.45 and min(x[:,1])>-.82 and max(x[:,1])<.65
    paired=[]
    for eta in [.35,.16]:
        row=min(r,key=lambda z:abs(z['eta']-eta));gain=1-row['tweedie_w2']/row['raw_w2'];paired.append(dict(eta=eta,t=row['t'],raw_w2=row['raw_w2'],tweedie_w2=row['tweedie_w2'],endpoint=row['endpoint'],gain=gain))
    assert paired[0]['gain']>0 and paired[1]['gain']<0
    assert all(v['tweedie_w2']<v['endpoint'] for v in paired)
    for name in ['figure05_curves.csv','figure04_normal.csv','numerical_audit.json']:
        assert (ROOT/'data'/name).read_bytes()==(ROOT/'numerics/data'/name).read_bytes()
    result=dict(passed=True,grid_minimum_verified=True,inset_times=times+[1.],max_posterior_coordinate_difference=max(diffs),normal_inset=normal,paired_total_error_readouts=paired,no_particle_clipping=True,curves_match_frozen_audited_inputs=True)
    (ROOT/'qa/data_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()