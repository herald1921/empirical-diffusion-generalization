from pathlib import Path
import csv,json
import numpy as np

root=Path(__file__).resolve().parent
assert (root/'all_complete.txt').exists()
coarse=np.load(root/'main/outputs_000.npz');fine=np.load(root/'refinement/outputs_000.npz')
assert set(coarse.files)==set(fine.files)
diffs={k:np.sqrt(np.mean((coarse[k][:8].astype(float)-fine[k].astype(float))**2,axis=1)) for k in fine.files}
rows={}
for tag,n in [('main',64),('refinement',8)]:
    manifest=json.loads((root/tag/'manifest.json').read_text())
    assert len(manifest['methods'])==92
    g=manifest['grid'];assert g[0]==0 and g[-1]==.99
    assert max(np.diff(g))<=1/manifest['nominal_steps_per_unit']+1e-10
    for m in manifest['methods']:
        if m['kind']=='pulse':
            assert abs(m['end']-m['start']-m['width'])<1e-10
            assert m['start'] in g and m['end'] in g
    rr=list(csv.DictReader((root/tag/'metrics.csv').open()))
    key=lambda r:(r['method'],int(r['seed']))
    assert len(rr)==92*n and len({key(r) for r in rr})==len(rr)
    for mid in [m['id'] for m in manifest['methods']]:assert sorted(int(r['seed']) for r in rr if r['method']==mid)==list(range(n))
    assert all(np.isfinite(float(v)) for r in rr for k,v in r.items() if k not in ['method','reused'])
    rows[tag]={key(r):r for r in rr}
lpips=list(csv.DictReader((root/'perceptual/metrics.csv').open()))
assert {key(r) for r in lpips}==set(rows['main'])
assert all(np.isfinite(float(r[k])) for r in lpips for k in ['lpips_train','lpips_from_emp','lpips_from_nn'])
largest=max(diffs,key=lambda k:diffs[k].max())
report=dict(maximum_state_rmse=float(diffs[largest].max()),maximum_state_case=largest,
    mean_state_rmse=float(np.concatenate(list(diffs.values())).mean()),
    maximum_training_distance_change=max(abs(float(r['train'])-float(rows['main'][k]['train'])) for k,r in rows['refinement'].items()),
    nearest_training_agreement=float(np.mean([r['train_index']==rows['main'][k]['train_index'] for k,r in rows['refinement'].items()])),
    main_rows=len(rows['main']),refinement_rows=len(rows['refinement']),perceptual_rows=len(lpips))
cross=[]
for first in [0,16,32,48]:
    a=np.load(root/f'main/outputs_{first:03d}.npz')
    b=np.load(root.parent/f'section64_sliding_20261018/main/outputs_{first:03d}.npz')
    for base in ['emp','nn']:
        k='full_'+base+'__0.99';cross.extend(np.sqrt(np.mean((a[k].astype(float)-b[k].astype(float))**2,1)))
report['maximum_baseline_change']=float(max(cross))
(root/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
