"""Read-only paired integration and protocol audit; writes a separate report."""
from pathlib import Path
import csv,json
import numpy as np

root=Path('/root/shared-nvme')
report={}
for family in ['switch_scan','sliding','lognoise','tail']:
    d=root/f'section64_{family}_20261018'
    assert (d/'all_complete.txt').exists(),family
    coarse=np.load(d/'main/outputs_000.npz')
    fine=np.load(d/'refinement/outputs_000.npz')
    assert set(coarse.files)==set(fine.files)
    diffs={k:np.sqrt(np.mean((coarse[k][:8].astype(float)-fine[k].astype(float))**2,axis=1)) for k in fine.files}
    rows={}
    for tag,n in [('main',64),('refinement',8)]:
        rr=list(csv.DictReader((d/tag/'metrics.csv').open()))
        key=lambda r:(r['method'],int(r['seed']),float(r['tau']))
        assert len({key(r) for r in rr})==len(rr)
        groups={}
        for r in rr:groups.setdefault((r['method'],r['tau']),[]).append(int(r['seed']))
        assert all(sorted(s)==list(range(n)) for s in groups.values())
        numeric=[v for r in rr for k,v in r.items() if k not in ['method','reused']]
        assert all(np.isfinite(float(v)) for v in numeric)
        rows[tag]={key(r):r for r in rr}
    distance_delta=max(abs(float(r['train'])-float(rows['main'][k]['train'])) for k,r in rows['refinement'].items())
    same_ids=np.mean([int(r['train_index'])==int(rows['main'][k]['train_index']) for k,r in rows['refinement'].items()])
    largest_key=max(diffs,key=lambda k:diffs[k].max())
    entry=dict(main_rows=len(rows['main']),refinement_rows=len(rows['refinement']),
        maximum_output_rmse=float(diffs[largest_key].max()),maximum_output_case=largest_key,
        mean_output_rmse=float(np.concatenate(list(diffs.values())).mean()),
        maximum_training_distance_change=distance_delta,nearest_training_agreement=float(same_ids))
    if family!='tail':
        manifest=json.loads((d/'main/manifest.json').read_text())
        methods=manifest['methods']
        for m in methods:
            if m['kind']=='pulse':assert abs(m['end']-m['start']-m['width'])<1e-10
            if m['kind']=='lognoise':
                a,b=m['start'],m['end']
                assert abs(((1-a)/a)/((1-b)/b)-2)<1e-8
        grids={tag:json.loads((d/tag/'manifest.json').read_text())['grid'] for tag in ['main','refinement']}
        for tag,steps in [('main',200),('refinement',400)]:
            g=grids[tag]
            assert max(np.diff(g))<=1/steps+1e-10
            assert g[0]==0 and g[-1]==.99
            assert all(m['start'] in g and m['end'] in g for m in methods if m['kind']!='baseline')
        pp=json.loads((d/'perceptual/complete.json').read_text())
        pr=list(csv.DictReader((d/'perceptual/metrics.csv').open()))
        assert len(pr)==len(rows['main']) and pp['rows']==len(pr)
        assert {key(r) for r in pr}==set(rows['main'])
        assert all(np.isfinite(float(r[k])) for r in pr for k in ['lpips_train','lpips_from_emp','lpips_from_nn'])
        entry.update(methods=len(methods),perceptual=pp,
            reuse=json.loads((d/'main/complete.json').read_text())['reuse'])
        if family=='sliding':
            for r in rows['main'].values():
                m=next(m for m in methods if m['id']==r['method'])
                if m['kind']=='pulse' and m['end']==.99:
                    assert abs(float(r['window_exit_rmse'])-float(r['change_from_'+m['base']]))<1e-10
    report[family]=entry
cross=[]
for first in [0,16,32,48]:
    a=np.load(root/f'section64_sliding_20261018/main/outputs_{first:03d}.npz')
    b=np.load(root/f'section64_lognoise_20261018/main/outputs_{first:03d}.npz')
    for base in ['emp','nn']:
        key='full_'+base+'__0.99'
        cross.extend(np.sqrt(np.mean((a[key].astype(float)-b[key].astype(float))**2,1)))
report['cross_family_baselines']=dict(maximum_output_rmse=float(max(cross)),mean_output_rmse=float(np.mean(cross)))
out=root/'section64_switch_scan_20261018/numerical_audit.json'
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
