from pathlib import Path
import sys,importlib.util,json
r=Path(__file__).resolve().parent
sys.path[:0]=[str(r/'deps'),str(r.parent/'JMLR_integrated_20261015/tools_deps')]
import numpy as np, gudhi
s=importlib.util.spec_from_file_location('gallery',r.parent/'figure8_appendixC_20261019/generate_figure8.py')
g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
report={}
for shape in ['ellipse','sphere','segment','eight']:
    z=g.training(shape);n=len(z);record=g.topology_record(shape)
    st=gudhi.AlphaComplex(points=z.tolist(),precision='exact').create_simplex_tree()
    st.persistence(homology_coeff_field=11,min_persistence=1e-12,persistence_dim_max=True)
    dim=z.shape[1];intervals=[st.persistence_intervals_in_dimension(k) for k in range(dim)]
    def betti(rad):return tuple(sum(b<=rad**2<d for b,d in pairs) for pairs in intervals)
    critical=sorted(set([0.]+[float(np.sqrt(v)) for pairs in intervals for pair in pairs for v in pair if np.isfinite(v)]))
    probes=list(record['eta_dense']*record['c_delta'])+[(a+b)/2 for a,b in zip(critical[:-1],critical[1:]) if b-a>1e-8]
    if shape in ['ellipse','sphere']:
        key=f"{shape}_N16_a{0.0 if shape=='sphere' else 0.1:.1f}_design_r0"
        prior=lambda rad:g.metadata_betti(g.META[key],rad,dim)
    else:
        edges,tri=g.planar_alpha(shape,z)
        prior=lambda rad:g.betti_from_alpha(n,edges,tri,rad)
    mismatches=[dict(radius=float(rad),saved=list(prior(rad)),verified=list(betti(rad))) for rad in probes if prior(rad)!=betti(rad)]
    assert not mismatches,(shape,mismatches)
    report[shape]=dict(probes=len(probes),mismatches=mismatches,intervals_radius=[[[float(np.sqrt(b)),float(np.sqrt(d)) if np.isfinite(d) else None] for b,d in pairs] for pairs in intervals])
(r/'evidence/topology_check.json').write_bytes(json.dumps(dict(gudhi=gudhi.__version__,field=11,precision='exact',min_persistence_squared=1e-12,checks=report),indent=2).encode())
print('All four topology implementations agree with independent GUDHI checks.',flush=True)
