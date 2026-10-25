from pathlib import Path
import sys, importlib.util, json, hashlib, time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'JMLR_integrated_20261015/tools_deps'))
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.special import softmax
spec=importlib.util.spec_from_file_location('gallery',ROOT.parent/'figure8_appendixC_20261019/generate_figure8.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
out=ROOT/'evidence';out.mkdir(exist_ok=True)
old=json.loads((g.OUT/'figure8_audit.json').read_text())
rows=[]
for shape in ['ellipse','sphere','segment','eight']:
    train=g.training(shape)
    for count,scramble in [(512,0),(1024,0),(512,1),(2048,0)]:
        target=g.target(shape,count,777+scramble)
        z,noise=g.noisy_nodes(train,count,2000+100000*scramble)
        endpoint=g.w2_equal(np.repeat(train,count//len(train),axis=0),target)
        for eta in g.ETA:
            t=time.time();x=z+eta*noise
            post=softmax(-cdist(x,train,'sqeuclidean')/(2*eta**2),axis=1)@train
            projection=g.project(post,shape)
            w2=g.w2_equal(post,target);wt=g.w2_equal(projection,target)
            wn=float(np.sqrt(np.mean(np.sum((post-projection)**2,axis=1))))
            row=dict(shape=shape,count=count,scramble=scramble,eta=float(eta),endpoint=endpoint,w2=w2,wt=wt,wn=wn)
            if count==512 and scramble==0:
                saved=next(r for r in old['supports'][shape]['distribution'] if r['eta']==eta)
                assert max(abs(row[k]-saved[k]) for k in ['w2','wt','wn'])<1e-10
                assert abs(endpoint-old['supports'][shape]['endpoint_SN'])<1e-10
            rows.append(row)
        (out/'figure8_refinement_rows.json').write_bytes(json.dumps(rows,indent=2).encode())
        print(shape,count,scramble,'complete',flush=True)
summary={}
for shape in ['ellipse','sphere','segment','eight']:
    subset=[r for r in rows if r['shape']==shape]
    comparisons={}
    for count,scramble in [(1024,0),(512,1),(2048,0)]:
        changes=[]
        for r in [r for r in subset if r['count']==count and r['scramble']==scramble]:
            base=next(b for b in subset if b['count']==512 and b['scramble']==0 and b['eta']==r['eta'])
            changes.append(dict(eta=r['eta'],**{k:abs(r[k]-base[k]) for k in ['w2','wt','wn']},
                 ratio=abs(r['w2']/r['endpoint']-base['w2']/base['endpoint'])))
        comparisons[f'{count}_q{scramble}']={k:max(z[k] for z in changes) for k in ['w2','wt','wn','ratio']}
    minima=[]
    for count,scramble in [(512,0),(1024,0),(512,1),(2048,0)]:
        rs=[r for r in subset if r['count']==count and r['scramble']==scramble]
        m=min(rs,key=lambda r:r['w2']/r['endpoint'])
        minima.append(dict(count=count,scramble=scramble,eta=m['eta'],ratio=m['w2']/m['endpoint'],gain=m['endpoint']-m['w2']))
    summary[shape]=dict(changes_from_512=comparisons,minima=minima)
result=dict(numpy=np.__version__,scipy=scipy.__version__,source_sha256=hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest(),baseline_reproduced=True,summary=summary)
(out/'figure8_refinement_summary.json').write_bytes(json.dumps(result,indent=2).encode())
print(json.dumps(summary,indent=2),flush=True)
