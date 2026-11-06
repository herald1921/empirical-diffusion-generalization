"""Paired raw/denoised diagnostics on the archived Figure 7 designs and nodes."""
from pathlib import Path
import argparse, importlib.util, json, hashlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax

def module(repo):
    path=repo/'appendix_c_geometries/figure8_appendixC_20261019/generate_figure8.py'
    spec=importlib.util.spec_from_file_location('archived_geometry',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m,path

def calculate(m,shape,count,etas,seed=2000):
    train=m.training(shape);truth=m.target(shape,count)
    z,g=m.noisy_nodes(train,count,seed)
    endpoint=m.w2_equal(z,truth);rows=[]
    for eta in etas:
        x=z+eta*g
        post=softmax(-cdist(x,train,'sqeuclidean')/(2*eta**2),axis=1)@train
        px=m.project(x,shape);pv=m.project(post,shape)
        row=dict(eta=float(eta),raw_w2=m.w2_equal(x,truth),raw_wt=m.w2_equal(px,truth),
            raw_wn=float(np.sqrt(np.mean(np.sum((x-px)**2,axis=1)))),
            denoised_w2=m.w2_equal(post,truth),denoised_wn=float(np.sqrt(np.mean(np.sum((post-pv)**2,axis=1)))))
        # The exact finite-node coupling obeys the same lower/upper bound.
        assert row['raw_wn']<=row['raw_w2']+1e-10
        assert row['raw_w2']<=row['raw_wt']+row['raw_wn']+1e-10
        assert row['denoised_wn']<=row['denoised_w2']+1e-10
        rows.append(row)
    return dict(endpoint_SN=endpoint,distribution=rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repository',type=Path,required=True);ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);m,path=module(a.repository)
    old=json.loads((path.parent/'output/figure8_audit.json').read_text())['supports']
    result={};checks={}
    for shape in ('ellipse','sphere','segment','eight'):
        result[shape]=calculate(m,shape,512,m.ETA)
        result[shape]['stages_eta']=old[shape]['stages_eta']
        diff=max(abs(new['denoised_'+k]-prior[k]) for new,prior in zip(result[shape]['distribution'],old[shape]['distribution']) for k in ('w2','wn'))
        assert diff<1e-12,(shape,diff)
        assert abs(result[shape]['endpoint_SN']-old[shape]['endpoint_SN'])<1e-12
        checks[shape]={'archived_denoised_max_abs_difference':diff,'refinement':[]}
        for count,seed in ((1024,2000),(512,5000)):
            refined=calculate(m,shape,count,[.06,.38,1.4],seed)
            delta={key:max(abs(row[key]/refined['endpoint_SN']-next(b for b in result[shape]['distribution'] if b['eta']==row['eta'])[key]/result[shape]['endpoint_SN']) for row in refined['distribution']) for key in ('raw_w2','raw_wt','raw_wn','denoised_w2','denoised_wn')}
            checks[shape]['refinement'].append(dict(count=count,seed=seed,max_abs_normalized_change=delta,values=refined))
        print(shape,'validated',flush=True)
    (a.output/'curves.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'numeric_audit.json').write_text(json.dumps(dict(count=512,seed=2000,eta_grid=m.ETA.tolist(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks=checks),indent=2)+'\n')
