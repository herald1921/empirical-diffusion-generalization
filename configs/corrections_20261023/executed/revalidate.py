from pathlib import Path
import sys,time,json,hashlib,traceback,gc,csv
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from nearest_exact import nearest_rmse
BASE=Path('/root/figure07_cfm_20260921')
GAP=Path('/root/figure08_gap_20260922')
DATA=Path('/root/shared-nvme/experiment-real-data/datasets/celeba80_author_order')
sys.path.insert(0,str(GAP/'code'))
import common
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def dump(p,obj):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def status(**kw):dump(ROOT/'status.json',dict(time=time.time(),**kw))
def load_groups():
 manifest=json.loads((DATA/'manifest.json').read_text())
 with np.load(DATA/'paper_partitions.npz') as p: ids=[p['group_a_raw_indices'],p['group_b_raw_indices']]
 groups=[torch.empty((len(i),6400),device='cuda') for i in ids]
 for s in manifest['shards']:
  with np.load(DATA/s['filename']) as z:raw=z['gray_rgb_sum'].reshape(s['count'],6400)
  for group,ix in zip(groups,ids):
   mask=(ix>=s['start'])&(ix<s['start']+s['count'])
   if mask.any():group[torch.from_numpy(np.flatnonzero(mask)).cuda()]=torch.from_numpy(raw[ix[mask]-s['start']].astype(np.float32)).cuda()/765
 return groups,ids
def main():
 started=time.time();status(state='loading_data');groups,ids=load_groups()
 source=np.load(BASE/'noise.npz')['noise'];taus=np.array(common.CFG['gap']['tau'])
 assert source.shape==(256,6400) and len(taus)==32
 report={'protocol':'All six N, two directions, three fields, all 256 original Gaussian seeds, all 32 original times; same weights and 160-interval Heun trajectory; exhaustive float64 direct pixel RMSE', 'cases':[], 'source_noise_sha256':sha(BASE/'noise.npz'),'baseline_commit':'9cd693773bfb88779d623f17da2b0d1760cda58d','training_performed':False,'nearest_source_sha256':sha(ROOT/'nearest_exact.py')}
 for n in [1,10,100,1000,10000,98023]:
  refs=[groups[0][:n],groups[1][-n:]]
  for kind in ['empirical','mlp','unet']:
   for gi,g in enumerate(['s1','s2']):
    tag=f'{kind}_N{n}_{g}_baseline200k';out=ROOT/'readouts'/tag;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():report['cases'].append(json.loads((out/'complete.json').read_text()));continue
    old=GAP/'data'/tag;oldmetric=np.load(old/'metrics.npz') if (old/'metrics.npz').exists() else None
    oldgallery=np.load(old/'gallery.npz') if (old/'gallery.npz').exists() else None
    own,other=refs[gi],refs[1-gi]
    if kind=='empirical':m=None;field=common.empirical_field(own);cp='exact_empirical'
    else:
     path=common.checkpoint(kind,n,g);m=common.model(kind,path);field=m;cp=sha(path)
    values={k:[] for k in ['readout_train','readout_reference','nearest_train_index','nearest_reference_index','legacy_train','legacy_reference']}
    gallery={};replay_max=0.;oldgallery_neighbor_changes=0
    with torch.inference_mode():
     for b in range(0,256,32):
      status(state='readout_evaluation',case=tag,first_seed=b,completed_cases=len(report['cases']),seconds=time.time()-started)
      states,ys,grid=common.trajectory(field,torch.from_numpy(source[b:b+32]).cuda(),160)
      if b==0:
       gallery['state']=states[:,:8].cpu().numpy();gallery['readout']=ys[:,:8].cpu().numpy()
       gallery['nearest_train']=[];gallery['nearest_reference']=[]
       if oldgallery is not None:replay_max=float(np.max(np.abs(gallery['readout']-oldgallery['readout'])))
      block={k:[] for k in values}
      for ti,t in enumerate(taus):
       for label,r in [('train',own),('reference',other)]:
        dd,ix=nearest_rmse(ys[ti],r)
        block['readout_'+label].append(dd.cpu().numpy());block['nearest_'+label+'_index'].append(ix.cpu().numpy())
        legacy,_=common.nearest_rmse(ys[ti],r)
        block['legacy_'+label].append(legacy.cpu().numpy())
        if b==0:
         nearest=r[ix[:8]].cpu().numpy();gallery['nearest_'+label].append(nearest)
         if oldgallery is not None:oldgallery_neighbor_changes+=int(np.any(nearest!=oldgallery['nearest_'+label][ti],axis=1).sum())
      for k in values:values[k].append(np.stack(block[k]))
      del states,ys
    arrays={k:np.concatenate(v,axis=1) for k,v in values.items()}
    arrays.update(tau=taus,seeds=np.arange(256))
    np.savez_compressed(out/'metrics.npz',**arrays);np.savez_compressed(out/'gallery.npz',**gallery)
    error={label:float(np.max(abs(arrays['readout_'+label]-arrays['legacy_'+label]))) for label in ['train','reference']}
    olderror={label:float(np.max(abs(arrays['readout_'+label]-oldmetric['readout_'+label]))) for label in ['train','reference']} if oldmetric is not None else None
    entry={'case':tag,'n':n,'method':kind,'group':g,'checkpoint_sha256':cp,'samples':256,'timepoints':32,'scalar_distances':16384,'max_distance_change_same_readout':error,'max_difference_from_archived_metrics':olderror,'gallery_readout_max_difference_from_archive':replay_max if oldgallery is not None else None,'gallery_nearest_image_changes_vs_archive':oldgallery_neighbor_changes if oldgallery is not None else None,'metrics_sha256':sha(out/'metrics.npz'),'gallery_sha256':sha(out/'gallery.npz'),'seconds':time.time()-started}
    dump(out/'complete.json',entry);report['cases'].append(entry);dump(ROOT/'readout_report.json',report)
    if oldmetric is not None:oldmetric.close();oldgallery.close()
    del m,field;gc.collect();torch.cuda.empty_cache()
 # Independently recompute all saved wider-window final-state distances, no ODE rerun.
 root=Path('/root/shared-nvme/section64_wider_20261018/main');rows=list(csv.DictReader((root/'metrics.csv').open()))
 outrows=[];fieldreport=[]
 for p in sorted(root.glob('outputs_*.npz')):
  start=int(p.stem.split('_')[1])
  with np.load(p) as z:
   for key in z.files:
    if not key.endswith('__0.99'):continue
    method=key[:-6];x=torch.from_numpy(z[key]).cuda()
    if x.ndim!=2 or x.shape[1]!=6400:continue
    status(state='field_replacement_distances',file=p.name,method=method)
    d,ix=nearest_rmse(x,groups[0]);dr,ir=nearest_rmse(x,groups[1])
    for j in range(len(x)):fieldreport.append({'method':method,'seed':start+j,'d_train':float(d[j]),'d_reference':float(dr[j]),'nearest_train_index':int(ix[j]),'nearest_reference_index':int(ir[j])})
 dump(ROOT/'field_replacement_exact.json',{'records':fieldreport,'source_metrics_sha256':sha(root/'metrics.csv'),'no_trajectory_changes':True})
 report['elapsed_seconds']=time.time()-started;report['total_scalar_distances']=sum(x['scalar_distances'] for x in report['cases']);dump(ROOT/'complete.json',report);status(state='complete',seconds=time.time()-started)
if __name__=='__main__':
 try:main()
 except Exception:
  status(state='failed',traceback=traceback.format_exc());raise
