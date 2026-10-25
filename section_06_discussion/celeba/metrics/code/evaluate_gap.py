from common import *
import time,argparse

def run_case(kind,n,g,groups,ids,tag='baseline200k',path=None,do_validation=True):
 d=ROOT/'data'/f'{kind}_N{n}_{g}_{tag}';d.mkdir(parents=True,exist_ok=True)
 if (d/'complete.json').exists():return
 own=active(groups,ids,g,n)[0].cuda();other=active(groups,ids,'s2' if g=='s1' else 's1',n)[0].cuda();val=(groups[0][10000:12048] if g=='s1' else groups[1][-12048:-10000]).cuda()
 if kind=='empirical':field=empirical_field(own);m=None;cp_sha='exact_empirical'
 else:
  p=Path(path) if path else checkpoint(kind,n,g);m=model(kind,p);field=m;cp_sha=sha(p)
 if n==10000:
  if not (d/'calibration.json').exists():dump(d/'calibration.json',calibration(own,other,val))
  if m is not None and do_validation and not (d/'validation.json').exists():
   status(state='independent_validation',method=kind,n=n,group=g,tag=tag);dump(d/'validation.json',validation_errors(m,own,val))
 source=np.load(BASE/'noise.npz')['noise'];taus=CFG['gap']['tau'];metrics={key:[] for key in ['raw_train','raw_reference','readout_train','readout_reference','clipped_train','clipped_reference','clip_fraction']};galleries={};refinement=None
 status(state='gap_sampling',method=kind,n=n,group=g,tag=tag)
 for b in range(0,len(source),32):
  noise=torch.from_numpy(source[b:b+32]).cuda();states,outs,grid=trajectory(field,noise,160)
  if b==0:
   if n==10000:
    fs,fo,_=trajectory(field,noise,320);refinement={'state_rmse':(states-fs).square().mean(2).sqrt().cpu().numpy(),'readout_rmse':(outs-fo).square().mean(2).sqrt().cpu().numpy()};del fs,fo
   galleries['state']=states[:,:8].cpu().numpy();galleries['readout']=outs[:,:8].cpu().numpy();galleries['nearest_train']=[];galleries['nearest_reference']=[]
   galleries['training_mean']=own.mean(0).cpu().numpy();galleries['real_training']=own[:8].cpu().numpy();galleries['real_validation']=val[:8].cpu().numpy()
  block={k:[] for k in metrics}
  for ti,t in enumerate(taus):
   for label,q in [('raw',states[ti]),('readout',outs[ti]),('clipped',outs[ti].clamp(0,1))]:
    a,ai=nearest_rmse(q,own);z,zi=nearest_rmse(q,other);block[label+'_train'].append(a.cpu().numpy());block[label+'_reference'].append(z.cpu().numpy())
    if b==0 and label=='readout':galleries['nearest_train'].append(own[ai[:8]].cpu().numpy());galleries['nearest_reference'].append(other[zi[:8]].cpu().numpy())
   block['clip_fraction'].append(((outs[ti]<0)|(outs[ti]>1)).float().mean(1).cpu().numpy())
  for k in metrics:metrics[k].append(np.stack(block[k]))
  del states,outs;print('GAP_BATCH',kind,n,g,tag,b+len(noise),flush=True)
 metrics={k:np.concatenate(v,axis=1) for k,v in metrics.items()};metrics.update(tau=np.array(taus),reverse_t=np.array(CFG['gap']['display_reverse_t']),seeds=np.arange(256))
 assert all(np.isfinite(v).all() for v in metrics.values())
 np.savez_compressed(d/'metrics.npz',**metrics);np.savez_compressed(d/'gallery.npz',**galleries)
 if refinement is not None:np.savez_compressed(d/'refinement.npz',**refinement);dump(d/'refinement.json',{'state_rmse_max':float(refinement['state_rmse'].max()),'readout_rmse_max':float(refinement['readout_rmse'].max()),'samples':32,'base_steps':[160,320],'requires_review':bool(refinement['readout_rmse'].max()>.01)})
 dump(d/'complete.json',{'method':kind,'n':n,'group':g,'tag':tag,'checkpoint_sha256':cp_sha,'protocol_sha256':sha(ROOT/'protocol/followup_experiment.json'),'noise_sha256':sha(BASE/'noise.npz'),'metrics_sha256':sha(d/'metrics.npz'),'samples':256,'timepoints':len(taus),'finished':time.time(),'human_study':False})
 del own,other,val,m,field;gc.collect();torch.cuda.empty_cache();print('CASE_COMPLETE',d.name,flush=True)

if __name__=='__main__':
 setup();p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--n',type=int,default=10000);p.add_argument('--group',required=True);p.add_argument('--tag',default='baseline200k');p.add_argument('--checkpoint');a=p.parse_args();groups,ids=load_data();run_case(a.kind,a.n,a.group,groups,ids,a.tag,a.checkpoint)
