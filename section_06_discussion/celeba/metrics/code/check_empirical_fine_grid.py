from common import *
import time
setup()
with torch.inference_mode():
 groups,ids=load_data();ref=active(groups,ids,'s1',10000)[0].cuda();field=empirical_field(ref);noise=torch.from_numpy(np.load(BASE/'noise.npz')['noise'][:32]).cuda();outs={};states={}
 for steps in [160,320,640,1280]:
  s,o,_=trajectory(field,noise,steps);states[steps]=s.cpu();outs[steps]=o.cpu();print('REFINEMENT',steps,flush=True)
 report={'n':10000,'group':'s1','sources':list(range(32)),'tau':CFG['gap']['tau'],'pairs':[]};arrays={}
 for a,b in [(160,320),(320,640),(640,1280),(160,1280)]:
  diff=(outs[a]-outs[b]).square().mean(2).sqrt();ix=np.unravel_index(diff.numpy().argmax(),diff.shape);arrays[f'readout_{a}_{b}']=diff.numpy();report['pairs'].append({'steps':[a,b],'max_readout_rmse':float(diff.max()),'max_mean_readout_rmse_over_32_sources':float(diff.mean(1).max()),'worst_tau':CFG['gap']['tau'][ix[0]],'worst_source':int(ix[1]),'terminal_max':float(diff[-1].max())})
 dump(ROOT/'qa/empirical_s1_fine_grid_review.json',report);np.savez_compressed(ROOT/'qa/empirical_s1_fine_grid_review.npz',**arrays);print(json.dumps(report),flush=True)
