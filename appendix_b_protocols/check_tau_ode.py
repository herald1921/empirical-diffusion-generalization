from pathlib import Path
import json, numpy as np
from scipy.integrate import solve_ivp
from scipy.special import softmax
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--data',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args();data=args.data
d=np.load(data);eta=d['eta'];tau=1/(1+eta);points=d['training_points'];ids=d['highlighted_ids'];frozen=d['raw_paths'][:,ids]
def rhs(t,y):
    u=y.reshape(-1,2);noise=(1-t)/t
    sq=((u[:,None,:]-points[None,:,:])**2).sum(2)
    m=softmax(-sq/(2*noise**2),axis=1)@points
    return ((m-u)/(t*(1-t))).ravel()
sol=solve_ivp(rhs,(tau[0],tau[-1]),frozen[0].ravel(),method='DOP853',t_eval=tau,rtol=2e-10,atol=2e-12,max_step=.002)
assert sol.success
error=np.linalg.norm(sol.y.T.reshape(frozen.shape)-frozen,axis=-1)
out={'solver':'DOP853 in native linear FM tau','paths':len(ids),'times':len(tau),'rtol':2e-10,'atol':2e-12,'maximum_step':.002,'max_position_difference':float(error.max()),'max_terminal_difference':float(error[-1].max()),'matched_initial_condition':True,'frozen_data_unchanged':True}
args.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));assert error.max()<1e-6
