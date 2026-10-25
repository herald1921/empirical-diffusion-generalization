from common import *
from evaluate_gap import run_case
import time,fcntl
setup()
lock=(ROOT/'gap_N1000.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
dump(ROOT/'protocol/gap_N1000_addendum.json',{'n':1000,'methods':['empirical','unet','mlp'],'groups':['s1','s2'],'samples_per_group':256,'timepoints':32,'network_steps':200000,'primary_output':'unclipped one-step readout','requested_maximum_N':100000,'actual_maximum_N':98023,'base_protocol_sha256':sha(ROOT/'protocol/followup_experiment.json')})
groups,ids=load_data()
for kind in ['empirical','unet','mlp']:
 for group in ['s1','s2']:
  run_case(kind,1000,group,groups,ids)
dump(ROOT/'qa/gap_N1000_complete.json',{'state':'complete','finished':time.time()})
print('ALL_N1000_COMPLETE',flush=True)
