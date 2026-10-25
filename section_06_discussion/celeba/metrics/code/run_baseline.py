from common import *
from evaluate_gap import run_case
import time,subprocess,fcntl,traceback
setup();lock=(ROOT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);dump(ROOT/'supervisor.json',{'pid':os.getpid(),'state':'running','started':time.time()})
try:
 groups,ids=load_data()
 jobs=[(k,n,g) for n in [10000,98023,100] for k in ['empirical','unet','mlp'] for g in ['s1','s2']]
 while jobs:
  progressed=False
  for kind,n,g in list(jobs):
   if kind!='empirical' and not checkpoint(kind,n,g).exists():continue
   run_case(kind,n,g,groups,ids);jobs.remove((kind,n,g));progressed=True
   subprocess.run([sys.executable,str(ROOT/'code/plot_results.py'),'--root',str(ROOT)],check=True)
  if not progressed:
   status(state='waiting_for_existing_training',pending=jobs);time.sleep(30)
 status(state='baseline_gap_complete');dump(ROOT/'supervisor.json',{'pid':os.getpid(),'state':'baseline_complete','finished':time.time()})
except Exception:
 dump(ROOT/'supervisor.json',{'pid':os.getpid(),'state':'error','traceback':traceback.format_exc()});raise
