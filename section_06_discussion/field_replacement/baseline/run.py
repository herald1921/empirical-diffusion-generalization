from pathlib import Path
import subprocess,sys
r=Path(__file__).resolve().parent
for cmd in [[sys.executable,'-u',str(r/'evaluate_scan.py')],[sys.executable,'-u',str(r/'evaluate_scan.py'),'--steps','400','--seeds','8','--tag','refinement'],[sys.executable,'-u',str(r/'evaluate_lpips.py')]]:
 subprocess.run(cmd,check=True,cwd=r)
(r/'all_complete.txt').write_text('All planned evaluations completed.\n')
