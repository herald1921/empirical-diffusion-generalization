"""Small, explicit entry points. No job trains a model implicitly."""
from pathlib import Path
import argparse,ast,json,subprocess,sys,os,hashlib
R=Path(__file__).resolve().parent
JOBS={
 'figure1':['section_03_topological_phases/figure01.py'],
 'figure2':['section_04_distribution/figure02/make_figure_02.py','plot'],
 'figure3':['section_05_early_stopping/figure03/make_figure.py'],
 'figure4':['section_06_discussion/figure04/plot.py'],
 'figure5':['section_06_discussion/figure05/plot.py'],
 'figure6':['section_06_discussion/figure06/plot.py'],
 'figure7':['section_06_discussion/figure07/plot.py'],
 'figure8':['appendix_c_geometries/figure8_appendixC_20261019/generate_figure8.py'],
 'figure9':['appendix_d_images/figure09/plot_revised.py'],
 'figure10':['appendix_d_images/figure10/make_figure.py'],
 'time':['appendix_b_protocols/check_tau_ode.py','--data',str(R/'section_03_topological_phases/frozen_data/figure_01_data.npz'),'--output',str(R/'validation/time.json')],
 'smoke':['shared/synthetic/code/run_synthetic.py','--smoke'],
 'topology':['appendix_c_geometries/reliability_revision_20261019/audit_topology.py'],
 'geometry':['appendix_c_geometries/reliability_revision_20261019/audit_current_figure.py'],
 'mnist-metrics':['appendix_d_images/figure09/test_numerics.py'],
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',choices=['list','check-source','verify-manifest','figures',*JOBS]);a=p.parse_args()
 if a.job=='list':
  for k,v in JOBS.items():print(k,':',' '.join(v))
  return
 if a.job=='check-source':
  files=list(R.rglob('*.py'))
  files=[x for x in files if not any(y in x.parts for y in ['.git','work','.venv'])]
  for f in files:ast.parse(f.read_text(encoding='utf-8-sig'),filename=str(f))
  print(f'{len(files)} Python sources parsed successfully.');return
 if a.job=='verify-manifest':
  m=json.loads((R/'configs/source_manifest.json').read_text())
  for x in m:
   f=R/x['path'];assert f.is_file(),x['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==x['sha256'],x['path']
  print(f'{len(m)} distributed source/input hashes verified.');return
 jobs=['figure2','figure3','figure4','figure5','figure6','figure7','figure8','figure9','figure10'] if a.job=='figures' else [a.job]
 (R/'validation').mkdir(exist_ok=True)
 env=dict(os.environ,MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
 for name in jobs:
  v=JOBS[name];print('Running',name,flush=True);subprocess.run([sys.executable,str(R/v[0]),*v[1:]],cwd=R,env=env,check=True)
if __name__=='__main__':main()
