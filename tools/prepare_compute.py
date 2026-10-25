"""Prepare relocated copies of the original Linux computation sources.

This never executes training or touches the original server experiment tree.
Data and checkpoints are explicit external inputs. The original source hashes
remain in configs/source_manifest.json; relocated copies have different hashes.
"""
from pathlib import Path
import argparse,ast,json,re
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--work-dir',type=Path,required=True)
 p.add_argument('--datasets',type=Path,required=True,help='Directory containing celeba80_author_order and/or mnist/MNIST/raw')
 a=p.parse_args();work=a.work_dir.expanduser().resolve();datasets=a.datasets.expanduser().resolve()
 if work.exists() and any(work.iterdir()):p.error('work-dir must be new or empty; existing experiments are never overwritten')
 work.mkdir(parents=True,exist_ok=True)
 rows=json.loads((R/'configs/source_manifest.json').read_text());copied=[]
 mappings=[('/root/shared-nvme/experiment-real-data/datasets',datasets.as_posix()),('/root/shared-nvme/experiment-mnist/celeba80_cloud_bundle',(work/'root/shared-nvme/experiment-real-data/projects/celeba80_cloud_bundle-main').as_posix()),('/root/',(work/'root').as_posix()+'/'),('/dev/shm/',(work/'temporary').as_posix()+'/')]
 for row in rows:
  origin=row.get('compute_destination',row.get('origin',''))
  if not origin.startswith('/root/') or Path(row['path']).suffix not in {'.py','.json','.yaml','.yml','.txt','.npz','.npy'}:continue
  source=R/row['path'];target=work/origin.lstrip('/')
  target.parent.mkdir(parents=True,exist_ok=True)
  if source.suffix in {'.npz','.npy'}:target.write_bytes(source.read_bytes())
  else:
   text=source.read_text(encoding='utf-8-sig')
   for old,new in mappings:text=text.replace(old,new)
   if source.suffix=='.py':ast.parse(text,filename=str(target))
   target.write_text(text,encoding='utf-8',newline='\n')
  copied.append(dict(repository=row['path'],prepared=str(target)))
 (work/'prepared_sources.json').write_text(json.dumps(copied,indent=2),encoding='utf-8')
 print(f'Prepared {len(copied)} source/configuration files in {work}; no computation started.')
 print('Place required trained checkpoints in the corresponding root/.../runs or checkpoints directories.')
 print('Historical source-hash assertions may deliberately reject relocated resume inputs; do not bypass them.')
if __name__=='__main__':main()
