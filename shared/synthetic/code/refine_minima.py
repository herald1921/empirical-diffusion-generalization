"""Refine exploratory grid minima after the prespecified extension scan.

The adaptive selection is recorded explicitly. No result is interpreted as
an independently validated stopping rule or as a uniform integration bound.
"""
import json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from run_synthetic import ROOT,DATA,key,solve,dump,configurations

def main():
    rows=json.loads((DATA/'all_rows.json').read_text())
    baseline=[r for r in rows if r['resolution']=='base' and r['scramble']==0]
    tasks=[];selections=[]
    for c in configurations():
        rr=[r for r in baseline if key(r)==key(c)]
        etas=sorted(set(min(rr,key=lambda r:r[m])['eta'] for m in ['raw_w2','tweedie_w2']))
        selections.append(dict(configuration=c,selected_eta=etas,rule='union of raw and Tweedie minimum on the 16-scale base grid'))
        for eta in etas:
            tasks.extend([(c,eta,'fine',0),(c,eta,'fine',1)])
    # A matched-N geometry figure also needs full representative precision checks.
    for c in configurations():
        if c['shape']=='sphere' and c['n']==16 and c['sampling']=='design':
            tasks.extend((c,e,'fine',s) for e in [.025,.06,.11,.20,.38,.72,1.4] for s in [0,1])
    # Deduplicate and retain the exact audit plan before executing it.
    tasks=list({(key(c),e,r,s):(c,e,r,s) for c,e,r,s in tasks}.values())
    dump(ROOT/'minima_refinement_plan.json',dict(selected_after_base_scan=True,selections=selections,tasks=[dict(configuration=c,eta=e,resolution=r,scramble=s) for c,e,r,s in tasks]))
    completed=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(solve,t) for t in tasks]):
            completed.append(future.result())
            dump(ROOT/'minima_refinement_status.json',dict(state='computing',completed=len(completed),total=len(tasks),pid=os.getpid()))
    dump(DATA/'minima_refinement.json',completed)
    dump(ROOT/'minima_refinement_status.json',dict(state='complete_pending_review',completed=len(completed),total=len(tasks),pid=os.getpid(),finished=time.time()))

if __name__=='__main__':main()
