"""Track the finite, authorized CelebA queue without treating budgets as quality approval."""
from pathlib import Path
import json,time,subprocess,sys,fcntl,traceback,hashlib,os
B=Path('/root/figure07_cfm_20260921');R=Path('/root/figure08_gap_20260922');O=R/'output'
NS=[1,10,100,1000,10000,98023]

def read(p):
    try:return json.loads(p.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}

def dump(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)

def alive(pid):
    if not pid:return False
    p=Path('/proc')/str(pid)
    if not p.exists():return False
    try:return p.joinpath('stat').read_text().split(') ')[1].split()[0]!='Z'
    except (FileNotFoundError,IndexError):return False

def snapshot():
    expected_models=[B/'runs'/k/f'N{n}'/g/'status.json' for k in ['unet','mlp'] for n in NS for g in ['s1','s2']]
    missing_models=[str(p.parent.relative_to(B/'runs')) for p in expected_models if read(p).get('state') not in ['trained_pending_quality_review','reused_ema_verified']]
    expected_samples=[B/'data'/k/f'N{n}'/'complete.json' for k in ['empirical','unet','mlp'] for n in NS]
    missing_samples=[str(p.parent.relative_to(B/'data')) for p in expected_samples if not p.exists()]
    expected_gap=[R/'data'/f'{k}_N{n}_{g}_baseline200k'/'complete.json' for n in [10000,98023,100] for k in ['empirical','unet','mlp'] for g in ['s1','s2']]
    missing_gap=[p.parent.name for p in expected_gap if not p.exists()]
    extension={};missing_extension=[]
    for g in ['s1','s2']:
        c=read(R/'extension'/g/'complete.json');extension[g]=c
        if c.get('state')=='extension_not_triggered':continue
        if c.get('state')!='extension_budget_completed_pending_review' or c.get('step')!=300000:missing_extension.append(g+': total-step budget not complete')
        for step in [250000,300000]:
            if not (R/'data'/f'mlp_N10000_{g}_extension{step}'/'complete.json').exists():missing_extension.append(f'{g}: {step} evaluation')
    earlystop=(B/'data/empirical_earlystop/complete.json').exists()
    finished=not(missing_models or missing_samples or missing_gap or missing_extension) and earlystop
    return {'updated_unix':time.time(),'state':'compute_complete_pending_qualitative_review' if finished else 'running_or_waiting','human_study':False,'quality_review_complete':False,'mlp_max_total_steps':300000,'baseline_fixed_steps':200000,'models_complete':24-len(missing_models),'models_expected':24,'paired_sample_cases_complete':18-len(missing_samples),'paired_sample_cases_expected':18,'baseline_gap_cases_complete':18-len(missing_gap),'baseline_gap_cases_expected':18,'missing_models':missing_models,'missing_samples':missing_samples,'missing_gap':missing_gap,'missing_extension':missing_extension,'extension':extension,'current_training':read(B/'status.json'),'current_gap':read(R/'status.json'),'current_extension':read(R/'extension_status.json'),'earlystop_complete':earlystop,'legacy_run_status':read(Path('/root/figure06_comparison_20260921/status.json'))}

def write_report(s):
    dump(O/'celeba_completion.json',s)
    text=['# CelebA 实验完成检查','',f"状态：{s['state']}。此状态不等于视觉质量验收。",'','| 项目 | 已完成 / 计划 |','|---|---|',f"| U-Net/MLP 模型（含复用） | {s['models_complete']} / 24 |",f"| 三方法 × 六个 N 的配对四行图数据 | {s['paired_sample_cases_complete']} / 18 |",f"| 三方法 × 三个 N × 双方向 gap | {s['baseline_gap_cases_complete']} / 18 |",f"| 当前协议经验 ODE 早停读出及重新检索 | {'完成' if s['earlystop_complete'] else '未完成'} |",'','MLP 总步数上限 300000；基线固定预算 200000，N10000按独立验证触发追加至250000/300000并分别评估。不以参照集 gap 挑选模型。','','## 当前运行','','```json',json.dumps({k:s[k] for k in ['current_training','current_gap','current_extension']},indent=2,ensure_ascii=False),'```','','## 待完成配置','']
    for k in ['missing_models','missing_samples','missing_gap','missing_extension']:
        text.append(k+': '+(', '.join(s[k]) or '无'))
    text+=['','旧 blind-denoiser 队列继续独立运行，其结果作为历史诊断保存。它与显式时间 CFM 不合并成同一配置。','','所有新结果先通过有限值、来源、数值细化等检查，再进行定性视觉审阅。这里没有生成任何人类评分，也不把计算完成自动写成“充分训练”。']
    (O/'CelebA_实验进度与待办.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

def main():
    lock=(R/'finish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(R/'finish_supervisor.json',{'pid':os.getpid(),'started':time.time(),'state':'watching'})
    attempts=read(R/'qa/worker_recovery.json');(R/'qa').mkdir(exist_ok=True)
    workers=[('cfm',B/'supervisor.json',B/'code/run.py',B/'pipeline.log',{'completed_pending_review'}),('gap',R/'supervisor.json',R/'code/run_baseline.py',R/'pipeline.log',{'baseline_complete'}),('extension',R/'extension_status.json',R/'code/run_extension.py',R/'extension.log',{'extension_work_complete_pending_review'})]
    while True:
        s=snapshot();write_report(s)
        if s['state']=='compute_complete_pending_qualitative_review':
            for script,root in [(B/'code/plot_results.py',B),(B/'code/plot_earlystop.py',B),(R/'code/plot_results.py',R)]:subprocess.run([sys.executable,str(script),'--root',str(root)],check=True)
            s=snapshot();s['final_figures_rendered']=True;write_report(s);dump(R/'finish_supervisor.json',{'pid':os.getpid(),'state':s['state'],'finished':time.time()});break
        for name,statusfile,script,logfile,done in workers:
            status=read(statusfile)
            if status.get('state') in done or alive(status.get('pid')):continue
            # Numerical/configuration errors need inspection, not blind re-execution.
            if status.get('state') in {'error','sampler_review_required_before_extension'}:continue
            count=attempts.get(name,0)
            if count>=3:continue
            with logfile.open('ab') as log:proc=subprocess.Popen([sys.executable,'-u',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            attempts[name]=count+1;attempts['last_event']={'worker':name,'new_pid':proc.pid,'time':time.time(),'reason':'unexpected process exit; restart from saved checkpoint; no active process stopped'};dump(R/'qa/worker_recovery.json',attempts)
        time.sleep(60)
if __name__=='__main__':
    if '--once' in sys.argv:write_report(snapshot())
    else:
        try:main()
        except Exception:dump(R/'finish_supervisor.json',{'state':'error','traceback':traceback.format_exc()});raise
