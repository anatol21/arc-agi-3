"""Separate non-scoring batch-capacity ladder; never execute or reuse its outputs."""
import argparse
import json
import os
import signal
import time
import uuid
from pathlib import Path
if __package__:
    from .runtime import NAME,ready,variant_recipe
    from .run import load_model
    from . import controller,batched,records,telemetry
else:
    from runtime import NAME,ready,variant_recipe
    from run import load_model
    import controller,batched,records,telemetry

def main(folder,window=128):
    _,base,m=ready(folder);io=m['result_io'];out=folder.parent/'runs'/NAME
    with m['run_checks'].output_lock(out):
        budget=m['deadline'].allocation_budget();signals=[];handlers={}
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):handlers[sig]=signal.getsignal(sig);signal.signal(sig,lambda n,f:signals.append(n))
        stop=lambda:bool(signals) or time.time()>=budget['stop_at'];loaded=None
        allocation={'allocation_id':uuid.uuid4().hex,'job_id':os.environ.get('SLURM_JOB_ID'),'phase':'running','engineering_only':True,'started_at':io.utcnow(),'window':window}
        os.environ['ARC_CODE_ALLOCATION_ID']=allocation['allocation_id']
        try:
            io.atomic_json(out/'capacity-sandbox-preflight.json',m['sandbox'].preflight())
            tasks=m['protocol'].rows(folder/'tasks.jsonl')
            # Identical representative prompts make short-window throughput
            # comparable across sizes. This is not a heterogeneous ARC solve test.
            lengths=json.loads((folder/'template-check.json').read_text())['initial_prompt_lengths']
            ordered=sorted(tasks,key=lambda t:(lengths[t['task_id']],t['task_id']))
            source=ordered[(len(ordered)-1)//2]
            allocation.update(source_task_id=source['task_id'],source_prompt_tokens=lengths[source['task_id']],profile='fixed median-length prompt duplicated per row')
            for size in (4,8,16,32,64):
                recipe=variant_recipe(base,size)
                for arm in base['arm_order']:
                    path=out/'capacity'/('window'+str(window))/('batch'+str(size)+'-'+arm+'.json')
                    if path.exists():
                        existing=json.loads(path.read_text())
                        if existing.get('benchmark_id')!=recipe['benchmark_id'] or existing.get('capacity_limit')!=window:raise RuntimeError('Capacity identity differs')
                        if existing.get('phase')=='complete':continue
                        if existing.get('phase')=='oom':raise RuntimeError('Previous capacity OOM: review before another attempt')
                        # Interrupted engineering probes cannot be treated as full
                        # capacity passes. Preserve them, require explicit review.
                        raise RuntimeError('Interrupted capacity probe requires review; original artifact preserved')
                    if stop():allocation['phase']='incomplete';return
                    if loaded is None:loaded=load_model(folder,recipe,m,allocation)
                    if stop():allocation['phase']='incomplete';return
                    torch,model,tokenizer=loaded;prepared=[]
                    for row_index in range(size):
                        req={'task_id':source['task_id']+'-capacity-'+str(row_index),'arm':arm,'messages':controller.messages_for(source['visible'],[]),'prefix':[]}
                        row=batched.prepare(tokenizer,req)
                        if row is None:raise RuntimeError('Capacity prompt exceeds context')
                        row['remaining']=min(row['remaining'],window);prepared.append(row)
                    if len(batched.groups(prepared,size))!=1:raise RuntimeError('Capacity batch cannot fit this padded context without changing prompts')
                    try:
                        with telemetry.Monitor() as monitor:
                            result=batched.generate(torch,model,tokenizer,prepared,arm,recipe,stop,m['generation'],capacity_limit=window)
                        result.update(source_task_id=source['task_id'],source_prompt_tokens=lengths[source['task_id']],profile='fixed median-length prompt duplicated per row',gpu_telemetry=monitor.report(),benchmark_id=recipe['benchmark_id'],phase='incomplete' if any(r['finish_reason']=='interrupted' for r in result['rows']) else 'complete')
                        io.atomic_json(path,result)
                        print('CAPACITY '+json.dumps({k:result[k] for k in ('actual_batch_size','arm','seconds','tokens_per_second','memory','phase')}),flush=True)
                        if result['phase']!='complete':allocation['phase']='incomplete';return
                    except torch.OutOfMemoryError as failure:
                        io.atomic_json(path,{'phase':'oom','benchmark_id':recipe['benchmark_id'],'capacity_limit':window,'batch_size':size,'arm':arm,'error':str(failure),'engineering_only':True})
                        allocation['phase']='stopped_at_oom';return
            allocation['phase']='short_window_ladder_complete'
        except BaseException as failure:
            allocation.update(phase='failed',error=repr(failure));raise
        finally:
            allocation['finished_at']=io.utcnow();io.atomic_json(out/'capacity-allocations'/(allocation['allocation_id']+'.json'),allocation)
            for sig,h in handlers.items():signal.signal(sig,h)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--window',type=int,choices=(128,1024),default=128);a=p.parse_args();main(Path(__file__).resolve().parent,a.window)
