"""One frozen model; globally grouped, alternating same-arm generation waves."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
if __package__:
    from .runtime import NAME,ready,variant_recipe
    from . import controller,records,batched,scheduler,telemetry
else:
    from runtime import NAME,ready,variant_recipe
    import controller,records,batched,scheduler,telemetry

def save_state(output,task,arm,state,recipe,io):
    state.update(benchmark_id=recipe['benchmark_id'],task_id=task['task_id'],arm=arm)
    trace=output/task['task_id']/arm
    io.atomic_json(trace/'state.json',state)
    for a in state['attempts']:
        stem='candidate-'+str(a['candidate'])
        io.atomic_json(trace/(stem+'.json'),a)
        io.atomic_text(trace/(stem+'.raw.txt'),a['response']['raw_text'])
        if 'code' in a:io.atomic_text(trace/(stem+'.py'),a['code'])
    if state['complete']:
        for label in ('first','selected'):io.atomic_json(trace/(label+'-output.json'),state[label+'_output_json'])

def load_model(folder,recipe,m,allocation):
    import torch
    from axolotl.cli.utils import load_model_and_tokenizer
    if torch.cuda.device_count()!=3 or int(os.environ.get('WORLD_SIZE','1'))!=1:raise RuntimeError('One process and three GPUs required')
    start=time.monotonic();torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model,tokenizer,_=load_model_and_tokenizer(cfg=m['configuration'].load_inference_config(folder/'inference-config.json'),inference=False)
    m['ple_placement'].place_small_ple_buffers(model);layout=m['guarded_train'].check_layout(model)
    if layout['trainable_parameters']!=33478656:raise RuntimeError('Adapter coverage changed')
    adapter=m['checkpoint_validation'].validate_adapters(torch,Path(recipe['checkpoint']),model)
    if str(model.get_input_embeddings().weight.device)!='cuda:0':raise RuntimeError('Input placement changed')
    if tokenizer.eos_token_id!=recipe['eos_token_id'] or tokenizer.pad_token_id!=recipe['pad_token_id']:raise RuntimeError('Tokenizer changed')
    model.requires_grad_(False);model.eval();model.gradient_checkpointing_disable()
    for module in model.modules():
        config=getattr(module,'config',None)
        if config is not None and hasattr(config,'use_cache'):config.use_cache=True
    proof={'layout':layout,'adapter':adapter,'optimizer_updates':0,'requires_grad_parameters':sum(p.requires_grad for p in model.parameters()),'allocation_id':allocation['allocation_id'],'loading_seconds':time.monotonic()-start}
    m['result_io'].atomic_json(folder.parent/'runs'/NAME/'model-reload.json',proof)
    return torch,model,tokenizer

def debug_targets(tasks, initial_prompt_lengths):
    """Use a fixed prefix plus the globally longest prompt, independent of cap."""
    if not tasks:
        return []
    if any(t['task_id'] not in initial_prompt_lengths for t in tasks):
        raise ValueError('Initial prompt lengths are missing debug task identities')
    longest = min(tasks, key=lambda t: (-initial_prompt_lengths[t['task_id']], t['task_id']))
    wanted = {t['task_id'] for t in tasks[:32]} | {longest['task_id']}
    return [t for t in tasks if t['task_id'] in wanted]


def alternating_groups(planned, arm_order):
    """Interleave ready groups without concurrent model or adapter operations."""
    for index in range(max((len(planned[a]) for a in arm_order), default=0)):
        for arm in arm_order:
            if index < len(planned[arm]):
                yield arm, planned[arm][index]


def run_variant(folder,recipe,tasks,m,loaded,allocation,stop):
    io=m['result_io'];output=folder.parent/'runs'/NAME/('batch'+str(recipe['batch_size']))
    if (output/'recipe.json').exists() and json.loads((output/'recipe.json').read_text())!=recipe:raise RuntimeError('Existing variant recipe differs')
    io.atomic_json(output/'recipe.json',recipe);cache=scheduler.load_cache(output,recipe)
    states={}
    for task in tasks:
        for arm in recipe['arm_order']:
            path=output/task['task_id']/arm/'state.json'
            state=json.loads(path.read_text()) if path.exists() else controller.initial_state(task['visible'])
            if path.exists() and (state.get('benchmark_id')!=recipe['benchmark_id'] or state.get('task_id')!=task['task_id'] or state.get('arm')!=arm):raise RuntimeError('State identity differs')
            controller.validate_state(task['visible'],state);states[task['task_id'],arm]=state
    task_by_id={task['task_id']:task for task in tasks}
    if len(task_by_id)!=len(tasks):raise RuntimeError('Duplicate target task identity')

    def drive(task,arm,consume_only=False):
        state=states[task['task_id'],arm]
        save=lambda s:save_state(output,task,arm,s,recipe,io)
        def delivery_stop():
            # A durable generation response must be applied after the cutoff.
            # Once it is applied, its pending request disappears/changes; no
            # execution or new generation is authorized beyond that cutoff.
            pending=state['pending']
            if pending:
                req=scheduler.request_for(task,arm,pending['messages'],pending['prefix'])
                if batched.request_id(recipe,req) in cache:return False
            return stop()
        return scheduler.drive(task,arm,state,recipe,cache,m['sandbox'].execute,
                               save,delivery_stop if consume_only else stop)

    # Every unfinished task contributes at most one request to a generation wave.
    # Global length grouping avoids a slow cohort blocking later independent
    # tasks. Repairs and checkpoint continuations enter the following wave.
    while not stop():
        requests={arm:[] for arm in recipe['arm_order']}
        for arm in recipe['arm_order']:
            for task in tasks:
                request=drive(task,arm)
                if request:requests[arm].append(request)
                if stop():return loaded
        if not any(requests.values()):
            if all(s['complete'] for s in states.values()):return loaded
            raise RuntimeError('Unfinished states yielded no generation request')
        if loaded is None:
            loaded=load_model(folder,recipe,m,allocation)
            if stop():return loaded
        torch,model,tokenizer=loaded
        planned={arm:[] for arm in recipe['arm_order']}
        for arm in recipe['arm_order']:
            prepared=[]
            for req in requests[arm]:
                row=batched.prepare(tokenizer,req)
                if row is None:
                    cache[batched.request_id(recipe,req)]={'finish_reason':'context_limit'}
                    drive(task_by_id[req['task_id']],arm,consume_only=True)
                else:prepared.append(row)
                if stop():return loaded
            planned[arm]=batched.groups(prepared,recipe['batch_size'])
        for arm,group in alternating_groups(planned,recipe['arm_order']):
            if stop():return loaded
            with telemetry.Monitor() as monitor:
                evidence=batched.generate(torch,model,tokenizer,group,arm,recipe,stop,m['generation'])
            evidence['gpu_telemetry']=monitor.report()
            evidence.update(benchmark_id=recipe['benchmark_id'],allocation_id=allocation['allocation_id'])
            evidence['sha256']=records.digest(evidence)
            # Journal all rows atomically before updating any task controller.
            io.atomic_json(output/'batches'/(evidence['batch_id']+'.json'),evidence)
            for r in evidence['rows']:cache[r['request_id']]=r
            for item in group:
                drive(task_by_id[item['request']['task_id']],arm,consume_only=True)
            print('ARC_CODE_BATCH '+json.dumps({k:evidence[k] for k in ('batch_id','arm','actual_batch_size','seconds','new_generated_tokens','tokens_per_second','memory')}),flush=True)
            if stop():return loaded
    return loaded

def main(folder,debug=True,batch_size=32):
    _,recipe,m=ready(folder);io=m['result_io'];root=folder.parent/'runs'/NAME
    from_path=folder/'review.py'
    if not debug:
        # Explicit full invocation plus successful debug evidence required; never
        # auto-submit or silently fall back to a different batch size after OOM.
        result=subprocess.run([sys.executable,str(from_path),'--require-pass','--batch-size',str(batch_size)],check=False)
        if result.returncode:raise RuntimeError('Review a successful matching debug before a full allocation')
    tasks=m['protocol'].rows(folder/'tasks.jsonl')
    lengths=json.loads((folder/'template-check.json').read_text())['initial_prompt_lengths']
    targets=debug_targets(tasks,lengths) if debug else tasks
    # Exercise the real adaptive workload first. Both it and the four-task
    # serial control must eventually pass review; partial work is resumable.
    sizes=[batch_size,1] if debug else [batch_size]
    if len(set(sizes))!=len(sizes):sizes=[1]
    with m['run_checks'].output_lock(root):
        budget=m['deadline'].allocation_budget();signals=[];handlers={}
        for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.getsignal(sig);signal.signal(sig,lambda n,f:signals.append(n))
        stop=lambda:bool(signals) or time.time()>=budget['stop_at']
        allocation={'allocation_id':uuid.uuid4().hex,'job_id':os.environ.get('SLURM_JOB_ID'),'phase':'running','debug':debug,'batch_size':batch_size,'target_tasks':[t['task_id'] for t in targets],'started_at':io.utcnow(),'budget':budget}
        os.environ['ARC_CODE_ALLOCATION_ID']=allocation['allocation_id'];loaded=None;failure=None
        io.atomic_json(root/'last-allocation.json',allocation)
        try:
            io.atomic_json(root/'sandbox-preflight.json',m['sandbox'].preflight())
            for size in sizes:
                if stop():break
                loaded=run_variant(folder,variant_recipe(recipe,size),targets[:4] if debug and size==1 else targets,m,loaded,allocation,stop)
            allocation['phase']='allocation_finished'
        except BaseException as e:
            failure=e;allocation.update(phase='failed',error=repr(e),traceback=traceback.format_exc())
        finally:
            allocation['finished_at']=io.utcnow()
            io.atomic_json(root/'allocations'/(allocation['allocation_id']+'.json'),allocation);io.atomic_json(root/'last-allocation.json',allocation)
            for sig,h in handlers.items():signal.signal(sig,h)
        if failure:raise failure
    subprocess.run([sys.executable,str(folder/'summarize.py')],check=True)
    if debug:subprocess.run([sys.executable,str(folder/'review.py'),'--batch-size',str(batch_size)],check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');parser.add_argument('--batch-size',type=int,choices=(1,2,4,8,16,32),default=32);args=parser.parse_args()
    main(Path(__file__).resolve().parent,debug=not args.full,batch_size=args.batch_size)
