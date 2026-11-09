"""Dashboard-managed native large-batch inference; no code execution."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid
import gc
from bridge import NAME, atomic, dependencies, digest, output
import batched
import journal
import policy
import records
from telemetry import Monitor, cuda_gpu_uuids
import scheduling


dashboard_stopping = lambda: False
dashboard_terminal = lambda row: None
dashboard_progress = lambda *args: None

def allocation_budget(native_deadline, now=None):
    native = native_deadline.allocation_budget()
    stop = native['allocation_end'] - 90
    if os.environ.get('WM_DEADLINE'):
        stop = min(stop, float(os.environ['WM_DEADLINE']))
    return dict(native, stop_at=stop, reserve_seconds=90,
                remaining_generation_seconds=max(0, stop - (time.time() if now is None else now)))


def load_model(folder, recipe, modules, root, allocation):
    import torch
    from axolotl.cli.utils import load_model_and_tokenizer
    if torch.cuda.device_count() != 3 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('One process with three GPUs is required')
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    started = time.monotonic()
    cfg = modules['configuration'].load_inference_config(folder / 'inference-config.json')
    cfg['output_dir'] = str(root / 'loader')
    model, tokenizer, _ = load_model_and_tokenizer(cfg=cfg, inference=False)
    modules['ple_placement'].place_small_ple_buffers(model)
    layout = modules['guarded_train'].check_layout(model)
    if layout['trainable_parameters'] != 33478656:
        raise RuntimeError('LoRA parameter coverage changed')
    adapter = modules['checkpoint_validation'].validate_adapters(torch, Path(recipe['checkpoint']), model)
    if str(model.get_input_embeddings().weight.device) != 'cuda:0':
        raise RuntimeError('Input embedding placement changed')
    if (tokenizer.eos_token_id != recipe['eos_token_id'] or
            tokenizer.pad_token_id != recipe['pad_token_id']):
        raise RuntimeError('Tokenizer identity changed')
    model.requires_grad_(False); model.eval(); model.gradient_checkpointing_disable()
    for module in model.modules():
        cfg = getattr(module, 'config', None)
        if cfg is not None and hasattr(cfg, 'use_cache'):
            cfg.use_cache = True
    atomic(root / 'model-reload.json', dict(layout=layout, adapter=adapter,
        requires_grad_parameters=sum(p.requires_grad for p in model.parameters()),
        optimizer_updates=0, loading_seconds=time.monotonic()-started,
        allocation_id=allocation['allocation_id'], cuda_uuid_mapping={str(i):u for i,u in enumerate(cuda_gpu_uuids(torch))}))
    return torch, model, tokenizer


def memory_snapshot(torch):
    if torch is None:
        return []
    try:
        return [dict(device=i,allocated_bytes=torch.cuda.memory_allocated(i),
            reserved_bytes=torch.cuda.memory_reserved(i),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(i),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(i))for i in range(3)]
    except Exception:
        return []


def attempt(torch, model, tokenizer, group, arm, recipe, stop, frozen, uuids):
    """Discard the exception frame before empty_cache; it may own large temporary tensors."""
    started=time.monotonic(); evidence=None; oom=None
    with Monitor(uuids) as monitor:
        try:
            evidence=batched.generate(torch,model,tokenizer,group,arm,recipe,stop,frozen,on_terminal=dashboard_terminal)
        except Exception as error:
            if torch is None or not isinstance(error,torch.OutOfMemoryError):
                raise
            oom=dict(error=repr(error),failed_attempt_seconds=time.monotonic()-started,
                memory=memory_snapshot(torch),new_tokens_unknown=True)
    return evidence,oom,monitor.report(),time.monotonic()-started


def clear_failed_temporaries(torch):
    gc.collect()
    for i in range(3):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


def pending_rows(prompts, arm, states, tokenizer, tokenized, eligible):
    prepared=[]
    for prompt in prompts:
        if prompt['task_id'] not in eligible[arm]:continue
        previous=states.get((prompt['task_id'],arm))
        if previous and previous['finish_reason'] in ('eos','token_limit'):
            continue
        prefix=previous['completion_token_ids'] if previous else []
        item=batched.prepare(tokenizer,journal.request(prompt,arm,prefix))
        if item is None or item['prompt']!=tokenized[prompt['task_id']]:
            raise RuntimeError('No headroom or changed complete prompt; no truncation')
        prepared.append(item)
    return prepared


def required(recipe):
    return sum(len(ids)for ids in recipe['eligible'].values())


def progress_record(states,batches,prompts,recipe,allocation):
    done=journal.completed(states)
    return dict(phase='generating',job_id=allocation['job_id'],allocation_id=allocation['allocation_id'],
        completed_predictions=len(done),required_predictions=required(recipe),
        eligible_by_arm={a:len(ids)for a,ids in recipe['eligible'].items()},
        completed_by_arm={a:sum(k[1]==a for k in done)for a in recipe['arms']},
        batch_calls=len(batches),pending_responses=len(states)-len(done),
        targets=allocation['targets'],failed_attempts=allocation['failed_attempts'],
        blocked_arms=allocation['blocked_arms'],updated_at=time.time())


def drive(torch,model,tokenizer,folder,root,prompts,modules,recipe,states,batches,allocation,stop,uuids,tokenized):
    by_id={p['task_id']:p for p in prompts}
    from admission import Admission, hardware
    learned = Admission(root,recipe['benchmark_id'],hardware(torch),batches)
    last={a:{}for a in recipe['arms']}
    for b in batches:
        for row in b['rows']:last[row['arm']][row['task_id']]=b['sequence']
    while not stop() and len(journal.completed(states))<required(recipe):
        acted=False
        for arm in recipe['arms']:
            if stop():break
            if arm in allocation['blocked_arms']:continue
            prepared=pending_rows(prompts,arm,states,tokenizer,tokenized,recipe['eligible'])
            if not prepared:continue
            group,target,band=learned.choose(prepared,recipe['cohorts'],last[arm],arm)
            allocation['targets'][arm]=target
            if not group:
                allocation['deferred'][arm]=dict(reason='No distinct eligible group fits the minimum18 and fixed token bounds',
                    task_ids=[x['request']['task_id']for x in prepared],remaining=len(prepared))
                continue
            acted=True
            active=dict(hardware=learned.devices,length_band=band,
                size_probe=len(group)>learned.safe.get((arm,band),0),job_id=allocation['job_id'],allocation_id=allocation['allocation_id'],
                benchmark_id=recipe['benchmark_id'],arm=arm,target=allocation['targets'][arm],
                task_ids=[x['request']['task_id']for x in group],
                continuation_rows=sum(bool(x['prefix'])for x in group),
                admission=policy.admission_metrics(group),started_at=time.time(),
                completed_predictions=len(journal.completed(states)))
            atomic(root/'active-batch.json',active)
            print('ARC_EVAL_BATCH_START '+json.dumps(active),flush=True)
            evidence,oom,telemetry,elapsed=attempt(torch,model,tokenizer,group,arm,recipe,stop,
                modules['run_benchmark'],uuids)
            if oom is not None:
                journal.merge_terminals(root,states,recipe,by_id,tokenized)
                allocation['failed_attempts']+=1
                lower=scheduling.after_oom(allocation['targets'][arm],len(group))
                failure=dict(**active,**oom,gpu_telemetry=telemetry,
                    failed_attempt_id=uuid.uuid4().hex,ended_at=time.time(),attempt_wall_seconds=elapsed,
                    action='lower request target; never below18; preserve committed work',new_target=lower)
                failure['sha256']=digest(failure)
                atomic(root/'oom-attempts'/(failure['failed_attempt_id']+'.json'),failure)
                learned.observe(arm,band,len(group),lower,failed=True)
                clear_failed_temporaries(torch)
                if lower is None:
                    allocation['blocked_arms'].append(arm)
                    allocation['deferred'][arm]=dict(reason='OOM at the18-request floor; smaller batches forbidden',
                        task_ids=[x['request']['task_id']for x in prepared],remaining=len(prepared))
                else:
                    allocation['targets'][arm]=lower
                    allocation['maximum_targets'][arm]=lower
                atomic(root/'last-allocation.json',allocation)
                print('ARC_EVAL_OOM '+json.dumps(dict(arm=arm,failed_size=len(group),new_target=lower)),flush=True)
                if allocation['failed_attempts']>=policy.POLICY['max_failed_attempts_per_allocation']:
                    raise RuntimeError('OOM attempt budget exhausted; journals and failed costs retained')
                continue
            learned.observe(arm,band,len(group),None,failed=False)
            evidence.update(hardware=learned.devices,length_band=band,size_probe=active['size_probe'],sequence=len(batches)+1,benchmark_id=recipe['benchmark_id'],
                allocation_id=allocation['allocation_id'],gpu_telemetry=telemetry,
                target=allocation['targets'][arm],attempt_wall_seconds=elapsed)
            for row in evidence['rows']:
                if row['finish_reason'] in ('eos','token_limit'):
                    saved=journal.load_terminal(root/'terminals'/f"{arm}-{row['task_id']}.json")
                    if not journal.same_terminal(saved,row):
                        raise RuntimeError('Terminal tokens changed before batch commit')
            evidence['sha256']=digest(evidence)
            atomic(root/'batches'/('batch-%06d.json'%evidence['sequence']),evidence)
            journal.apply_batch(evidence,states,recipe,by_id,tokenized)
            batches.append(evidence)
            for row in evidence['rows']:last[arm][row['task_id']]=evidence['sequence']
            allocation['maximum_targets'][arm]=learned.caps.get((arm,band),64)
            allocation['deferred'].pop(arm,None)
            progress=progress_record(states,batches,prompts,recipe,allocation)
            atomic(root/'progress.json',progress)
            dashboard_progress(states, batches, allocation)
            allocation.update(last_progress_at=progress['updated_at'],
                completed_predictions=progress['completed_predictions'],total_batch_calls=len(batches))
            atomic(root/'last-allocation.json',allocation)
            print('ARC_EVAL_BATCH '+json.dumps(dict(arm=arm,actual_batch_size=evidence['actual_batch_size'],
                seconds=evidence['seconds'],new_generated_tokens=evidence['new_generated_tokens'],
                tokens_per_second=evidence['tokens_per_second'],memory=evidence['memory'],
                gpu_busy_percent=telemetry['mean_gpu_busy_percent'],completed_predictions=len(journal.completed(states)),
                required_predictions=required(recipe))),flush=True)
        if not acted:
            atomic(root/'deferred.json',allocation['deferred'])
            return


def run(folder):
    from score import verify_runtime
    verify_runtime()
    parent,_,prompts,modules,recipe=dependencies(folder)
    modules['common'].verify_training_package(folder.parent,parent)
    modules['common'].verify_loader_sources(parent)
    root=output(folder);tokenized=json.loads((folder/'tokenized-prompts.json').read_text())
    if set(tokenized)!={p['task_id']for p in prompts}or recipe['policy']!=policy.POLICY:
        raise RuntimeError('Staged token/policy identity changed')
    with modules['common'].output_lock(root):
        states,batches=journal.collect(root,recipe,prompts,tokenized)
        budget=allocation_budget(modules['deadline'])
        allocation=dict(allocation_id=uuid.uuid4().hex,job_id=os.environ.get('SLURM_JOB_ID'),
            benchmark_id=recipe['benchmark_id'],started_at=time.time(),phase='loading',budget=budget,
            prior_batch_calls=len(batches),prior_completed_predictions=len(journal.completed(states)),
            failed_attempts=0,targets={a:24 for a in recipe['arms']},
            maximum_targets={a:64 for a in recipe['arms']},blocked_arms=[],deferred={})
        atomic(root/'last-allocation.json',allocation)
        os.environ['ARC_EVAL_ALLOCATION_ID']=allocation['allocation_id']
        caught=[];handlers={};failure=None;torch=None
        for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
            handlers[sig]=signal.getsignal(sig)
            signal.signal(sig,lambda signum,frame:caught.append(signum))
        stop=lambda:bool(caught)or time.time()>=budget['stop_at']or dashboard_stopping()
        try:
            if not stop()and len(journal.completed(states))<required(recipe):
                torch,model,tokenizer=load_model(folder,recipe,modules,root,allocation)
                uuids=cuda_gpu_uuids(torch)
                for prompt in prompts:
                    if records.render(tokenizer,prompt['messages'])!=tokenized[prompt['task_id']]:
                        raise RuntimeError('Reloaded template changed staged prompt tokens')
                allocation['phase']='generating';atomic(root/'last-allocation.json',allocation)
                drive(torch,model,tokenizer,folder,root,prompts,modules,recipe,states,batches,
                    allocation,stop,uuids,tokenized)
            allocation['phase']='complete'if len(journal.completed(states))==required(recipe)else'partial_checkpointed'
        except BaseException as error:
            failure=error
            allocation.update(phase='failed',error=repr(error),traceback=traceback.format_exc(),failure_memory=memory_snapshot(torch))
            atomic(root/'failure.json',allocation)
        finally:
            atomic(root/'deferred.json',allocation['deferred'])
            allocation.update(finished_at=time.time(),received_signals=caught,
                completed_predictions=len(journal.completed(states)),total_batch_calls=len(batches))
            allocation['process_elapsed_seconds']=allocation['finished_at']-allocation['started_at']
            for sig,previous in handlers.items():signal.signal(sig,previous)
            atomic(root/'last-allocation.json',allocation)
            atomic(root/'allocations'/(allocation['allocation_id']+'.json'),allocation)
            result=subprocess.run([sys.executable,str(folder/'score.py')],check=False,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'))
            if result.returncode and failure is None:
                failure=RuntimeError('CPU scoring failed; committed journals retained')
        if failure:raise failure


if __name__=='__main__':run(Path(__file__).resolve().parent)


