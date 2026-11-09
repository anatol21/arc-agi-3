"""Thirty-minute, three-H100 direct-answer utilization experiment; no code execution."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid
from bridge import NAME, atomic, dependencies, digest, output
import batched
import journal
import policy
import records
from telemetry import Monitor


def allocation_budget(native_deadline, now=None):
    native = native_deadline.allocation_budget()
    stop = native['allocation_end'] - 90
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
        allocation_id=allocation['allocation_id']))
    return torch, model, tokenizer


def interleave(planned, arms):
    for index in range(max((len(planned[a]) for a in arms), default=0)):
        for arm in arms:
            if index < len(planned[arm]):
                yield arm, planned[arm][index]


def run(folder):
    parent, _, prompts, modules, recipe = dependencies(folder)
    modules['common'].verify_training_package(folder.parent, parent)
    modules['common'].verify_loader_sources(parent)
    root = output(folder)
    required_predictions = len(prompts) * len(recipe['arms'])
    tokenized = json.loads((folder / 'tokenized-prompts.json').read_text())
    if set(tokenized) != {p['task_id'] for p in prompts} or recipe['policy'] != policy.POLICY:
        raise RuntimeError('Staged tokens or aggressive policy changed')
    with modules['common'].output_lock(root):
        states, batches = journal.collect(root, recipe, prompts, tokenized)
        by_id = {p['task_id']: p for p in prompts}
        budget = allocation_budget(modules['deadline'])
        allocation = dict(allocation_id=uuid.uuid4().hex, job_id=os.environ.get('SLURM_JOB_ID'),
            benchmark_id=recipe['benchmark_id'], started_at=time.time(), phase='loading', budget=budget,
            prior_batch_calls=len(batches), prior_completed_predictions=len(journal.completed(states)))
        atomic(root / 'last-allocation.json', allocation)
        atomic(root / 'allocations' / (allocation['allocation_id'] + '.json'), allocation)
        os.environ['ARC_EVAL_ALLOCATION_ID'] = allocation['allocation_id']
        caught = []; handlers = {}; failure = None; torch = None
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda signum, frame: caught.append(signum))
        stop = lambda: bool(caught) or time.time() >= budget['stop_at']
        try:
            if not stop() and len(journal.completed(states)) < required_predictions:
                torch, model, tokenizer = load_model(folder, recipe, modules, root, allocation)
                for prompt in prompts:
                    if records.render(tokenizer, prompt['messages']) != tokenized[prompt['task_id']]:
                        raise RuntimeError('Reloaded tokenizer changed staged prompt tokens')
                allocation['phase'] = 'generating'
                atomic(root / 'last-allocation.json', allocation)
                while not stop() and len(journal.completed(states)) < required_predictions:
                    planned = {arm: [] for arm in recipe['arms']}
                    for arm in recipe['arms']:
                        prepared = []
                        for prompt in prompts:
                            previous = states.get((prompt['task_id'], arm))
                            if previous and previous['finish_reason'] in ('eos', 'token_limit'):
                                continue
                            prefix = previous['completion_token_ids'] if previous else []
                            item = batched.prepare(tokenizer, journal.request(prompt, arm, prefix))
                            if item is None:
                                raise RuntimeError('No context headroom; refusing prompt truncation')
                            if item['prompt'] != tokenized[prompt['task_id']]:
                                raise RuntimeError('Prepared prompt identity changed')
                            prepared.append(item)
                        planned[arm] = policy.groups(prepared, policy.MAX_BATCH)
                    if not any(planned.values()):
                        raise RuntimeError('Unfinished tasks produced no work')
                    for arm, group in interleave(planned, recipe['arms']):
                        if stop():
                            break
                        active = dict(job_id=allocation['job_id'], allocation_id=allocation['allocation_id'],
                            benchmark_id=recipe['benchmark_id'], arm=arm,
                            task_ids=[x['request']['task_id'] for x in group],
                            admission=policy.admission_metrics(group), started_at=time.time(),
                            completed_predictions=len(journal.completed(states)))
                        atomic(root / 'active-batch.json', active)
                        print('ARC_EVAL_BATCH_START ' + json.dumps(active), flush=True)
                        with Monitor() as monitor:
                            evidence = batched.generate(torch, model, tokenizer, group, arm,
                                recipe, stop, modules['run_benchmark'])
                        evidence.update(sequence=len(batches)+1, benchmark_id=recipe['benchmark_id'], allocation_id=allocation['allocation_id'],
                            gpu_telemetry=monitor.report())
                        evidence['sha256'] = digest(evidence)
                        # A single atomic journal commits every row together. Recovery replays this first.
                        atomic(root / 'batches' / ('batch-%06d.json' % evidence['sequence']), evidence)
                        journal.apply_batch(evidence, states, recipe, by_id, tokenized)
                        batches.append(evidence)
                        done = journal.completed(states)
                        progress = dict(phase='generating', job_id=allocation['job_id'],
                            completed_predictions=len(done), required_predictions=required_predictions,
                            completed_by_arm={a:sum(k[1] == a for k in done) for a in recipe['arms']},
                            paired_tasks=sum(all((p['task_id'], a) in done for a in recipe['arms']) for p in prompts),
                            batch_calls=len(batches), pending_responses=len(states)-len(done), updated_at=time.time())
                        atomic(root / 'progress.json', progress)
                        print('ARC_EVAL_BATCH ' + json.dumps(dict(arm=arm,
                            actual_batch_size=evidence['actual_batch_size'], seconds=evidence['seconds'],
                            new_generated_tokens=evidence['new_generated_tokens'],
                            tokens_per_second=evidence['tokens_per_second'], memory=evidence['memory'],
                            gpu_busy_percent=evidence['gpu_telemetry']['mean_gpu_busy_percent'],
                            completed_predictions=len(done), required_predictions=required_predictions)), flush=True)
            allocation['phase'] = 'complete' if len(journal.completed(states)) == required_predictions else 'partial_checkpointed'
        except BaseException as error:
            failure = error
            allocation.update(phase='failed', error=repr(error), traceback=traceback.format_exc())
            if torch is not None:
                try:
                    allocation['failure_memory'] = [dict(device=i,
                        allocated_bytes=torch.cuda.memory_allocated(i),
                        reserved_bytes=torch.cuda.memory_reserved(i)) for i in range(3)]
                except Exception:
                    pass
            atomic(root / 'failure.json', allocation)
        finally:
            allocation.update(finished_at=time.time(), received_signals=caught,
                completed_predictions=len(journal.completed(states)), total_batch_calls=len(batches))
            for sig, previous in handlers.items():
                signal.signal(sig, previous)
            atomic(root / 'last-allocation.json', allocation)
            atomic(root / 'allocations' / (allocation['allocation_id'] + '.json'), allocation)
            # CPU scorer runs even after OOM. Only this child reads reference query answers.
            result = subprocess.run([sys.executable, str(folder / 'score.py')], check=False,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
            if result.returncode and failure is None:
                failure = RuntimeError('Saved predictions could not be scored; inspect CPU scorer log')
        if failure:
            raise failure


if __name__ == '__main__':
    run(Path(__file__).resolve().parent)
