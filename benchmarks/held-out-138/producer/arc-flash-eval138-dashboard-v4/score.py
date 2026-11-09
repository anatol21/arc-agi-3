"""Separate CPU scorer. The generator never parses query reference answers."""
import json
import os
import sys
from pathlib import Path
from bridge import atomic, dependencies, output, digest
from journal import collect, completed


def verify_runtime():
    if sys.version_info < (3,10):
        raise RuntimeError('Use .venv-arc/bin/python (Python>=3.10); no scores were written')
    if list(zip([1],[1],strict=True)) != [(1,1)]:
        raise RuntimeError('Unsupported scorer runtime')


def score_run(folder):
    verify_runtime()
    parent, _, prompts, modules, recipe = dependencies(folder)
    root = output(folder)
    tokenized = json.loads((folder / 'tokenized-prompts.json').read_text())
    states, batches = collect(root, recipe, prompts, tokenized)
    predictions = completed(states)
    answers = modules['protocol'].rows(parent / 'answers.scorer-only.jsonl')
    if len(answers)!=138:raise RuntimeError('Expected original138 public references')
    scored_predictions={k:dict(v,text='' if v['finish_reason']=='token_limit' else v['text'])for k,v in predictions.items()}
    summary, details = modules['protocol'].score(answers, scored_predictions)
    for arm,ids in recipe['eligible'].items():
        part=[a for a in answers if a['task_id']in set(ids)]
        summary['arms'][arm]=modules['protocol'].score(part,scored_predictions)[0]['arms'][arm]
    summary['complete']=all(a['complete']for a in summary['arms'].values())
    for row in details:
        if predictions[(row['task_id'],row['arm'])]['finish_reason']=='token_limit':
            row['format_error']='Output token limit reached; scored as failure without repair'
    failed=[]
    for path in (root/'oom-attempts').glob('*.json'):
        attempt=json.loads(path.read_text())
        if attempt.get('benchmark_id')!=recipe['benchmark_id']or digest({k:v for k,v in attempt.items()if k!='sha256'})!=attempt['sha256']:
            raise RuntimeError('Invalid failed-attempt cost record')
        failed.append(attempt)
    seconds = sum(b['seconds'] for b in batches)
    tokens = sum(b['new_generated_tokens'] for b in batches)
    failed_seconds=sum(a['attempt_wall_seconds']for a in failed)
    attempt_seconds=sum(b.get('attempt_wall_seconds',b['seconds'])for b in batches)+failed_seconds
    samples=[s for a in batches+failed for s in a.get('gpu_telemetry',{}).get('samples',[])]
    activity={}
    for i in range(3):
        values=[s['gpu_busy_percent']for s in samples if s['cuda_ordinal']==i]
        activity[str(i)]=dict(mean_busy_percent=sum(values)/len(values)if values else None,samples=len(values))
    summary.update(benchmark_id=recipe['benchmark_id'], global_step=660,
        split='public_validation_unfinished_pairs',dataset_tasks=138,
        eligible_by_arm={a:len(ids)for a,ids in recipe['eligible'].items()},
        excluded_prior_finished_by_arm={a:138-len(ids)for a,ids in recipe['eligible'].items()},
        unattempted_eligible_pairs=sum(len(ids)for ids in recipe['eligible'].values())-len(states),
        deferred=json.loads((root/'deferred.json').read_text())if(root/'deferred.json').exists()else{},
        adapter_sha256=recipe['adapter_sha256'],
        generation=recipe['generation'], policy=recipe['policy'],
        numerical_variant=recipe['numerical_scope'],
        allocation_cap_minutes=30, pending_responses=len(states)-len(predictions),
        generation_seconds=seconds, new_tokens=tokens,
        new_tokens_per_second=tokens/seconds if seconds else None,
        failed_attempts=len(failed),failed_attempt_wall_seconds=failed_seconds,
        all_attempt_wall_seconds=attempt_seconds,
        new_tokens_per_attempt_wall_second_including_oom=tokens/attempt_seconds if attempt_seconds else None,
        failed_attempt_generated_tokens_unknown=bool(failed),
        replayed_prefix_tokens=sum(b['replayed_prefix_tokens']for b in batches),
        gpu_activity_by_cuda_ordinal=activity,
        gpu_activity_scope='complete three-UUID frames during attempted generation; excludes loading and idle host work',
        measured_batch_sizes=sorted({b['actual_batch_size'] for b in batches}),
        peak_allocated_GiB={str(i): max((v['peak_allocated_bytes']/2**30 for b in batches
            for v in b['memory'] if v['device'] == i), default=None) for i in range(3)},
        limits='Unfinished-pair selection and a batch floor defer work; incomplete results are not representative138-task accuracy. No prior responses pooled.',
        ttt=False, python_execution=False)
    atomic(root / 'summary.json', summary)
    atomic(root / 'per-task.json', details)
    # Derived complete answers only; unfinished tokens remain in durable batch journals.
    atomic(root / 'completed-predictions.json', list(predictions.values()))
    print('ARC_EVAL_UTIL_SCORE ' + json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    score_run(Path(__file__).resolve().parent)
