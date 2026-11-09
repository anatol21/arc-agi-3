"""Separate CPU scorer. The generator never parses query reference answers."""
import json
import os
from pathlib import Path
from bridge import atomic, dependencies, output
from journal import collect, completed


def score_run(folder):
    parent, _, prompts, modules, recipe = dependencies(folder)
    root = output(folder)
    tokenized = json.loads((folder / 'tokenized-prompts.json').read_text())
    states, batches = collect(root, recipe, prompts, tokenized)
    predictions = completed(states)
    answers = modules['protocol'].rows(parent / 'answers.scorer-only.jsonl')
    summary, details = modules['protocol'].score(answers, predictions)
    seconds = sum(b['seconds'] for b in batches)
    tokens = sum(b['new_generated_tokens'] for b in batches)
    summary.update(benchmark_id=recipe['benchmark_id'], global_step=660,
        split='public_validation_138', adapter_sha256=recipe['adapter_sha256'],
        generation=recipe['generation'], policy=recipe['policy'],
        numerical_variant=recipe['numerical_scope'],
        allocation_cap_minutes=30, pending_responses=len(states)-len(predictions),
        generation_seconds=seconds, new_tokens=tokens,
        new_tokens_per_second=tokens/seconds if seconds else None,
        measured_batch_sizes=sorted({b['actual_batch_size'] for b in batches}),
        peak_allocated_GiB={str(i): max((v['peak_allocated_bytes']/2**30 for b in batches
            for v in b['memory'] if v['device'] == i), default=None) for i in range(3)},
        limits='Partial length-scheduled coverage is not representative full-set accuracy; no pooling with serial',
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
