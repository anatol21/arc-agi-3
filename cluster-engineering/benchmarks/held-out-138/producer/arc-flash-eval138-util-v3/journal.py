"""Atomic whole-batch journals, exact prefix chains, and resumable public predictions."""
import json
from pathlib import Path
from bridge import digest
import records
import policy


def request(prompt, arm, prefix):
    return dict(task_id=prompt['task_id'], arm=arm, messages=prompt['messages'], prefix=prefix)


def request_id(recipe, item):
    return records.digest(dict(benchmark_id=recipe['benchmark_id'], task_id=item['task_id'],
        arm=item['arm'], messages=item['messages'], prefix=item['prefix']))


def apply_batch(batch, states, recipe, prompts, tokenized):
    """Validate all rows before changing any state; no partial batch application."""
    unsigned = {k: v for k, v in batch.items() if k != 'sha256'}
    if batch.get('sha256') != digest(unsigned) or batch['benchmark_id'] != recipe['benchmark_id']:
        raise RuntimeError('Journal hash or benchmark identity mismatch')
    if batch['actual_batch_size'] != len(batch['rows']) or batch['arm'] not in recipe['arms']:
        raise RuntimeError('Invalid journal batch size/arm')
    prepared = [dict(request={'task_id':r['task_id']}, ids=[0]*r['generation_input_tokens'],
                    remaining=r['token_budget']-r['replayed_prefix_tokens']) for r in batch['rows']]
    if (not policy.accepts(prepared, recipe['batch_size']) or
            batch['admission'] != policy.admission_metrics(prepared) or
            batch['new_generated_tokens'] != sum(r['new_generated_tokens'] for r in batch['rows'])):
        raise RuntimeError('Journal admission or token ledger differs')
    updates = {}; keys = set()
    for row in batch['rows']:
        records.validate_record(row)
        if not isinstance(row.get('text'), str):
            raise RuntimeError('Missing decoded scorer text')
        key = (row['task_id'], row['arm'])
        if key in keys or row['task_id'] not in prompts or row['arm'] != batch['arm']:
            raise RuntimeError('Duplicate, unknown or mixed-arm row')
        keys.add(key)
        if key[0] not in recipe['eligible'].get(key[1],[]):
            raise RuntimeError('Previously finished/ineligible pair in new journal')
        previous = states.get(key)
        if previous and previous['finish_reason'] in ('eos', 'token_limit'):
            raise RuntimeError('Finished answer generated again')
        prefix = previous['completion_token_ids'] if previous else []
        prompt = prompts[key[0]]; ids = tokenized[key[0]]
        expected = request_id(recipe, request(prompt, key[1], prefix))
        if (row['benchmark_id'] != recipe['benchmark_id'] or row['request_id'] != expected or
                row['prompt_tokens'] != len(ids) or row['prompt_sha256'] != records.digest(ids) or
                row['messages_sha256'] != records.digest(prompt['messages']) or
                row['completion_token_ids'][:len(prefix)] != prefix or
                row['generation_input_tokens'] != len(ids) + len(prefix) or
                row['replayed_prefix_tokens'] != len(prefix) or
                row['generated_tokens'] != len(prefix) + row['new_generated_tokens']):
            raise RuntimeError('Response/prefix/prompt identity changed')
        updates[key] = row
    states.update(updates)


def collect(root, recipe, prompts, tokenized):
    by_id = {p['task_id']: p for p in prompts}; states = {}; batches = []
    for expected, path in enumerate(sorted((Path(root) / 'batches').glob('batch-*.json')), 1):
        batch = json.loads(path.read_text())
        if batch['sequence'] != expected or path.name != 'batch-%06d.json' % expected:
            raise RuntimeError('Missing or reordered journal sequence')
        apply_batch(batch, states, recipe, by_id, tokenized)
        batches.append(batch)
    return states, batches


def completed(states):
    return {k: r for k, r in states.items() if r['finish_reason'] in ('eos', 'token_limit')}
