"""Atomic whole-batch journals, exact prefix chains, and resumable public predictions."""
import json
import os
import tempfile
from pathlib import Path
from bridge import digest
import records
import policy


def request(prompt, arm, prefix):
    return dict(task_id=prompt['task_id'], arm=arm, messages=prompt['messages'], prefix=prefix)


def request_id(recipe, item):
    return records.digest(dict(benchmark_id=recipe['benchmark_id'], task_id=item['task_id'],
        arm=item['arm'], messages=item['messages'], prefix=item['prefix']))


def verify_prefix(row, states, recipe, prompts, tokenized):
    key = (row['task_id'], row['arm'])
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
        verify_prefix(row, states, recipe, prompts, tokenized)
        updates[key] = row
    states.update(updates)


def collect(root, recipe, prompts, tokenized):
    by_id = {p['task_id']: p for p in prompts}; states = {}; batches = []
    seed=Path(__file__).with_name('continuation.json')
    from bridge import sha
    if sha(seed)!=recipe['continuation_sha256']: raise ValueError('Continuation identity mismatch')
    for row in json.loads(seed.read_text())['prefixes']:
        task,arm,tokens=row['task_id'],row['arm'],row['prefix']
        if task not in recipe['eligible'][arm] or (task,arm) in states: raise ValueError('Invalid seeded pair')
        states[task,arm]=dict(completion_token_ids=tokens,generated_tokens=len(tokens),finish_reason='checkpoint')
    for expected, path in enumerate(sorted((Path(root) / 'batches').glob('batch-*.json')), 1):
        batch = json.loads(path.read_text())
        if batch['sequence'] != expected or path.name != 'batch-%06d.json' % expected:
            raise RuntimeError('Missing or reordered journal sequence')
        apply_batch(batch, states, recipe, by_id, tokenized)
        batches.append(batch)
    merge_terminals(root, states, recipe, by_id, tokenized)
    return states, batches


def completed(states):
    return {k: r for k, r in states.items() if r['finish_reason'] in ('eos', 'token_limit')}

def publish_terminal(root, row):
    records.validate_record(row)
    directory = Path(root)/'terminals'
    directory.mkdir(exist_ok=True)
    path = directory/f"{row['arm']}-{row['task_id']}.json"
    value = {'row': row, 'sha256': digest(row)}
    with tempfile.NamedTemporaryFile(dir=directory, delete=False, mode='w') as stream:
        try:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(stream.name, path)
            fd = os.open(directory, os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        finally:
            os.unlink(stream.name)


def same_terminal(a,b):
    return all(a[k]==b[k] for k in ('request_id','prompt_sha256','messages_sha256','arm','task_id',
               'benchmark_id','batch_id','finish_reason','completion_token_ids','raw_text','text'))


def load_terminal(path):
    value=json.loads(path.read_text())
    row=value['row']
    if (digest(row)!=value['sha256'] or path.name!=f"{row['arm']}-{row['task_id']}.json"
            or row['finish_reason'] not in ('eos','token_limit')):
        raise ValueError('Invalid terminal receipt: '+str(path))
    records.validate_record(row)
    return row


def merge_terminals(root,states,recipe,prompts,tokenized):
    for path in sorted((Path(root)/'terminals').glob('*.json')):
        row=load_terminal(path)
        key=row['task_id'],row['arm']
        if key[0] not in recipe['eligible'].get(key[1],[]) or row['benchmark_id']!=recipe['benchmark_id']:
            raise ValueError('Terminal receipt outside frozen task inventory')
        old=states.get(key)
        if old and old['finish_reason'] in ('eos','token_limit'):
            if not same_terminal(old,row): raise ValueError('Batch and early terminal receipt disagree')
        else:
            verify_prefix(row,states,recipe,prompts,tokenized)
            states[key]=row
