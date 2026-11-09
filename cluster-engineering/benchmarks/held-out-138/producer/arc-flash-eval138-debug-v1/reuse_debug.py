"""Explicit CPU-only import of reviewed debug predictions into the full run."""
import json
from pathlib import Path
from debug_common import NAME, NATIVE, dependencies, digest
from score_debug import score_run


def compatible(old, new):
    keys = ('benchmark_id', 'task_id', 'arm', 'text', 'finish_reason', 'prompt_tokens',
            'generated_tokens', 'token_budget', 'adapter_layers_checked')
    return all(old.get(key) == new.get(key) for key in keys)


def reuse(folder):
    native, recipe, m, receipt = dependencies(folder)
    if score_run(folder)['debug']['phase'] != 'passed':
        raise RuntimeError('Review a completed, passing debug before reuse')
    source = folder.parent/'runs'/NAME
    target = folder.parent/'runs'/NATIVE
    with m['common'].output_lock(source), m['common'].output_lock(target):
        records = m['protocol'].load_predictions(source/'predictions.jsonl', set(receipt['target_task_ids']))
        recipe_path = target/'benchmark-recipe.json'
        if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
            raise RuntimeError('Full output belongs to a different scientific recipe')
        if (target/'predictions.jsonl').exists() and not recipe_path.exists():
            raise RuntimeError('Existing full predictions lack their original recipe')
        prompts = m['protocol'].rows(native/'prompts.jsonl')
        existing = m['protocol'].load_predictions(target/'predictions.jsonl', {p['task_id'] for p in prompts})
        if any(r.get('benchmark_id') != recipe['benchmark_id'] for r in existing.values()):
            raise RuntimeError('Existing full prediction provenance differs')
        if len(records) != 6 or any(r.get('benchmark_id') != recipe['benchmark_id'] for r in records.values()):
            raise RuntimeError('Debug records are incomplete or have changed provenance')
        for key in set(existing) & set(records):
            if not compatible(existing[key], records[key]):
                raise RuntimeError('Conflicting predictions; refusing to choose an answer: ' + repr(key))
        m['common'].atomic_json(recipe_path, recipe)
        imported = []
        for task in receipt['target_task_ids']:
            for arm in m['protocol'].ARMS:
                key = (task, arm)
                if key not in existing:
                    m['protocol'].append_record(target/'predictions.jsonl', records[key])
                    imported.append(list(key))
        m['common'].atomic_json(source/'reuse-receipt.json', dict(
            benchmark_id=recipe['benchmark_id'], imported=imported,
            source_predictions_sha256=digest(source/'predictions.jsonl'),
            full_predictions_sha256=digest(target/'predictions.jsonl'), jobs_modified=False))
    print('DEBUG PREDICTIONS REUSED: ' + str(len(imported)) + '; no job submitted or modified', flush=True)


if __name__ == '__main__':
    reuse(Path(__file__).resolve().parent)
