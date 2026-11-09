"""Verify the frozen pilot and reuse completed responses without reading answers."""
import json
import os
from pathlib import Path
from pilot_common import NAME, dependencies, digest


def compatible(a, b):
    return all(a.get(k) == b.get(k) for k in ('benchmark_id','task_id','arm','text',
        'finish_reason','prompt_tokens','generated_tokens','token_budget','adapter_layers_checked'))


def stage(folder):
    native, recipe, m, receipt = dependencies(folder, require_ready=False)
    ready = folder/'READY.json'
    if ready.exists():dependencies(folder)
    else:m['common'].atomic_json(ready, receipt)
    output = folder.parent/'runs'/NAME
    source = folder.parent/'runs/arc-flash-eval138-debug-v1'
    imported = []
    with m['common'].output_lock(output):
        recipe_path = output/'benchmark-recipe.json'
        if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
            raise RuntimeError('Existing pilot predictions belong to a different benchmark')
        existing = m['protocol'].load_predictions(output/'predictions.jsonl',set(receipt['target_task_ids']))
        if any(r.get('benchmark_id') != recipe['benchmark_id'] for r in existing.values()):
            raise RuntimeError('Pilot prediction provenance changed')
        if source.exists():
            with m['common'].output_lock(source):
                source_recipe = source/'benchmark-recipe.json'
                if source_recipe.exists():
                    if json.loads(source_recipe.read_text()) != recipe:
                        raise RuntimeError('Previous debug has a different scientific recipe')
                    records = m['protocol'].load_predictions(source/'predictions.jsonl',set(receipt['cohorts']['ordered_prefix']))
                    if any(r.get('benchmark_id') != recipe['benchmark_id'] for r in records.values()):
                        raise RuntimeError('Previous debug prediction provenance changed')
                    for key in set(records)&set(existing):
                        if not compatible(existing[key],records[key]):
                            raise RuntimeError('Conflicting completed responses; refusing to choose an answer')
                    for task in receipt['target_task_ids']:
                        for arm in m['protocol'].ARMS:
                            key = task,arm
                            if key in records and key not in existing:
                                m['protocol'].append_record(output/'predictions.jsonl',records[key])
                                imported.append(list(key))
                    m['common'].atomic_json(output/'imported-debug.json',dict(
                        source=str(source),source_sha256=digest(source/'predictions.jsonl') if (source/'predictions.jsonl').exists() else None,
                        imported=imported,all_completed_results_retained_regardless_of_correctness=True))
        m['common'].atomic_json(recipe_path,recipe)
        m['common'].atomic_json(output/'selection.json',json.loads((folder/'selection.json').read_text()))
    print('VALIDATION PILOT STAGED: 10 fixed-order tasks + 2 prior-API successes; '
          'both models; thinking off; unchanged native serial generation',flush=True)
    print('Previously completed responses reused: '+str(len(imported)),flush=True)
    print('Two-hour allocation cap; completion not guaranteed. No job submitted or modified.',flush=True)


if __name__=='__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    stage(Path(__file__).resolve().parent)
