"""Freeze answer-free prompts and real prefixes; never submit an allocation."""
import json
from pathlib import Path
if __package__:
    from .runtime import NAME, bridge, ready, sha, digest
else:
    from runtime import NAME, bridge, ready, sha, digest


def reference(task, arm, state, controller):
    controller.validate_state(task['visible'], state)
    messages = controller.messages_for(task['visible'], [])
    if state['attempts']:
        candidate = state['attempts'][0]
        record = candidate['response']
        ids = record['completion_token_ids']
    elif state['pending'] and state['pending']['segments']:
        candidate = state['pending']
        record = candidate['segments'][0]
        ids = candidate['prefix']
    else:
        return None
    if candidate['messages'] != messages or len(ids) < 512 or record['eos_token_id'] in ids[:512]:
        return None
    if (record['prompt_tokens'] + 1024 > 18432 or
            8 * (record['prompt_tokens'] + 512) > 44448 or
            8 * (record['prompt_tokens'] + 512 + 256) > 60000):
        return None
    return {'task_id': task['task_id'], 'arm': arm, 'messages': messages,
            'prefix': ids[:512], 'prompt_tokens': record['prompt_tokens'],
            'source_state_sha256': digest(state)}


def stage(folder):
    source, original, m, v = bridge(folder)
    if (folder / 'READY.json').exists():
        ready(folder)
        print('CACHE PILOT ALREADY STAGED; no job submitted')
        return
    tasks = m['protocol'].rows(source / 'tasks.jsonl')
    lengths = json.loads((source / 'template-check.json').read_text())['initial_prompt_lengths']
    median = sorted(lengths.values())[(len(lengths) - 1) // 2]
    variant = v['runtime'].variant_recipe(original, 32)
    pairs = []
    for task in tasks:
        refs = {}
        for arm in ('finetuned', 'base'):
            path = source.parent / 'runs' / source.name / 'batch32' / task['task_id'] / arm / 'state.json'
            if not path.exists():
                break
            state = json.loads(path.read_text())
            if (state.get('benchmark_id'), state.get('task_id'), state.get('arm')) != (
                    variant['benchmark_id'], task['task_id'], arm):
                raise RuntimeError('Reference state identity changed')
            ref = reference(task, arm, state, v['controller'])
            if ref is None:
                break
            if ref['prompt_tokens'] != lengths[task['task_id']]:
                raise RuntimeError('Reference prompt length changed')
            refs[arm] = ref
        if len(refs) == 2:
            if refs['base']['messages'] != refs['finetuned']['messages']:
                raise RuntimeError('Different arm prompts')
            pairs.append((abs(lengths[task['task_id']] - median), task['task_id'], refs))
    if not pairs:
        raise RuntimeError('No common eligible 512-token prefix; retain V2 progress and inspect it')
    _, _, refs = min(pairs, key=lambda value: (value[0], value[1]))
    # Fixed mixed-length control prompts. No query outputs or source labels copied.
    controls = [{'task_id': task['task_id'],
                 'messages': v['controller'].messages_for(task['visible'], []), 'prefix': []}
                for task in tasks[:2]]
    io = m['result_io']
    io.atomic_json(folder / 'references.json', refs)
    io.atomic_json(folder / 'controls.json', controls)
    config = json.loads((source / 'inference-config.json').read_text())
    config['output_dir'] = str(folder.parent / 'runs' / NAME / 'loader')
    io.atomic_json(folder / 'inference-config.json', config)
    m['configuration'].schema_preflight(folder / 'inference-config.json')
    recipe = {'name': NAME, 'engineering_only': True, 'source_recipe_sha256': sha(source / 'recipe.json'),
              'checkpoint': original['checkpoint'], 'eos_token_id': original['eos_token_id'],
              'pad_token_id': original['pad_token_id'], 'enable_thinking': False, 'context': 18432,
              'batch_size': 8, 'segment_tokens': 128, 'segments': 2,
              'control_segment_tokens': 16, 'control_segments': 2,
              'references_sha256': sha(folder / 'references.json'), 'controls_sha256': sha(folder / 'controls.json'),
              'methods': ['continuous', 'replay', 'retained'], 'seed': 42,
              'methodology': 'Forced-length cache engineering pilot; same per-arm prompt and prefix; no execution, refinement, scoring or scientific prediction reuse',
              'placement': 'validated native layer placement 16/16/16; pipeline and tensor parallelism are not tested'}
    recipe['benchmark_id'] = digest(recipe)
    io.atomic_json(folder / 'recipe.json', recipe)
    io.atomic_json(folder / 'READY.json', {'files': {name: sha(folder / name) for name in
                   ('recipe.json', 'references.json', 'controls.json', 'inference-config.json')}})
    print('CACHE PILOT STAGED: both models; continuous vs replay vs retained cache; separate results; no job submitted')


if __name__ == '__main__':
    stage(Path(__file__).resolve().parent)
