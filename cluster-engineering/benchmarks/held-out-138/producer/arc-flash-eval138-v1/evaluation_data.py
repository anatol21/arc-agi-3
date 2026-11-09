"""Answer-free reserved-task prompts, with the original direct-answer format."""
import json


def render(visible, instruction):
    if not instruction.endswith('\nTask:\n'):
        raise ValueError('Instruction must retain its exact Task delimiter')
    public = {'train': [{'input': p['input'], 'output': p['output']} for p in visible['train']],
              'test': [{'input': p['input']} for p in visible['test']]}
    return instruction + json.dumps(public, separators=(',', ':'))


def split_evaluation(path, manifest, instruction):
    # This module never opens scoring/answers.jsonl or imports a scorer.
    if __package__:
        from experiments.cluster_benchmark.protocol import grid
    else:
        from protocol import grid
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [t['task_id'] for t in tasks]
    expected = manifest['task_ids']['evaluation']
    if len(ids) != 138 or len(set(ids)) != 138 or ids != expected:
        raise ValueError('Reserved evaluation IDs/order differ from the fixed138-task set')
    if set(ids) & (set(manifest['task_ids']['train']) | set(manifest['task_ids']['dev'])):
        raise ValueError('Reserved tasks overlap training/development IDs')
    prompts = []
    for task in tasks:
        if set(task) != {'task_id', 'train', 'test'} or not task['train'] or not task['test']:
            raise ValueError('Unexpected reserved task fields or empty examples')
        for pair in task['train']:
            if set(pair) != {'input', 'output'}:
                raise ValueError('Unexpected demonstration fields')
            grid(pair['input']); grid(pair['output'])
        for pair in task['test']:
            if set(pair) != {'input'}:
                raise ValueError('Query answer or metadata leaked into a prompt')
            grid(pair['input'])
        prompts.append({'task_id': task['task_id'], 'messages': [
            {'role': 'user', 'content': render(task, instruction)}],
            'query_count': len(task['test'])})
    if sum(p['query_count'] for p in prompts) != 185:
        raise ValueError('Expected185 reserved query grids')
    return prompts
