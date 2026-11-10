"""Export verified public-task predictions from the BF16 sampling campaign, without scoring inputs."""
import hashlib
import json
from pathlib import Path
import runpy
import sys


def export(workspace):
    package = workspace / 'arc-bf16-sampling-validation-v1'
    sys.path.insert(0, str(package))
    from sample_cont_stage import read_plan, verified_pilot, arm_id, arm_prompts
    from continuation_score import checked_states
    from sampling_score import score_single
    plan = read_plan(package)
    _, pilot = verified_pilot(workspace)
    root = workspace / 'runs' / package.name / plan['benchmark_id']
    native = workspace / 'arc-flash-eval138-v1'
    protocol = runpy.run_path(str(native / 'protocol.py'))
    oracle = native / 'answers.scorer-only.jsonl'
    assert hashlib.sha256(oracle.read_bytes()).hexdigest() == json.loads((native / 'benchmark-recipe.json').read_bytes())['answers_sha256']
    answers = protocol['rows'](oracle)
    assert len(answers) == 138
    for slug in plan['arm_order']:
        continuation = checked_states(root / 'arms' / slug, arm_prompts(plan, slug), arm_id(plan, slug), plan['eos_token_id'])
        assert not pilot[slug].keys() & continuation.keys()
        states = {**pilot[slug], **continuation}
        result = score_single(answers, states, protocol['score'])
        for score in sorted(result['tasks'], key=lambda r:r['task_id']):
            t = score['task_id']
            s = states[t]
            assert s['sampling'] == plan['sampling'][slug]
            row = dict(strategy=slug, task_id=t, stage='pilot' if t in pilot[slug] else 'continuation',
                       sampling=s['sampling'], text=s['text'], response_sha256=score['response_sha256'],
                       task_correct=score['task_correct'], grid_correct=score['grid_correct'],
                       format_error=score['format_error'], finish_reason=s['finish_reason'],
                       generated_tokens=s['generated_tokens'], continuation_id=plan['benchmark_id'])
            print(json.dumps(row,ensure_ascii=False))


if __name__ == '__main__':
    export(Path(sys.argv[1]).resolve())
