"""Reuse verified answer-free v1 inputs; publish a distinct chunked policy recipe."""
import json
from pathlib import Path
if __package__:
    from .runtime import NAME,bridge,ready,sha
    from . import controller,policy
else:
    from runtime import NAME,bridge,ready,sha
    import controller,policy

def verified_predecessor(folder):
    previous=folder.parent/'arc-flash-python-nothink-v1'
    expected=json.loads((folder/'nothink-v1-package-manifest.json').read_text())
    if json.loads((previous/'package-manifest.json').read_text())!=expected:raise RuntimeError('v1 identity changed')
    for name,h in expected['files_sha256'].items():
        if Path(name).name!=name or sha(previous/name)!=h:raise RuntimeError('v1 package changed: '+name)
    proof=json.loads((previous/'READY.json').read_text())
    for name in ('recipe.json','tasks.jsonl','inference-config.json','template-check.json'):
        if proof['files'].get(name)!=sha(previous/name):raise RuntimeError('Restage changed v1 input: '+name)
    return previous

def stage(folder):
    _,_,original,m=bridge(folder)
    if (folder/'READY.json').exists():ready(folder);print('ADAPTIVE CODE-ONLY EXPERIMENT ALREADY STAGED; no job submitted');return
    previous=verified_predecessor(folder);old=json.loads((previous/'recipe.json').read_text())
    if old['source_benchmark_id']!=original['benchmark_id'] or old['enable_thinking'] is not False:raise RuntimeError('Wrong predecessor protocol')
    if old['max_candidates']!=4 or old['max_tokens_per_candidate']!=8192 or old['max_tokens_per_task_arm']!=32768:raise RuntimeError('Candidate budget changed')
    tasks=m['protocol'].rows(previous/'tasks.jsonl')
    if len(tasks)!=128 or [r['task_id'] for r in tasks]!=old['task_ids'] or len(set(old['task_ids']))!=128:raise RuntimeError('Task order/identity changed')
    for task in tasks:controller.initial_state(task['visible'])
    for name in ('tasks.jsonl','template-check.json'):(folder/name).write_bytes((previous/name).read_bytes())
    cfg=json.loads((previous/'inference-config.json').read_text());cfg['output_dir']=str(folder.parent/'runs'/NAME/'loader')
    m['result_io'].atomic_json(folder/'inference-config.json',cfg);m['configuration'].schema_preflight(folder/'inference-config.json')
    recipe=dict(old,name=NAME,controller_sha256=sha(folder/'controller.py'),policy=policy.POLICY,
        policy_sha256=sha(folder/'policy.py'),generation_sha256=sha(folder/'batched.py'),
        predecessor_manifest_sha256=sha(previous/'package-manifest.json'),
        execution_change='Global length-sorted same-arm wavefront; deterministic512-token prefix checkpoints; exact prefix replay and rebatching. Not output-equivalent to unchunked generation.',
        serial_debug_task_ids=old['task_ids'][:4],batched_debug_policy='first32tasks plus global longest initial prompt',
        capacity_evidence='job8026684 user-reported short-probe measurements; admission bounds are estimates, not GPU fit guarantees')
    m['result_io'].atomic_json(folder/'recipe.json',recipe)
    m['result_io'].atomic_json(folder/'READY.json',{'files':{n:sha(folder/n) for n in ('recipe.json','tasks.jsonl','inference-config.json','template-check.json')}})
    print('ADAPTIVE CODE-ONLY REFINEMENT STAGED: cap32; token/padding-aware groups; 512-token checkpoints; four candidates and original task budgets preserved. v1 results untouched. No job submitted.')
if __name__=='__main__':stage(Path(__file__).resolve().parent)
