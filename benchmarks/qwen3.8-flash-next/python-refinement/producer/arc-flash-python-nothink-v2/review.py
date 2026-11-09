"""Review adaptive admission and representative completion; never submit a job."""
import argparse
import json
from pathlib import Path
if __package__:
    from .runtime import NAME,ready,variant_recipe
    from .summarize import collect
    from .run import debug_targets
    from . import policy
else:
    from runtime import NAME,ready,variant_recipe
    from summarize import collect
    from run import debug_targets
    import policy

def review(folder,size):
    _,base,m=ready(folder);root=folder.parent/'runs'/NAME;tasks=m['protocol'].rows(folder/'tasks.jsonl');checks={};variants={}
    lengths=json.loads((folder/'template-check.json').read_text())['initial_prompt_lengths']
    model_path=root/'model-reload.json';sandbox_path=root/'sandbox-preflight.json'
    if model_path.exists():
        model=json.loads(model_path.read_text());checks['frozen_model_and_adapter_verified']=model['requires_grad_parameters']==0 and model['optimizer_updates']==0 and bool(model['adapter'])
    else:checks['frozen_model_and_adapter_verified']=False
    checks['allocation_sandbox_preflight_passed']=sandbox_path.exists() and all(json.loads(sandbox_path.read_text()).get(k) is True for k in ('identity_execution','stdout_capture','workspace_hidden','network_probe_blocked','environment_allowlist','readonly_filesystem_probe','gpu_devices_hidden'))
    for n in sorted({1,size}):
        r=variant_recipe(base,n);wanted=tasks[:4] if n==1 else debug_targets(tasks,lengths)
        states,_=collect(folder,r,tasks);keys=[(t['task_id'],arm) for t in wanted for arm in base['arm_order']]
        checks['cap'+str(n)+'_representative_tasks_complete']=all(k in states and states[k]['complete'] for k in keys)
        checks['cap'+str(n)+'_each_arm_emitted_and_executed_code']=all(any(k[1]==arm and any('demonstration_execution' in a for a in states.get(k,{}).get('attempts',[])) for k in keys) for arm in base['arm_order'])
        batches=[json.loads(p.read_text()) for p in (root/('batch'+str(n))/'batches').glob('*.json')]
        checks['cap'+str(n)+'_each_arm_generated']=all(any(b['arm']==arm for b in batches) for arm in base['arm_order'])
        admitted=True
        for b in batches:
            if b['actual_batch_size']!=len(b['rows']) or any(row['arm']!=b['arm'] for row in b['rows']):admitted=False;continue
            prepared=[{'request':{'task_id':row['task_id']},'ids':[0]*row['generation_input_tokens'],
                       'remaining':row['token_budget']-row['replayed_prefix_tokens']} for row in b['rows']]
            if not policy.accepts(prepared,n) or b['admission']!=policy.admission_metrics(prepared):admitted=False
        checks['cap'+str(n)+'_all_batches_admitted']=bool(batches) and admitted
        if n>1:checks['cap'+str(n)+'_multirow_generation_each_arm']=all(any(b['arm']==arm and b['actual_batch_size']>1 for b in batches) for arm in base['arm_order'])
        seconds=sum(b['seconds'] for b in batches);tokens=sum(b['new_generated_tokens'] for b in batches)
        variants[str(n)]={'target_tasks_per_arm':len(wanted),'complete_task_arms':sum(k in states and states[k]['complete'] for k in keys),
            'batch_calls':len(batches),'new_tokens':tokens,'generation_seconds':seconds,'new_tokens_per_second':tokens/seconds if seconds else None,
            'actual_batch_sizes':sorted({b['actual_batch_size'] for b in batches}),
            'replayed_prefix_tokens':sum(b['replayed_prefix_tokens'] for b in batches),
            'padded_prefill_tokens':sum(b['padded_prefill_tokens'] for b in batches),
            'checkpoint_segments':sum(row['finish_reason']=='checkpoint' for b in batches for row in b['rows']),
            'peak_allocated_GiB':{str(i):max((v['peak_allocated_bytes']/2**30 for b in batches for v in b['memory'] if v['device']==i),default=None) for i in range(3)}}
    result={'phase':'passed' if all(checks.values()) else 'incomplete','checks':checks,'batch_size_cap':size,'variants':variants,'policy':policy.POLICY,
        'limits':'Adaptive/chunked protocol differs numerically from v1. A pass validates representative infrastructure, not worst-case GPU memory or full accuracy. Review replay overhead, code/results and runtime. No job submitted.'}
    m['result_io'].atomic_json(root/('debug-review-cap'+str(size)+'.json'),result)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,choices=(1,2,4,8,16,32),default=32);p.add_argument('--require-pass',action='store_true');a=p.parse_args();r=review(Path(__file__).resolve().parent,a.batch_size);print('ADAPTIVE_DEBUG_REVIEW '+json.dumps(r))
    if a.require_pass and r['phase']!='passed':raise SystemExit(1)
