"""Stage a verified snapshot of the paused parent's unfinished pairs, saved prefixes and batch history."""
import json
import os
from pathlib import Path
import shutil
import sys

FOLDER=Path(__file__).resolve().parent
WORK=FOLDER.parent
OLD=WORK/'arc-flash-eval138-dashboard-v4'
RUN_ID='8146a5b796f8482bab9a5e245711fdf2'
RUN=WORK/'cluster-ui'/RUN_ID


def main():
    sys.path[:0]=[str(OLD),str(RUN)]
    import bridge
    import journal
    import runtime
    from sdk import lock,read_json
    with lock(RUN/'.control.lock',blocking=True):
        m=runtime.load_run(RUN)
        w=m['waves'][-1]
        if (m['state']!='needs_attention' or w['job_id']!='8470445' or not w.get('handled') or
                runtime.Scheduler(WORK,m['profile']).status(w['job_id'])['state']!='TERMINAL'):
            raise RuntimeError('Parent allocation is not quiescent')
        runtime.validate_result(RUN,m,w,0,read_json(RUN/'workers/0/result.json'))
        parent,_,prompts,_,old=bridge.dependencies(OLD)
        if m['config']['fingerprints']['recipe']!=bridge.sha(OLD/'recipe.json'):
            raise ValueError('Parent recipe mismatch')
        tokenized=read_json(OLD/'tokenized-prompts.json')
        states,batches=journal.collect(RUN/'native',old,prompts,tokenized)
        done=journal.completed(states)
        eligible={arm:[t for t in ids if (t,arm) not in done] for arm,ids in old['eligible'].items()}
        if len(done)!=19 or list(map(len,[eligible['finetuned'],eligible['base']]))!=[48,43]:
            raise RuntimeError('Unexpected inventory')
        if (FOLDER/'READY.json').exists():
            ready=read_json(FOLDER/'READY.json')
            if any(bridge.sha(FOLDER/name)!=sha for name,sha in ready['files_sha256'].items()):
                raise ValueError('Existing continuation modified')
            return
        history=[dict(hardware=b['hardware'],arm=b['arm'],length_band=b['length_band'],
                      size=b['actual_batch_size'],failed=False) for b in batches]
        for p in sorted((RUN/'native/oom-attempts').glob('*.json')):
            d=read_json(p)
            if d['benchmark_id']!=old['benchmark_id'] or d['sha256']!=bridge.digest({k:v for k,v in d.items() if k!='sha256'}):
                raise ValueError('Invalid OOM receipt')
            history.append(dict(hardware=d['hardware'],arm=d['arm'],length_band=d['length_band'],
                                size=len(d['task_ids']),failed=True))
        prefixes=[dict(task_id=t,arm=a,prefix=r['completion_token_ids']) for (t,a),r in sorted(states.items()) if (t,a) not in done]
        proof=read_json(OLD/'eligibility.json')
        proof['eligible']=eligible
        proof['excluded_finished_pairs']+= [dict(task_id=t,arm=a) for t,a in sorted(done)]
        proof.update(unfinished_prefixes_imported=True,prior_responses_imported=False)
        carry=dict(parent_run=RUN_ID,parent_job=w['job_id'],parent_benchmark=old['benchmark_id'],
                   parent_checkpoint=bridge.sha(RUN/'workers/0/result.json'),prefixes=prefixes,admission_history=history,
                   prior_reserved_gpu_hours=sum(x['reserved_gpu_hours'] for x in m['waves'] if x['status']=='submitted'))
        if carry['prior_reserved_gpu_hours']!=4.5 or m['config']['budget']['gpu_hours']!=6:
            raise ValueError('Unexpected shared reservation ledger')
        bridge.atomic(FOLDER/'continuation.json',carry)
        bridge.atomic(FOLDER/'eligibility.json',proof)
        for name in ('tokenized-prompts.json','inference-config.json'):
            shutil.copyfile(OLD/name,FOLDER/name)
        # Read the child policy constants without importing the already loaded parent policy module.
        import runpy
        policy=runpy.run_path(str(FOLDER/'policy.py'))['POLICY']
        recipe=dict(old,name=FOLDER.name,eligible=eligible,policy=policy,eligibility_sha256=bridge.sha(FOLDER/'eligibility.json'),
                    continuation_sha256=bridge.sha(FOLDER/'continuation.json'),
                    schedule='prefer 18 or more; allow smaller batches under shape or OOM constraints',
                    output_reuse='exclude every terminal pair; restore verified unfinished parent tokens')
        recipe.pop('benchmark_id')
        recipe['benchmark_id']=bridge.digest(recipe)
        bridge.atomic(FOLDER/'recipe.json',recipe)
        files=('tokenized-prompts.json','inference-config.json','continuation.json','eligibility.json','recipe.json')
        bridge.atomic(FOLDER/'READY.json',dict(own_manifest_sha256=bridge.sha(FOLDER/'package-manifest.json'),
            parent_ready_sha256=bridge.sha(parent/'READY.json'),files_sha256={n:bridge.sha(FOLDER/n) for n in files}))
        print('STAGED',len(prefixes),'unfinished prefixes;',len(done),'excluded terminals',flush=True)


if __name__=='__main__':
    main()
