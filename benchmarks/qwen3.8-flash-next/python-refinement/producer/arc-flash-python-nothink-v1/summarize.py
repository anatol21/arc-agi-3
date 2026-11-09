"""CPU-only query scorer; query labels never enter task controllers."""
import importlib
import json
from pathlib import Path
if __package__:
    from .runtime import NAME,ready,variant_recipe,sha
    from . import controller,scheduler
else:
    from runtime import NAME,ready,variant_recipe,sha
    import controller,scheduler

def collect(folder,recipe,tasks):
    output=folder.parent/'runs'/NAME/('batch'+str(recipe['batch_size']));states={};frontier={}
    cache=scheduler.load_cache(output,recipe)
    for task in tasks:
        for arm in recipe['arm_order']:
            p=output/task['task_id']/arm/'state.json'
            if not p.exists():continue
            state=json.loads(p.read_text())
            if (state.get('benchmark_id'),state.get('task_id'),state.get('arm'))!=(recipe['benchmark_id'],task['task_id'],arm):raise RuntimeError('Wrong scored state')
            controller.validate_state(task['visible'],state)
            attempts=list(state['attempts'])
            if state['pending']:attempts.append({'segments':state['pending']['segments']})
            for a in attempts:
                for r in a['segments']:
                    if cache.get(r['request_id'])!=r:raise RuntimeError('State differs from durable batch response')
            states[task['task_id'],arm]=state;frontier[str(p.relative_to(output))]=sha(p)
    return states,frontier

def score(folder):
    benchmark,base,m=ready(folder);io=m['result_io'];root=folder.parent/'runs'/NAME
    tasks=m['protocol'].rows(folder/'tasks.jsonl')
    scorer=importlib.import_module('score')
    if Path(scorer.__file__).resolve()!=(folder.parent/'arc-flash-program-agent-v4/score.py').resolve():raise RuntimeError('Wrong strict scoring dependency')
    # Only this separate process opens query labels. No score or test outputs are
    # supplied to the generation process or candidate-selection controller.
    answers=m['protocol'].rows(benchmark/'answers.scorer-only.jsonl')
    with io.results_lock(root):
        for size in (1,2,4,8,16,32,64):
            output=root/('batch'+str(size))
            if not (output/'recipe.json').exists():continue
            recipe=variant_recipe(base,size)
            if json.loads((output/'recipe.json').read_text())!=recipe:raise RuntimeError('Variant recipe changed')
            states,frontier=collect(folder,recipe,tasks);report=scorer.summarize(answers,states,recipe)
            batches=[json.loads(p.read_text()) for p in (output/'batches').glob('*.json')]
            seconds=sum(b['seconds'] for b in batches);new=sum(b['new_generated_tokens'] for b in batches)
            report.update(thinking=False,batch_size_cap=size,state_frontier=frontier,scored_at=io.utcnow(),
                batch_generation={'calls':len(batches),'wall_seconds':seconds,'new_tokens':new,'tokens_per_second':new/seconds if seconds else None,
                'actual_batch_sizes':sorted({b['actual_batch_size'] for b in batches}),
                'cost_note':'Batch time counted once. Row generation_seconds uses equal shares, not per-response latency. Includes durable generated responses awaiting controller delivery.'})
            serial=root/'batch1'/'report.json'
            if size!=1 and serial.exists():
                control=json.loads(serial.read_text())
                first={(r['task_id'],r['arm']):r for r in control['per_task']['selected']}
                current={(r['task_id'],r['arm']):r for r in report['per_task']['selected']}
                comparison={}
                for arm in base['arm_order']:
                    keys=[k for k in first.keys() & current.keys() if k[1]==arm]
                    comparison[arm]={'paired_tasks':len(keys),'serial_correct':sum(first[k]['task_correct'] for k in keys),
                        'batched_correct':sum(current[k]['task_correct'] for k in keys),
                        'serial_only_correct':sum(first[k]['task_correct'] and not current[k]['task_correct'] for k in keys),
                        'batched_only_correct':sum(current[k]['task_correct'] and not first[k]['task_correct'] for k in keys)}
                report['serial_control_comparison']=comparison
            report['report_id']=controller.message_digest(report);io.atomic_json(output/'report.json',report)
            io.atomic_json(output/'summary.json',{k:v for k,v in report.items() if k!='per_task'})
            print('CODE_ONLY_SCORE '+json.dumps({'batch_size':size,'complete':report['complete'],'selected':report['selected']['arms'],'generation':report['batch_generation']}),flush=True)
if __name__=='__main__':score(Path(__file__).resolve().parent)
