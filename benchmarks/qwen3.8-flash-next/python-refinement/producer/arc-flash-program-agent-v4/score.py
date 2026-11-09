"""Separate CPU scorer: query labels never enter generation or program execution."""
import json
from pathlib import Path

if __package__:
    from .agent import validate_state, message_digest
    from .setup_runtime import NAME, ready, atomic_json, results_lock, utcnow
    from .protocol import rows, score as strict_score
else:
    from agent import validate_state, message_digest
    from setup_runtime import NAME, ready, atomic_json, results_lock, utcnow
    from protocol import rows, score as strict_score


def collect(folder, recipe):
    """Atomic state files give a consistent per-task frontier during manual scoring."""
    output=folder.parent/'runs'/NAME
    states={}; frontier={}
    for task in rows(folder/'prompts.jsonl'):
        visible=json.loads(task['messages'][0]['content'].split('\nTask:\n',1)[1])
        for arm in ('finetuned','base'):
            path=output/task['task_id']/arm/'state.json'
            if not path.exists():continue
            data=path.read_bytes();state=json.loads(data)
            if (state.get('benchmark_id')!=recipe['benchmark_id'] or state.get('task_id')!=task['task_id']
                or state.get('arm')!=arm):raise ValueError('Wrong saved state provenance')
            validate_state(visible,state)
            key=(task['task_id'],arm);states[key]=state
            frontier[str(path.relative_to(output))]=__import__('hashlib').sha256(data).hexdigest()
    return states,frontier


def summarize(answers,states,recipe):
    predictions={label:{} for label in ('first','selected')}
    execution_metrics={arm:{'candidates_with_executable_demo':0,'executable_demonstration_cases':0,
                            'valid_grid_demonstration_cases':0,'first_query_executable_tasks':0}
                       for arm in ('finetuned','base')}
    costs={arm:{'new_generated_tokens':0,'replayed_prefix_tokens':0,'generation_seconds':0,
                'execution_seconds':0,'completed_candidates':0,'programs_executed':0,
                'first_all_demonstrations_matched':0,'selected_all_demonstrations_matched':0,
                'selected_query_executable_tasks':0,'pending_responses':0} for arm in ('finetuned','base')}
    for key,state in states.items():
        cost=costs[key[1]];cost['new_generated_tokens']+=state['total_generated_tokens']
        metrics=execution_metrics[key[1]]
        for attempt in state['attempts']:
            feedback=attempt.get('feedback',{})
            metrics['candidates_with_executable_demo']+=feedback.get('executed_cases',0)>0
            metrics['executable_demonstration_cases']+=feedback.get('executed_cases',0)
            metrics['valid_grid_demonstration_cases']+=sum(c['valid_grid'] for c in feedback.get('cases',[]))
        segments=[r for a in state['attempts'] for r in a['segments']]
        if state['pending']:
            segments+=state['pending']['segments'];cost['pending_responses']+=1
        cost['generation_seconds']+=sum(r['seconds'] for r in segments)
        cost['replayed_prefix_tokens']+=sum(r['replayed_prefix_tokens'] for r in segments)
        cost['completed_candidates']+=len(state['attempts'])
        cost['programs_executed']+=sum('demonstration_execution' in a for a in state['attempts'])
        executions=[a['demonstration_execution'] for a in state['attempts'] if 'demonstration_execution' in a]
        if 'first_query_execution' in state:executions.append(state['first_query_execution'])
        if 'selected_query_execution' in state and state.get('selected_candidate')!=0:executions.append(state['selected_query_execution'])
        cost['execution_seconds']+=sum(c.get('seconds',0) for e in executions for c in e['cases'])
        if not state['complete']:continue
        metrics['first_query_executable_tasks']+=all(c['executable'] for c in state['first_query_execution']['cases'])
        first=state['attempts'][0] if state['attempts'] else None
        selected=state['attempts'][state['selected_candidate']] if state['selected_candidate'] is not None else None
        for label,attempt in [('first',first),('selected',selected)]:
            if attempt and attempt['feedback']['passed']==attempt['feedback']['total']:
                cost[label+'_all_demonstrations_matched']+=1
            predictions[label][key]={'text':json.dumps(state[label+'_output_json']),
                                     'finish_reason':attempt['response']['finish_reason'] if attempt else 'invalid'}
        cost['selected_query_executable_tasks']+=all(c['executable'] for c in state['selected_query_execution']['cases'])
    reports={};details={}
    for label in predictions:reports[label],details[label]=strict_score(answers,predictions[label],thinking=False)
    by_label={label:{(r['task_id'],r['arm']):r for r in detail} for label,detail in details.items()}
    refinement={}
    for arm in costs:
        keys=[key for key in by_label['first'] if key[1]==arm]
        refinement[arm]={'paired_tasks':len(keys),
            'selected_only_correct':sum(not by_label['first'][k]['task_correct'] and by_label['selected'][k]['task_correct'] for k in keys),
            'first_only_correct':sum(by_label['first'][k]['task_correct'] and not by_label['selected'][k]['task_correct'] for k in keys)}
    return {'benchmark_id':recipe['benchmark_id'],'complete':reports['selected']['complete'],
            'thinking':True,'python_execution':True,'query_feedback':False,'ttt':False,
            'first':reports['first'],'selected':reports['selected'],'refinement':refinement,
            'costs':costs,'execution_metrics':execution_metrics,'per_task':details,
            'limits':'Partial dev results are not full-set accuracy; extra inference compute differs from direct-grid conditions.'}


def score(folder):
    folder=Path(folder).resolve();benchmark,recipe=ready(folder)
    output=folder.parent/'runs'/NAME
    with results_lock(output):
        states,frontier=collect(folder,recipe)
        # This is the only runtime module opening the hidden query answers.
        answers=rows(benchmark/'answers.scorer-only.jsonl')
        report=summarize(answers,states,recipe)
        allocation_files=sorted((output/'allocations').glob('*.json'))
        allocations={p.stem:json.loads(p.read_text()) for p in allocation_files}
        current=output/'last-allocation.json'
        if current.exists():
            active=json.loads(current.read_text())
            allocations.setdefault(active['allocation_id'],active)
        report.update(scored_at=utcnow(),state_frontier=frontier,allocation_history=list(allocations.values()),
                      allocation_cost_note='Generation/worker times are measured; incomplete allocation records have no final wall time. Use sacct for allocated GPU-hours.')
        report['report_id']=message_digest(report)
        atomic_json(output/'report.json',report)
        atomic_json(output/'summary.json',{k:v for k,v in report.items() if k!='per_task'})
    print('ARC_PROGRAM_BENCHMARK '+json.dumps({k:v for k,v in report.items() if k not in ('per_task','state_frontier')}),flush=True)
    return report


if __name__=='__main__':score(Path(__file__).resolve().parent)
