"""Resume only the time-limited pairs selected from the first A/B allocations."""
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
def limit_reason(state):
    if not state['done']:
        return None
    if state.get('stop_reason')=='task_time_limit':
        return 'task_time_limit'
    attempts=state['attempts']
    if attempts:
        reason=attempts[-1]['response']['finish_reason']
        if reason in {'token_limit','context_limit','probe_limit'}:
            return reason
    return None


def targets(parent,arm):
    result={}
    for path in sorted((parent/arm/'tasks').glob('*.json')):
        state=json.loads(path.read_bytes())
        reason=limit_reason(state)
        if reason:
            result[path.stem]=reason
    return result


def seed(parent, arm):
    from protocol_pilot import save
    keys=targets(parent,arm)
    for key,reason in keys.items():
        target=parent/'retry30'/arm/'tasks'/(key+'.json')
        if target.exists():
            continue
        state=json.loads((parent/arm/'tasks'/(key+'.json')).read_bytes())
        if state['task']!=key or state['condition']!=arm:
            raise ValueError('Unexpected retry parent '+key)
        for field in ('stop_reason','selected','selected_candidate','first'):
            state.pop(field,None)
        state['retry_reason']=reason
        state['parent_total']=state['total']
        state['parent_seconds']=state.get('active_seconds',0)
        state['retry_turns']=max(4,len(state['attempts'])+1)
        state['compact_prompt']=reason=='context_limit' and not state['prefix']
        state['done']=False
        save(target,state)


def run(arm):
    import worker
    import panel
    parent=worker.OUT
    keys=targets(parent,arm)
    seed(parent,arm)
    panel.OUT=worker.OUT=parent/'retry30'
    panel.TASK_SECONDS=1800
    def prepare(*args):
        tasks,states=panel.prepare(*args)
        return ({k:tasks[k] for k in keys},{k:states[k] for k in keys})
    worker.prepare=prepare
    worker.run(arm)

if __name__=='__main__':
    run(sys.argv[1])
