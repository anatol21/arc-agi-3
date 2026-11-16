"""Matched continuation from the original journals; outputs live in a separate run."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from protocol_pilot import PANEL, finish_turn, save
from decoding import decode, final_text


def restore(block, states, work, here, sha):
    manifest=json.loads((here/'parents.json').read_bytes())
    for key,state in states.items():
        source=work/'runs/arc-python-exploration-pilot-v4'/block/'tasks'/(key+'.json')
        expected=manifest[f'{block}/{key}']
        if sha(source)!=expected:
            raise RuntimeError(f'Changed source journal {source}')
        parent=json.loads(source.read_bytes())
        if (parent['done'] or parent['attempts'] or parent['probes'] or
            parent['task']!=key or parent['condition']!=block[0] or
            parent['total']!=len(parent['prefix']) or parent['reasoning']!=parent['total'] or
            parent.get('pending_end') or not parent['rng']):
            raise RuntimeError(f'Unexpected parent state {key}')
        for field in ('prefix','rng','reasoning','used','total'):
            state[field]=parent[field]
        state.update(parent_sha256=expected,inherited_tokens=parent['total'],turn_reason_limit=0)


def panel(block,states,tasks,model,tok,m,path,info,report,start,end_time,prepare_rows,controller):
    def persist(state):
        save(path/'tasks'/(state['task']+'.json'),state)
    def rows():
        return prepare_rows(states,tasks,tok,persist)
    def generate(batch,stop,chunk):
        m['generation'].clear_generation_state(model)
        decode(model,tok,batch,persist,stop,chunk=chunk,reasoning_cap=0,code_mode=True)
    boundary=tok.encode('</think>',add_special_tokens=False)
    before={k:(s['prefix'][:],s['rng']) for k,s in states.items()}
    generate(rows(),lambda:False,1)
    for k,s in states.items():
        prefix,rng=before[k]
        if s['prefix']!=prefix+boundary or s['rng']!=rng:
            raise RuntimeError(f'Resumed boundary or RNG mismatch: {k}')
    clock=time.time()
    deadline=min(clock+900,end_time-45)
    gen_end=deadline-180
    info.update(phase='panel',verified_switch=list(states),panel_start=clock,
                generation_deadline=gen_end,execution_deadline=deadline)
    report()
    def execute(code,data):
        remaining=deadline-time.time()
        if remaining<=0:
            raise TimeoutError('task deadline')
        return m['sandbox'].execute_case(code,data,timeout=min(12,remaining))
    def consume(s):
        if s.get('pending_end'):
            try:
                finish_turn(s,final_text(tok,s),s['pending_end'],controller,tasks[s['task']],execute,tok)
            except TimeoutError:
                return
            s.pop('pending_end',None)
            s.pop('turn_reason_limit',None)
            persist(s)
    with ThreadPoolExecutor(max_workers=4) as pool:
        while time.time()<gen_end:
            list(pool.map(consume,states.values()))
            batch=rows()
            if not batch and all(s['done'] for s in states.values()):
                break
            if not batch or time.time()>=gen_end:
                continue
            info['batches'].append(dict(tasks=[s['task'] for s,_ in batch],start=time.time()))
            generate(batch,lambda:time.time()>=gen_end,128)
            report()
        list(pool.map(consume,states.values()))
        # Score the best fully evaluated candidate even if time precluded another revision.
        def settle(s):
            if s['done'] or not any(a['feedback']['executed_cases'] for a in s['attempts']):
                return
            try:
                from protocol_pilot import select_program
                select_program(s,tasks[s['task']],execute)
            except TimeoutError:
                return
            persist(s)
        list(pool.map(settle,states.values()))
    info.update(phase='ended',finished=[k for k,s in states.items() if s['done']],
                new_tokens={k:s['total']-s['inherited_tokens'] for k,s in states.items()})
    report()
