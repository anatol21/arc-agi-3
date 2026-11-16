"""Held-out paired Python benchmark. Worker reads only visible grids."""
import copy
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from protocol_pilot import allowance, digest, finish_turn, messages, new_state, save, select_program
from decoding import decode, final_text, render

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'runs' / HERE.name
CHUNK = 128
CAP = 24
TASK_SECONDS = 900
EXECUTION_RESERVE = 180


def prepare(block, recipe, modules, path):
    inputs = [json.loads(line) for line in (HERE/'visible.jsonl').read_text().splitlines()]
    tasks = {r['task_id']: {'train': r['train'], 'test': r['test']} for r in inputs}
    states = {}
    for key, visible in tasks.items():
        assert all(set(q) == {'input'} for q in visible['test'])
        identity = digest([HERE.name, recipe, block, key, visible])
        journal = OUT/block/'tasks'/(key+'.json')
        state = json.loads(journal.read_text()) if journal.exists() else new_state(key, block, identity)
        if state['identity'] != identity:
            raise ValueError('Journal identity mismatch: '+key)
        states[key] = state
    return tasks, states


def admit(rows, caps):
    # Anchor on the earliest admitted unfinished task; refill from nearby lengths.
    def width(row):
        return len(row[1]) + len(row[0]['prefix'])
    anchor = min(rows, key=lambda r: (not r[0].get('total', 0), width(r), r[0]['task']))
    base = width(anchor)
    result = [anchor]
    for row in sorted((r for r in rows if r is not anchor), key=lambda r: (abs(width(r)-base),r[0]['task'])):
        sizes = [width(r) for r in result]+[width(row)]
        longest = max(sizes)
        count = len(sizes)
        if count > caps.get(str(longest.bit_length()), CAP):
            continue
        if longest*count > 90000 or longest*count > sum(sizes)*1.35:
            continue
        result.append(row)
    return result


def run_panel(block, states, tasks, model, tok, modules, path, info, report, start, end_time, controller):
    import torch
    caps_path = OUT/block/'batch-caps.json'
    caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
    end = end_time-50
    prompt_cache = {}
    def persist(s):
        save(OUT/block/'tasks'/(s['task']+'.json'), s)
    def consume(s):
        then = time.monotonic()
        left = TASK_SECONDS-s.get('active_seconds', 0)
        deadline = min(end, time.time()+max(0,left))
        def execute(code, data):
            seconds = deadline-time.time()
            if seconds <= 0:
                raise TimeoutError('execution deadline')
            return modules['sandbox'].execute_case(code,data,timeout=min(12,seconds))
        try:
            if s.get('pending_end'):
                finish_turn(s, final_text(tok,s), s['pending_end'], controller,tasks[s['task']],execute,tok)
                s.pop('pending_end')
                s.pop('turn_reason_limit',None)
                prompt_cache.pop(s['task'],None)
            if not s['done'] and (s.get('generation_seconds',0)>=TASK_SECONDS-EXECUTION_RESERVE or left<=EXECUTION_RESERVE):
                if s['attempts']:
                    select_program(s,tasks[s['task']],execute)
                else:
                    s.update(selected=[None]*len(tasks[s['task']]['test']),done=True)
                s['stop_reason']='task_time_limit'
        except TimeoutError:
            if time.time() < end:
                s.update(done=True, stop_reason='task_time_limit', selected=[None]*len(tasks[s['task']]['test']))
        finally:
            s['active_seconds']=s.get('active_seconds',0)+time.monotonic()-then
            persist(s)
    with ThreadPoolExecutor(max_workers=16) as pool:
        while time.time()<end:
            live=[s for s in states.values() if not s['done']]
            if not live:
                break
            pending=[s for s in live if s.get('pending_end') or s.get('generation_seconds',0)>=TASK_SECONDS-EXECUTION_RESERVE or s.get('active_seconds',0)>=TASK_SECONDS-EXECUTION_RESERVE]
            if pending:
                list(pool.map(consume,pending))
                continue
            if time.time() >= end-180:
                break
            rows=[]
            for s in live:
                key=s['task']
                if key not in prompt_cache:
                    history=messages(controller,tasks[key],s)
                    if s.get('compact_prompt'):
                        history[1]['content']='Task:\n'+json.dumps(tasks[key],separators=(',',':'))
                    prompt_cache[key]=render(tok,history)
                prompt=prompt_cache[key]
                if allowance(s,prompt)<=0:
                    s['pending_end']='context_limit'
                    persist(s)
                else:
                    rows.append((s,prompt))
            if not rows:
                continue
            batch=admit(rows,caps)
            snapshot=[copy.deepcopy(s) for s,_ in batch]
            width=max(len(p)+len(s['prefix']) for s,p in batch)
            event=dict(tasks=[s['task'] for s,_ in batch],width=width,start=time.time(),size=len(batch))
            info['batches'].append(event)
            report()
            deadline=min(end-180,time.time()+min(TASK_SECONDS-EXECUTION_RESERVE-s.get('active_seconds',0) for s,_ in batch))
            oom=False
            try:
                modules['generation'].clear_generation_state(model)
                # Bound fresh reasoning before code; each row keeps its own RNG journal.
                decode(model,tok,batch,lambda s:None,lambda:time.time()>=deadline,chunk=CHUNK,reasoning_cap=512,code_mode=True)
            except torch.cuda.OutOfMemoryError:
                oom=True
            elapsed=time.time()-event['start']
            if oom:
                for (s,_),old in zip(batch,snapshot):
                    s.clear()
                    s.update(old)
                gc.collect()
                torch.cuda.empty_cache()
                if len(batch)==1:
                    raise RuntimeError('OOM on a single row')
                new_cap=max(1,len(batch)//2)
                # Also bound the anchor bucket when two adjacent buckets were combined.
                for s,p in batch:
                    bucket=str((len(p)+len(s['prefix'])).bit_length())
                    caps[bucket]=min(caps.get(bucket,CAP),new_cap)
                save(caps_path,caps)
            for s,_ in batch:
                s['active_seconds']=s.get('active_seconds',0)+elapsed
                s['generation_seconds']=s.get('generation_seconds',0)+elapsed
                persist(s)
            event.update(seconds=elapsed,oom=oom)
            report()
    info.update(phase='ended',finished=[k for k,s in states.items() if s['done']])
    report()
