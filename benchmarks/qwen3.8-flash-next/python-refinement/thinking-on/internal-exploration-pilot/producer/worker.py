import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
NATIVE = WORK/'arc-flash-python-nothink-v2'
OUT = WORK/'runs'/HERE.name
sys.path[:0] = [str(HERE), str(NATIVE), str(WORK/'arc-python-base-speed-v1')]
from protocol_pilot import BLOCKS, RETRY_MINUTES, PANEL, SOLVER, allowance, digest, finish_turn, messages, new_state, run_probe, save
from decoding import decode, final_text, render
from cold import local_weights
import controller
import records
from runtime import ready, sha


def read(p):
    return json.loads(p.read_bytes())


def reference(model,tok):
    import torch
    from transformers import GenerationConfig
    data=read(WORK/'arc-flash-cache-pilot-v1/controls.json')
    old=read(HERE/'reference.json')
    assert [r['task_id'] for r in data]==old['task_ids']
    rows=[records.render(tok,r['messages']) for r in data]
    width=max(map(len,rows))
    assert width==old['initial_padded_width']
    ids=torch.tensor([[tok.pad_token_id]*(width-len(r))+r for r in rows],device='cuda:0')
    mask=torch.tensor([[0]*(width-len(r))+[1]*len(r) for r in rows],device='cuda:0')
    with torch.inference_mode():
        actual=model.generate(input_ids=ids,attention_mask=mask,logits_to_keep=1,
            generation_config=GenerationConfig(do_sample=False,max_new_tokens=4,min_new_tokens=4,
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True))[:,width:].tolist()
    expected=[r[:4] for r in old['new_token_ids']]
    return dict(actual=actual,expected=expected,matches=actual==expected)


def qualify(model,tok,sandbox,stop,path):
    # The only source for the random six digits in the solver turn is the probe observation.
    secret=[[secrets.randbelow(10) for _ in range(6)]]
    visible=dict(train=[],test=[dict(input=secret)])
    history=[dict(role='system',content='Python tool-use exercise. Follow the requested fenced formats exactly.'),
             dict(role='user',content='Inspect the hidden test grid by emitting exactly one fenced probe block containing print(test[0]["input"]).')]
    def turn(hist,key):
        s=new_state('synthetic-'+key,'B','qualification')
        prompt=render(tok,hist)
        while not stop() and s['used']<384 and not s.get('pending_end'):
            decode(model,tok,[(s,prompt)],lambda row:save(path/(key+'.json'),row),stop,
                   chunk=min(128,384-s['used']),reasoning_cap=64)
        if s.get('pending_end')!='eos' or not s['reasoning']:
            raise RuntimeError('Qualification failed at '+key)
        return final_text(tok,s)
    probe=turn(history,'probe')
    observed=run_probe(probe,visible,sandbox.execute_case,tok)
    save(path/'observation.json', dict(probe=probe, observation=observed))
    try:
        decoded = json.loads(observed)
    except ValueError as exc:
        raise RuntimeError('Qualification returned invalid grid: '+repr(observed)) from exc
    if decoded!=secret:
        raise RuntimeError('Qualification probe did not retrieve the challenge')
    history += [dict(role='assistant',content=probe),dict(role='user',content='Python output:\n'+observed+
                '\nWrite a fenced python block defining solve(grid) that returns the observed grid literally, for any argument.')]
    answer=turn(history,'solver')
    code=controller.extract_code(answer)
    controller.validate_code(code)
    result=sandbox.execute_case(code,[[0]])
    proof=dict(probe=probe,observation=observed,solver=answer,actual=result,expected=secret)
    save(path/'challenge.json',proof)
    if not result['executable'] or result['output']!=secret:
        raise RuntimeError('Solver failed the hidden observation challenge')


def prepare_panel(block,recipe,m,path):
    arm,half=block
    panel=PANEL[half]
    tasks={r['task_id']:r['visible'] for r in m['protocol'].rows(NATIVE/'tasks.jsonl') if r['task_id'] in panel}
    # ready() verifies the parent recipe against the original program benchmark.
    # Include both the source ID and every staged recipe field in the pilot identity.
    recipe_identity=digest([recipe['source_benchmark_id'],recipe])
    code_identity=[sha(HERE/name) for name in ('worker.py','decoding.py','protocol_pilot.py')]
    states={}
    for k in panel:
        controller.initial_state(tasks[k])
        identity=digest([block,k,tasks[k],recipe_identity,code_identity])
        file=path/'tasks'/(k+'.json')
        states[k]=read(file) if file.exists() else new_state(k,arm,identity)
        if states[k]['identity']!=identity:
            raise ValueError('Incompatible saved state for '+k)
    return tasks,states


def prepare_rows(states,tasks,tok,persist):
    rows=[]
    for k,s in states.items():
        if s['done']:
            continue
        prompt=render(tok,messages(controller,tasks[k],s))
        if allowance(s,prompt)<=0:
            s['pending_end']='context_limit' if 18432-len(prompt)-len(s['prefix'])<=0 else 'token_limit'
            persist(s)
        else:
            rows.append((s,prompt))

    rows.sort(key=lambda pair:(len(pair[1])+len(pair[0]['prefix']),pair[0]['task']))
    return rows


def run(block):
    import torch
    from axolotl.cli.utils import load_model_and_tokenizer
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack
    assert block in BLOCKS and torch.cuda.device_count()==4
    assert all('H100' in torch.cuda.get_device_name(i) for i in range(4))
    benchmark,recipe,m=ready(NATIVE)
    job=os.environ['SLURM_JOB_ID']
    path=OUT/block
    info=dict(job=job, block=block,phase='loading', batches=[],copy={})
    end_time=int(os.environ['SLURM_JOB_END_TIME'])
    continuation=(HERE/'parents.json').is_file()
    start=end_time-(25 if continuation else RETRY_MINUTES[block])*60
    stop=lambda:time.time()>=end_time-90
    def report():
        save(path/'allocation.json',info)
    report()
    try:
        tasks,states=prepare_panel(block,recipe,m,path)
        if continuation:
            from continuation import restore, panel
            restore(block,states,WORK,HERE,sha)
        info['sandbox']=m['sandbox'].preflight()
        cfg=m['configuration'].load_inference_config(NATIVE/'inference-config.json')
        plan=read(WORK/'arc-bf16-base-debug-v1/pilot-plan.json')
        assert Path(cfg['base_model']).resolve()==Path(plan['model_path']).resolve()
        assert cfg['load_in_4bit'] and cfg['quantize_moe_experts']
        with ExitStack() as stack:
            cfg['base_model']=str(stack.enter_context(local_weights(WORK,plan,info['copy'])))
            layout=dict(cfg['device_map'])
            for name,device in layout.items():
                layer=re.match(r'model\.language_model\.layers\.(\d+)(?:\.|$)',name)
                if device!='cpu':
                    layout[name]=int(layer[1])//12 if layer else (3 if device==2 else device)
            cfg['device_map']=layout
            model,tok,_=load_model_and_tokenizer(cfg=cfg,inference=False)
            m['ple_placement'].place_small_ple_buffers(model)
            info['adapter']=m['checkpoint_validation'].validate_adapters(torch,Path(recipe['checkpoint']),model)
            model.eval().requires_grad_(False)
            model.gradient_checkpointing_disable()
            stack.enter_context(model.disable_adapter())
            info['disabled_layers']=m['generation'].assert_adapter_mode(model,False)
            devices={p.device.index for p in model.parameters() if p.is_cuda}
            if devices!=set(range(4)) or not any(getattr(p,'quant_state',None) is not None for p in model.parameters()):
                raise RuntimeError('Missing quantized four-device base')
            if (tok.eos_token_id,tok.pad_token_id)!=(recipe['eos_token_id'],recipe['pad_token_id']):
                raise RuntimeError('Tokenizer identity mismatch')
            stack.enter_context(torch.autocast('cuda',dtype=torch.bfloat16))
            info.update(ready_seconds=time.time()-start,phase='qualification')
            report()
            info['greedy_control']=reference(model,tok)
            report()
            if continuation:
                panel(block,states,tasks,model,tok,m,path,info,report,start,end_time,prepare_rows,controller)
                return
            qualify(model,tok,m['sandbox'],stop,path/'control')
            info.update(phase='panel',qualified=True)
            report()
            def persist(s):
                save(path/'tasks'/(s['task']+'.json'),s)
            pool=stack.enter_context(ThreadPoolExecutor(max_workers=4))
            def consume(s):
                if s.get('pending_end'):
                    end=s.pop('pending_end')
                    finish_turn(s,final_text(tok,s),end,controller,tasks[s['task']],m['sandbox'].execute_case,tok)
                    s.pop('turn_reason_limit',None)
                    if s['done']:
                        s['completed_epoch']=time.time()
                    persist(s)
            while not stop():
                list(pool.map(consume, states.values()))
                rows=prepare_rows(states,tasks,tok,persist)
                if all(s['done'] for s in states.values()):
                    break
                if not rows:
                    continue
                info['batches'].append(dict(tasks=[s['task'] for s,_ in rows],start=time.time()))
                report()
                m['generation'].clear_generation_state(model)
                decode(model,tok,rows,persist,stop)
                report()
            info.update(phase='ended',finished=[k for k,s in states.items() if s['done']],
                        peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(4)])
    except BaseException as exc:
        info.update(phase='failed',error=repr(exc))
        raise
    finally:
        info['elapsed_seconds']=time.time()-start
        report()


if __name__=='__main__':
    run(sys.argv[1])
