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


from panel import prepare, run_panel

def run(block):
    import torch
    from axolotl.cli.utils import load_model_and_tokenizer
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack
    assert block in ('A','B') and torch.cuda.device_count()==4
    assert all('H100' in torch.cuda.get_device_name(i) for i in range(4))
    benchmark,recipe,m=ready(NATIVE)
    job=os.environ['SLURM_JOB_ID']
    path=OUT/'allocations'/job
    info=dict(job=job, block=block,phase='loading', batches=[],copy={})
    end_time=int(os.environ['SLURM_JOB_END_TIME'])
    continuation=(HERE/'parents.json').is_file()
    start=int(os.environ['SLURM_JOB_START_TIME'])
    stop=lambda:time.time()>=end_time-90
    def report():
        save(path/'allocation.json',info)
    report()
    try:
        tasks,states=prepare(block,recipe,m,path)
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
            if not info['greedy_control']['matches']:
                raise RuntimeError('Native control mismatch')
            run_panel(block,states,tasks,model,tok,m,path,info,report,start,end_time,controller)
    except BaseException as exc:
        info.update(phase='failed',error=repr(exc))
        raise
    finally:
        info['elapsed_seconds']=time.time()-start
        report()


if __name__=='__main__':
    run(sys.argv[1])
