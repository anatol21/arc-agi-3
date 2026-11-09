"""Thinking plus real Python refinement, with durable per-task state."""
import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

if __package__:
    from .agent import advance,initial_state,validate_state
    from .generation import generate_one
    from .sandbox import execute,preflight
    from .setup_runtime import NAME,ready,atomic_json,atomic_text,utcnow
    from .baseline_common import output_lock
    from .deadline import allocation_budget
    from .protocol import rows,validate_template
else:
    from agent import advance,initial_state,validate_state
    from generation import generate_one
    from sandbox import execute,preflight
    from setup_runtime import NAME,ready,atomic_json,atomic_text,utcnow
    from baseline_common import output_lock
    from deadline import allocation_budget
    from protocol import rows,validate_template


def run(folder,debug,allocation):
    benchmark,recipe=ready(folder)
    if __package__:from .probe_sandbox import verify_probe
    else:from probe_sandbox import verify_probe
    verify_probe(folder)
    output=folder.parent/'runs'/NAME
    budget=allocation_budget();atomic_json(output/'allocation-budget.json',budget)
    stopped=[]
    previous={}
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        previous[sig]=signal.getsignal(sig);signal.signal(sig,lambda n,f:stopped.append(n))
    stop=lambda:bool(stopped) or time.time()>=budget['stop_at']
    model=tokenizer=torch=None
    try:
        if stop():return
        print('PROGRAM_SANDBOX_PREFLIGHT_START',flush=True)
        atomic_json(output/'sandbox-preflight.json',preflight())
        print('PROGRAM_SANDBOX_PREFLIGHT_PASSED',flush=True)
        prompts=rows(folder/'prompts.jsonl')
        targets=prompts[:1] if debug else prompts
        if not debug:raise RuntimeError('This candidate prepares debug only; review its evidence before enabling a full run')
        output_recipe=output/'recipe.json'
        if output_recipe.exists() and json.loads(output_recipe.read_text())!=recipe:raise RuntimeError('Existing program experiment differs')
        atomic_json(output_recipe,recipe)
        for task in targets:
            visible=json.loads(task['messages'][0]['content'].split('\nTask:\n',1)[1])
            for arm in ('finetuned','base'):
                trace=output/task['task_id']/arm;trace.mkdir(parents=True,exist_ok=True)
                state_path=trace/'state.json'
                state=json.loads(state_path.read_text()) if state_path.exists() else initial_state(visible)
                if state.get('benchmark_id',recipe['benchmark_id'])!=recipe['benchmark_id']:raise RuntimeError('Wrong state provenance')
                validate_state(visible,state)
                if state['complete']:continue
                def save(value):
                    value.update(benchmark_id=recipe['benchmark_id'],task_id=task['task_id'],arm=arm)
                    atomic_json(state_path,value)
                    # Candidate code is for inspection; never execute it directly on the host.
                    for attempt in value['attempts']:
                        stem='candidate-'+str(attempt['candidate'])
                        if 'code' in attempt:atomic_text(trace/(stem+'.py'),attempt['code'])
                        for key,suffix in [('raw_text','raw.txt'),('reasoning','reasoning.txt'),('final_answer','final.txt')]:
                            atomic_text(trace/(stem+'.'+suffix),attempt['response'].get(key) or '')
                        atomic_json(trace/(stem+'.json'),attempt)
                    if value['pending']:
                        atomic_json(trace/'pending-response.json',value['pending'])
                    elif (trace/'pending-response.json').exists():
                        (trace/'pending-response.json').unlink()
                    if value['complete']:
                        atomic_json(trace/'first-output.json',value['first_output_json'])
                        atomic_json(trace/'selected-output.json',value['selected_output_json'])
                def generate(messages,prefix):
                    nonlocal model,tokenizer,torch
                    if model is None:
                        import torch as runtime_torch
                        torch=runtime_torch
                        if torch.cuda.device_count()!=3 or int(os.environ.get('WORLD_SIZE','1'))!=1:raise RuntimeError('Use one process, three GPUs')
                        from axolotl.cli.utils import load_model_and_tokenizer
                        if __package__:
                            from .configuration import load_inference_config
                            from experiments.cluster.guarded_train import check_layout
                            from experiments.cluster.ple_placement import place_small_ple_buffers
                            from experiments.cluster.checkpoint_validation import validate_adapters
                        else:
                            from configuration import load_inference_config
                            from guarded_train import check_layout
                            from ple_placement import place_small_ple_buffers
                            from checkpoint_validation import validate_adapters
                        started=time.monotonic()
                        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
                        model,tokenizer,_=load_model_and_tokenizer(cfg=load_inference_config(folder/'inference-config.json'),inference=False)
                        place_small_ple_buffers(model);layout=check_layout(model)
                        if layout['trainable_parameters']!=33478656:raise RuntimeError('Reloaded adapter coverage changed')
                        adapter=validate_adapters(torch,Path(recipe['checkpoint']),model)
                        if str(model.get_input_embeddings().weight.device)!='cuda:0':raise RuntimeError('Input device changed')
                        if tokenizer.eos_token_id!=recipe['eos_token_id'] or tokenizer.pad_token_id!=recipe['pad_token_id']:raise RuntimeError('Tokenizer changed')
                        model.requires_grad_(False);model.eval();model.gradient_checkpointing_disable()
                        for module in model.modules():
                            config=getattr(module,'config',None)
                            if config is not None and hasattr(config,'use_cache'):config.use_cache=True
                        allocation['loading_seconds']=time.monotonic()-started
                        atomic_json(output/'model-reload.json',{'layout':layout,'adapter':adapter,'optimizer_updates':0,'requires_grad_parameters':sum(p.requires_grad for p in model.parameters()),'allocation_id':allocation['allocation_id']})
                        atomic_json(output/'thinking-template-validation.json',validate_template(tokenizer,messages))
                    if stop():
                        # Controller checks stop before calling. A deadline crossed during loading
                        # must not generate or spend one of the candidate slots.
                        raise AllocationDeadline('Deadline reached while loading; state preserved')
                    original=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)
                    if len(original)>=18432:return {'finish_reason':'context_limit'}
                    response=generate_one(torch,model,tokenizer,{'task_id':task['task_id'],'messages':messages},arm,recipe,stop,prefix=prefix)
                    print('ARC_PROGRAM_GENERATION '+json.dumps({k:response[k] for k in ('task_id','arm','new_generated_tokens','generated_tokens','finish_reason','seconds')}),flush=True)
                    return response
                try:advance(visible,state,generate,execute,save,stop)
                except AllocationDeadline:return
                validate_state(visible,state)
                if state['complete']:
                    print('ARC_PROGRAM_RESULT '+json.dumps({'task_id':task['task_id'],'arm':arm,'candidates':len(state['attempts']),'selected':state['selected_candidate'],'tokens':state['total_generated_tokens']}),flush=True)
                if stop():return
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


class AllocationDeadline(RuntimeError):pass


def main(folder,debug=True):
    output=folder.parent/'runs'/NAME
    with output_lock(output):
        allocation={'allocation_id':uuid.uuid4().hex,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'hostname':socket.gethostname(),'started_at':utcnow(),'phase':'verifying'}
        os.environ['ARC_THINKING_ALLOCATION_ID']=allocation['allocation_id']
        atomic_json(output/'last-allocation.json',allocation)
        error=None
        try:run(folder,debug,allocation)
        except BaseException as failure:
            error=failure;allocation.update(phase='failed',error=repr(failure),traceback=traceback.format_exc())
        if (folder/'READY.json').exists():
            try:
                result=subprocess.run([sys.executable,str(folder/'score.py')],check=False)
                if result.returncode and error is None:error=RuntimeError('Program scoring failed')
            except BaseException as scoring_failure:
                if error is None:error=scoring_failure
        if error is None and debug:
            try:
                if __package__:from .debug_gate import check_debug
                else:from debug_gate import check_debug
                allocation['debug_phase']=check_debug(folder)['phase']
            except BaseException as gate_failure:error=gate_failure
        if error is not None:allocation.setdefault('error',repr(error))
        allocation.update(finished_at=utcnow(),phase='failed' if error else 'debug_finished')
        atomic_json(output/'allocations'/(allocation['allocation_id']+'.json'),allocation)
        atomic_json(output/'last-allocation.json',allocation)
        if error is not None:raise error



if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--debug',action='store_true');args=parser.parse_args()
    main(Path(__file__).resolve().parent,debug=args.debug)
