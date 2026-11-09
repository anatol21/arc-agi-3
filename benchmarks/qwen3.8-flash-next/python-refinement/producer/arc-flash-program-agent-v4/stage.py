import json
from pathlib import Path

if __package__:
    from .setup_runtime import NAME,digest,verified_dependencies,ready,atomic_json
    from .protocol import validate_template,rows
    from .configuration import schema_preflight
    from .agent import SYSTEM,MAX_CANDIDATES,TOTAL_TOKENS,initial_state,messages_for
else:
    from setup_runtime import NAME,digest,verified_dependencies,ready,atomic_json
    from protocol import validate_template,rows
    from configuration import schema_preflight
    from agent import SYSTEM,MAX_CANDIDATES,TOTAL_TOKENS,initial_state,messages_for


def stage(run):
    run=Path(run).resolve()
    benchmark,original=verified_dependencies(run)
    if (run/'READY.json').exists():
        ready(run);print('PROGRAM BENCHMARK ALREADY STAGED; unchanged inputs verified');return
    recipe={'name':NAME,'checkpoint':original['checkpoint'],'adapter_sha256':original['adapter_sha256'],
            'source_benchmark_id':original['benchmark_id'],'generation':dict(original['generation'],enable_thinking=True),
            'eos_token_id':original['eos_token_id'],'pad_token_id':original['pad_token_id'],
            'candidates_per_task_arm':MAX_CANDIDATES,'total_generated_tokens_per_task_arm':TOTAL_TOKENS,
            'reflection':'combined diagnosis and revised Python code','response_format':'one python code block defining solve(grid)',
            'selection':'first all-demo match; otherwise most exact demo matches, earliest tie; at least one executable demo',
            'sandbox_filesystem':'read-only; no writable scratch (bubblewrap 0.4.1 compatible)',
            'sandbox_environment':'Popen explicit PATH and LANG only',
            'query_feedback':False,'first_and_selected_queries_scored':True,'thinking':True,
            'history':'full task, previous candidate and feedback, compact score history; no truncation',
            'continuation':'exact partial response token prefix replay with remaining candidate budget; fresh cache',
            'system_prompt_sha256':__import__('hashlib').sha256(SYSTEM.encode()).hexdigest(),
            'prompts_sha256':digest(benchmark/'prompts.jsonl'),'optimizer_updates':0,'ttt':False}
    recipe['benchmark_id']=__import__('hashlib').sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    (run/'prompts.jsonl').write_bytes((benchmark/'prompts.jsonl').read_bytes())
    config=json.loads((benchmark/'inference-config.json').read_text())
    config['output_dir']=str(run.parent/'runs'/NAME/'loader')
    atomic_json(run/'inference-config.json',config)
    check=schema_preflight(run/'inference-config.json')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(recipe['checkpoint'],local_files_only=True)
    lengths=[]
    for task in rows(run/'prompts.jsonl'):
        visible=json.loads(task['messages'][0]['content'].split('\nTask:\n',1)[1])
        initial_state(visible)
        history=messages_for(visible,[])
        length=len(tokenizer.apply_chat_template(history,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True))
        if length>=18432:raise RuntimeError('Untruncated initial prompt exceeds context: '+task['task_id'])
        lengths.append({'task_id':task['task_id'],'prompt_tokens':length})
    atomic_json(run/'prompt-lengths.json',lengths)
    visible=json.loads(rows(run/'prompts.jsonl')[0]['messages'][0]['content'].split('\nTask:\n',1)[1])
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':'Task:\n'+json.dumps(visible)}]
    atomic_json(run/'thinking-template.json',validate_template(tokenizer,messages))
    atomic_json(run/'schema-preflight.json',check)
    atomic_json(run/'recipe.json',recipe)
    atomic_json(run/'READY.json',{'files':{p.name:digest(p) for p in run.iterdir() if p.is_file() and p.name!='READY.json'}})
    print('PROGRAM AGENT STAGED: both models, four candidates, thinking plus real Python execution')
    print('No job submitted. The debug allocation probes isolation before loading the model.')


if __name__=='__main__':stage(Path(__file__).resolve().parent)
