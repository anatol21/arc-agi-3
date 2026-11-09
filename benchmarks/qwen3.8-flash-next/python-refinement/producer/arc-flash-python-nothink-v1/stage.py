import json
from pathlib import Path
if __package__:
    from .runtime import NAME,bridge,ready,sha
    from . import controller,records
else:
    from runtime import NAME,bridge,ready,sha
    import controller,records

def stage(folder):
    program,benchmark,old,m=bridge(folder)
    if (folder/'READY.json').exists():ready(folder);print('CODE-ONLY EXPERIMENT ALREADY STAGED; no job submitted');return
    tasks=[]
    for task in m['protocol'].rows(program/'prompts.jsonl'):
        visible=json.loads(task['messages'][0]['content'].split('\nTask:\n',1)[1]);controller.initial_state(visible)
        tasks.append({'task_id':task['task_id'],'visible':visible})
    if len(tasks)!=128 or len({t['task_id'] for t in tasks})!=128:raise RuntimeError('Expected all 128 distinct dev tasks')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(old['checkpoint'],local_files_only=True)
    lengths={}
    for task in tasks:
        ids=records.render(tokenizer,controller.messages_for(task['visible'],[]))
        if len(ids)>=records.CONTEXT:raise RuntimeError('Full initial prompt exceeds context; no truncation permitted')
        lengths[task['task_id']]=len(ids)
    cfg=json.loads((program/'inference-config.json').read_text());cfg['output_dir']=str(folder.parent/'runs'/NAME/'loader')
    m['result_io'].atomic_json(folder/'inference-config.json',cfg);m['configuration'].schema_preflight(folder/'inference-config.json')
    m['result_io'].atomic_text(folder/'tasks.jsonl',''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in tasks))
    m['result_io'].atomic_json(folder/'template-check.json',{'thinking_disabled_verified':True,'initial_prompt_lengths':lengths,'tokenizer':old['checkpoint']})
    recipe={'name':NAME,'source_benchmark_id':old['benchmark_id'],'checkpoint':old['checkpoint'],'adapter_sha256':old['adapter_sha256'],
        'eos_token_id':old['eos_token_id'],'pad_token_id':old['pad_token_id'],'task_ids':[t['task_id'] for t in tasks],
        'context':18432,'max_tokens_per_candidate':8192,'max_candidates':4,'max_tokens_per_task_arm':32768,
        'enable_thinking':False,'do_sample':False,'num_beams':1,'seed':42,'arm_order':['finetuned','base'],
        'query_feedback':False,'weight_updates':False,'prompt_change':'Remove request for visible reasoning; otherwise preserve original Python synthesis instructions',
        'controller_sha256':sha(folder/'controller.py'),'source_manifest_sha256':sha(program/'package-manifest.json'),
        'serial_debug_task_ids':[t['task_id'] for t in tasks[:4]],'batched_debug_policy':'first max(4,batch_size) tasks','tasks_sha256':sha(folder/'tasks.jsonl')}
    m['result_io'].atomic_json(folder/'recipe.json',recipe)
    m['result_io'].atomic_json(folder/'READY.json',{'files':{n:sha(folder/n) for n in ('recipe.json','tasks.jsonl','inference-config.json','template-check.json')}})
    print('CODE-ONLY REFINEMENT STAGED: separate serial and batch-size variants through64; both models, four candidates, unchanged budgets. Debug a fixed task prefix; no job submitted.')
if __name__=='__main__':stage(Path(__file__).resolve().parent)
