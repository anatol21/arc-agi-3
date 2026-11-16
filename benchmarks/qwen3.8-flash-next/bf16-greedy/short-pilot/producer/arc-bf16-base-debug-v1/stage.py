import hashlib
import json
from pathlib import Path
import os


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def filehash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_plan(folder):
    plan=json.loads((folder/"pilot-plan.json").read_text())
    if plan['benchmark_id']!=digest({k:v for k,v in plan.items() if k!='benchmark_id'}):
        raise ValueError("Corrupt plan")
    return plan


def build(folder):
    w=folder.parent
    parent=w/'arc-flash-eval138-v1'
    receipt=json.loads((parent/'benchmark-recipe.json').read_text())
    gen=receipt['generation']
    expect=dict(max_new_tokens=8192, context=18432, do_sample=False,enable_thinking=False,use_cache=True,num_beams=1,num_return_sequences=1,seed=42)
    if any(gen[k]!=v for k,v in expect.items()):
        raise ValueError("Native generation configuration changed")
    arc=parent/'prompts.jsonl'
    if filehash(arc)!=receipt['prompts_sha256']:
        raise ValueError("Original prompts changed")
    util=w/'arc-flash-eval138-util-v1'
    tokfile=util/'tokenized-prompts.json'
    if filehash(tokfile)!=json.loads((util/'READY.json').read_text())['files_sha256'][tokfile.name]:
        raise ValueError("Archived tokens changed")
    tokens=json.loads(tokfile.read_text())
    prompts={p['task_id']:p for p in map(json.loads,arc.read_text().splitlines())}
    if len(tokens)!=138 or set(tokens)!=set(prompts):
        raise ValueError("Incomplete prompt inventory")
    ids=sorted(tokens,key=lambda t:(len(tokens[t]),t))[:12]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(receipt['checkpoint'],local_files_only=True)
    rows={}
    for task in ids:
        p=prompts[task]
        old=tokens[task]
        current=tok.apply_chat_template(p['messages'], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        if old!=current:
            raise ValueError(f'Tokenization changed on {task}')
        rows[task]=dict(p,token_ids=old,prompt_tokens=len(old),messages_sha256=digest(p['messages']),prompt_sha256=digest(old))
    plan=dict(model_path=str(w/'models/qwen3.8-flash-next'),checkpoint_tokenizer=receipt['checkpoint'],
              eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,groups=[ids[:4],ids[4:]],prompts=rows,generation=gen,
              model_manifest_sha256=filehash(w/'arc-flash-run-v6/model-verified.json'),package_sha256=filehash(folder/'package.sha256'))
    plan['benchmark_id']=digest(plan)
    out=folder/'pilot-plan.json'
    if out.exists() and read_plan(folder)!=plan: raise ValueError("Different pilot already staged")
    tmp=folder/'pilot-plan.tmp'
    tmp.write_text(json.dumps(plan)+"\n");tmp.replace(out)
    print('BF16_STAGED',plan['groups'],plan['benchmark_id'])


if __name__=='__main__':
    build(Path(__file__).resolve().parent)
