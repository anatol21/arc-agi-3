"""Code-only response contracts; no extraction from reasoning or answer repair."""
import hashlib
import json

CONTEXT=18432
MAX_TOKENS=8192

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def grid(value):
    if not isinstance(value,list) or not 1<=len(value)<=30 or not isinstance(value[0],list) or not 1<=len(value[0])<=30:
        raise ValueError('Grid must have 1–30 rows and columns')
    if any(not isinstance(row,list) or len(row)!=len(value[0]) for row in value):raise ValueError('Grid must be rectangular')
    if any(type(c)is not int or not 0<=c<=9 for row in value for c in row):raise ValueError('Cells must be integers 0–9')
    return value

def fields(raw):
    error='Unexpected thinking delimiter in code-only generation' if '<think>' in raw or '</think>' in raw else None
    return {'raw_text':raw,'final_answer':raw.strip() if error is None else None,'reasoning':None,'separation_error':error}

def validate_record(r):
    ids=r['completion_token_ids'];eos=r['eos_token_id']
    if r['enable_thinking'] is not False or r['reasoning'] is not None:raise ValueError('Thinking response in code-only experiment')
    if type(r['generated_tokens'])is not int or r['generated_tokens']!=len(ids) or not ids or any(type(t)is not int or t<0 for t in ids):raise ValueError('Invalid token ledger')
    if r['finish_reason'] not in ('eos','token_limit','interrupted','checkpoint'):raise ValueError('Unknown stopping reason')
    if r['finish_reason']=='eos':
        if ids[-1]!=eos or eos in ids[:-1]:raise ValueError('Invalid EOS boundary')
    elif eos in ids:raise ValueError('EOS mislabelled')
    if not 0<len(ids)<=r['token_budget']<=MAX_TOKENS or r['token_budget']!=min(MAX_TOKENS,CONTEXT-r['prompt_tokens']):raise ValueError('Invalid candidate/context budget')
    if r['finish_reason']=='token_limit' and len(ids)!=r['token_budget']:raise ValueError('False token cap')
    if r['finish_reason'] in ('interrupted','checkpoint') and len(ids)>=r['token_budget']:raise ValueError('Finished candidate marked pending')
    if r['finish_reason']=='checkpoint' and (r['new_generated_tokens']!=r['segment_new_token_limit'] or not 1<=r['segment_new_token_limit']<=512):raise ValueError('Invalid checkpoint boundary')
    if 'segment_new_token_limit' in r and not 0<r['new_generated_tokens']<=r['segment_new_token_limit']<=512:raise ValueError('Invalid segment token ledger')
    for k,v in fields(r['raw_text']).items():
        if r[k]!=v:raise ValueError('Saved response extraction changed')
    if r['reasoning_tokens']!=0 or r['final_answer_tokens']+r['eos_tokens']!=len(ids):raise ValueError('Code-only accounting mismatch')
    if r['eos_tokens']!=int(r['finish_reason']=='eos'):raise ValueError('EOS accounting mismatch')
    if r['seconds']<0 or r['adapter_layers_checked']<1:raise ValueError('Invalid runtime evidence')
    return r

def render(tokenizer,messages):
    off=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    on=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    if on==off or not off.endswith('<think>\n\n</think>\n\n') or not on.endswith('<think>\n'):
        raise RuntimeError('Expected explicit thinking-disabled template; refusing ambiguity')
    return tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
