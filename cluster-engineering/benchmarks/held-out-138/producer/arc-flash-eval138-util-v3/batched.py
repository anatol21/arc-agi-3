"""Native direct-answer batching with independent caps and exact saved-prefix replay."""
import contextlib
import os
import time
import uuid
if __package__:from . import records,policy
else:import records,policy

def request_id(recipe,request):
    return records.digest({'benchmark_id':recipe['benchmark_id'],'task_id':request['task_id'],
        'arm':request['arm'],'messages':request['messages'],'prefix':request['prefix']})

def prepare(tokenizer,request):
    prompt=records.render(tokenizer,request['messages']);prefix=request['prefix']
    budget=min(records.MAX_TOKENS,records.CONTEXT-len(prompt))
    if budget<=0:return None
    if len(prefix)>=budget or tokenizer.eos_token_id in prefix:raise RuntimeError('Invalid saved prefix')
    return {'request':request,'prompt':prompt,'prefix':prefix,'budget':budget,
            'ids':prompt+prefix,'remaining':budget-len(prefix)}

def split_new(tokens,eos,pad,remaining,interrupted):
    if eos in tokens:
        at=tokens.index(eos)+1
        if at>remaining or any(t!=pad for t in tokens[at:]):raise RuntimeError('Bad batched EOS/padding boundary')
        return tokens[:at],'eos'
    if len(tokens)>=remaining:
        if any(t!=pad for t in tokens[remaining:]):raise RuntimeError('Tokens generated after independent cap')
        return tokens[:remaining],'token_limit'
    if interrupted and tokens:return tokens,'interrupted'
    raise RuntimeError('Unexplained row termination')

def generate(torch,model,tokenizer,items,arm,recipe,stop,frozen,*,capacity_limit=None):
    from transformers import GenerationConfig,LogitsProcessor,LogitsProcessorList,StoppingCriteria,StoppingCriteriaList
    if not items or any(x['request']['arm']!=arm for x in items):raise ValueError('Mixed adapter batch')
    width=max(len(x['ids']) for x in items);limit=min(policy.CHUNK_TOKENS,max(x['remaining'] for x in items))
    if capacity_limit is None and not policy.accepts(items,recipe['batch_size'],chunk_tokens=policy.CHUNK_TOKENS):raise RuntimeError('Unadmitted batch shape')
    if capacity_limit is not None:
        if not 0<capacity_limit<=1024:raise ValueError('Engineering capacity window 1–1024 only')
        limit=min(limit,capacity_limit)
    if width+limit>records.CONTEXT:raise RuntimeError('Padded batch exceeds context')
    device=model.get_input_embeddings().weight.device;pad=recipe['pad_token_id'];eos=recipe['eos_token_id']
    if pad==eos:raise RuntimeError('Distinct EOS/padding required')
    ids=torch.full((len(items),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
    for i,x in enumerate(items):
        ids[i,width-len(x['ids']):]=torch.tensor(x['ids'],dtype=torch.long,device=device);mask[i,width-len(x['ids']):]=1
    class Finite(LogitsProcessor):
        def __call__(self,input_ids,scores):
            bad=~torch.isfinite(scores)
            if capacity_limit is not None:
                # HF MinNewTokens masks EOS with -inf during an engineering probe.
                bad[:,eos] &= ~torch.isneginf(scores[:,eos])
            if bad.any().item():raise RuntimeError('Nonfinite batched logits')
            return scores
    class Bounds(StoppingCriteria):
        def __call__(self,input_ids,scores,**kw):
            count=input_ids.shape[1]-width
            return torch.tensor([stop() or count>=min(x['remaining'],limit) for x in items],dtype=torch.bool,device=input_ids.device)
    cfg=GenerationConfig(do_sample=False,num_beams=1,num_return_sequences=1,max_new_tokens=limit,
        min_new_tokens=limit if capacity_limit is not None else 0,
        use_cache=True,cache_implementation=None,eos_token_id=eos,pad_token_id=pad,bos_token_id=tokenizer.bos_token_id,
        return_dict_in_generate=True,output_scores=False,output_hidden_states=False,output_attentions=False)
    frozen.clear_generation_state(model)
    def sync():
        if device.type=='cuda':
            for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
    if device.type=='cuda':
        for i in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(i)
    sync();started=time.monotonic();batch_id=uuid.uuid4().hex
    with model.disable_adapter() if arm=='base' else contextlib.nullcontext():
        layers=frozen.assert_adapter_mode(model,enabled=arm=='finetuned')
        with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            result=model.generate(input_ids=ids,attention_mask=mask,generation_config=cfg,logits_to_keep=1,
                logits_processor=LogitsProcessorList([Finite()]),stopping_criteria=StoppingCriteriaList([Bounds()]))
    sync();seconds=time.monotonic()-started
    if result.sequences.shape[0]!=len(items) or not torch.equal(result.sequences[:,:width],ids):raise RuntimeError('Batch prompt/row identity changed')
    rows=[]
    for x,sequence in zip(items,result.sequences):
        new,reason=split_new(sequence[width:].detach().cpu().tolist(),eos,pad,min(x['remaining'],limit),stop())
        if reason=='token_limit':
            if capacity_limit is not None:reason='probe_limit'
            elif len(x['prefix'])+len(new)<x['budget']:reason='checkpoint'
        tokens=x['prefix']+new;clean=tokens[:-1] if reason=='eos' else tokens
        raw=tokenizer.decode(clean,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        r={'benchmark_id':recipe['benchmark_id'],'task_id':x['request']['task_id'],'arm':arm,'enable_thinking':False,
            **records.fields(raw),'text':tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),'completion_token_ids':tokens,'generated_tokens':len(tokens),'token_budget':x['budget'],
            'prompt_tokens':len(x['prompt']),'prompt_sha256':records.digest(x['prompt']),
            'messages_sha256':records.digest(x['request']['messages']),'request_id':request_id(recipe,x['request']),
            'new_generated_tokens':len(new),'replayed_prefix_tokens':len(x['prefix']),'segment_new_token_limit':min(x['remaining'],limit),
            'generation_input_tokens':len(x['ids']),'padded_generation_input_tokens':width,'finish_reason':reason,
            'reasoning_tokens':0,'final_answer_tokens':len(clean),'eos_tokens':int(reason=='eos'),'eos_token_id':eos,
            'seconds':seconds/len(items),'seconds_policy':'equal share of batch wall time; not individual response latency',
            'batch_wall_seconds':seconds,'batch_id':batch_id,'batch_size':len(items),'attempt_id':uuid.uuid4().hex,
            'allocation_id':os.environ.get('ARC_EVAL_ALLOCATION_ID'),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'adapter_layers_checked':layers}
        
        if capacity_limit is None:records.validate_record(r)
        rows.append(r)
    memory=[]
    if device.type=='cuda':
        for i in range(torch.cuda.device_count()):memory.append({'device':i,'peak_allocated_bytes':torch.cuda.max_memory_allocated(i),'peak_reserved_bytes':torch.cuda.max_memory_reserved(i),'total_bytes':torch.cuda.get_device_properties(i).total_memory})
    evidence={'batch_id':batch_id,'arm':arm,'actual_batch_size':len(items),'batch_size_cap':recipe['batch_size'],
        'seconds':seconds,'new_generated_tokens':sum(r['new_generated_tokens'] for r in rows),
        'tokens_per_second':sum(r['new_generated_tokens'] for r in rows)/seconds,'padded_input_width':width,
        'padding_tokens':len(items)*width-sum(len(x['ids']) for x in items),'memory':memory,
        'request_ids':[r['request_id'] for r in rows],'rows':rows,'engineering_only':capacity_limit is not None,
        'capacity_limit':capacity_limit,'forced_output_length':capacity_limit is not None,
        'admission':policy.admission_metrics(items,chunk_tokens=policy.CHUNK_TOKENS),
        'replayed_prefix_tokens':sum(len(x['prefix']) for x in items),
        'unpadded_prefill_tokens':sum(len(x['ids']) for x in items),
        'padded_prefill_tokens':len(items)*width,
        'decode_slots_including_finished_row_padding':len(items)*(result.sequences.shape[1]-width)}
    del result,ids,mask
    return evidence
