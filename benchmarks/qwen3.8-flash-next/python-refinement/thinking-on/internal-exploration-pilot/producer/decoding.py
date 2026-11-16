"""Row-local sampled tokens, checkpointed RNG, and a bounded reasoning channel."""
import base64
from protocol_pilot import allowance, digest


def render(tok, messages):
    opts = dict(add_generation_prompt=True, enable_thinking=True, reasoning_effort='xhigh')
    text = tok.apply_chat_template(messages, tokenize=False, **opts)
    if not text.rstrip().endswith('<think>'):
        raise ValueError('Thinking template did not open reasoning')
    return tok.apply_chat_template(messages, tokenize=True, return_dict=False, **opts)


def decode(model, tok, rows, save_row, stop, chunk=128, reasoning_cap=4096, code_mode=False):
    import torch
    from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
    close_ids = tok.encode('</think>', add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError('Unsupported thinking delimiter')
    close = close_ids[0]
    device = model.get_input_embeddings().weight.device
    eos = tok.eos_token_id
    states = [s for s, _ in rows]
    lengths = [allowance(s,p) for s,p in rows]
    inputs = [p+s['prefix'] for s,p in rows]
    width = max(map(len,inputs))
    steps = min(chunk, 18432-width, max(lengths))
    if steps <= 0:
        raise ValueError('Exhausted rows must be consumed before decoding')
    ids = torch.tensor([[tok.pad_token_id]*(width-len(x))+x for x in inputs],device=device)
    mask = torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in inputs],device=device)
    limits = [min(reasoning_cap,s['reasoning'] + max(0,(n-1)//2)) for s,n in zip(states,lengths)]
    # Freeze the allowance for the entire turn across checkpoints.
    for s,limit in zip(states,limits):
        s.setdefault('turn_reason_limit',limit)
    gens = []
    for s in states:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(digest([s['task'],len(s['attempts'])])[:15],16))
        if s['rng']:
            gen.set_state(torch.tensor(list(base64.b64decode(s['rng'])),dtype=torch.uint8))
        gens.append(gen)
    banned = []
    if code_mode:
        for tag in ('<think>', '</think>'):
            encoded = tok.encode(tag,add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError('Unsupported reasoning control token')
            banned.extend(encoded)
    ends = [None]*len(rows)
    class Pick:
        def __call__(self,input_ids,scores):
            forced = torch.full_like(scores,float('-inf'))
            for i,(s,g) in enumerate(zip(states,gens)):
                if ends[i]:
                    token = eos
                else:
                    thinking = close not in s['prefix']
                    if thinking and s['reasoning'] >= s['turn_reason_limit']:
                        token = close
                        s['forced_closes'] = s.get('forced_closes',0)+1
                    else:
                        logits = scores[i].float() / (.6 if thinking else .3)
                        if code_mode and not thinking:
                            logits[banned] = float('-inf')
                        if thinking:
                            logits[eos] = float('-inf')
                        token = int(torch.multinomial(logits.softmax(-1),1,generator=g))
                    s['prefix'].append(token)
                    s['reasoning'] += int(thinking and token != close)
                    s['used'] += 1
                    s['total'] += 1
                    if token == eos:
                        ends[i]='eos'
                    elif allowance(s,rows[i][1]) <= 0:
                        ends[i]='token_limit'
                    if close in s['prefix']:
                        body = s['prefix'][s['prefix'].index(close)+1:]
                        if code_mode and len(body)>=1536 and token!=eos:
                            ends[i]='token_limit'
                        if len(body) >= 512 and tok.decode(body[:40]).lstrip().startswith('```probe') and token != eos:
                            ends[i]='probe_limit'
                forced[i,token]=0
            return forced
    class Halt:
        def __call__(self,input_ids,scores,**kwargs):
            return torch.tensor([bool(x) or stop() or (code_mode and any(ends)) for x in ends],device=device)
    cfg=GenerationConfig(do_sample=False,max_new_tokens=steps,eos_token_id=eos,
                         pad_token_id=tok.pad_token_id,use_cache=True)
    with torch.inference_mode():
        model.generate(logits_to_keep=1,input_ids=ids,attention_mask=mask,generation_config=cfg,
                       logits_processor=LogitsProcessorList([Pick()]),stopping_criteria=StoppingCriteriaList([Halt()]))
    for i,(s,g) in enumerate(zip(states,gens)):
        s['rng']=base64.b64encode(bytes(g.get_state().tolist())).decode()
        if ends[i]:
            s['pending_end']=ends[i]
        save_row(s)
    return ends


def final_text(tok,state):
    delim = tok.encode('</think>',add_special_tokens=False)[0]
    ids=state['prefix']
    if delim not in ids:
        return ''
    return tok.decode([x for x in ids[ids.index(delim)+1:] if x!=tok.eos_token_id],
                      skip_special_tokens=False)
