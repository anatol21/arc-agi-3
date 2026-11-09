"""Four-candidate code-only synthesis; query correctness never enters refinement."""
import ast
import hashlib
import json
import re

if __package__:
    from .records import grid
else:
    from records import grid

MAX_CANDIDATES = 4
TOTAL_TOKENS = 32768
SYSTEM = '''You are an expert solver of ARC grid-transformation tasks.
Infer a single general transformation from all demonstration input-output pairs.
Implement it as deterministic standard-library Python: def solve(grid): ...
The function receives one list-of-lists input grid and returns one rectangular
list-of-lists output grid, with integer colors 0-9 and dimensions 1-30.
Consider objects, connected components, counting, filtering, relative positions,
rotation, reflection, translation, scaling, patterns, color mappings and compositions.
Input and output dimensions may differ. Avoid arbitrary example-specific cases.
Check every demonstration. On a revision,
diagnose the supplied execution failures: is the rule wrong, or is its implementation
wrong? Then revise the program and recheck all demonstrations, including prior passes.
Your final answer must be exactly one ```python code block with the complete source,
including solve(grid). Put a concise description of the rule in a docstring or comment.
Do not return output grids or JSON: the runner executes your program to obtain grids.
Only standard-library modules are available. No files, external services or hardcoded
query answers. You have at most four candidate programs, with execution feedback.'''


def message_digest(messages):
    return hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()


def extract_code(final):
    if not isinstance(final,str) or final.count('```') != 2:
        raise ValueError('Expected exactly one Python code block')
    match = re.fullmatch(r'\s*```python[ \t]*\n(.*?)\n```\s*',final,re.S)
    if not match: raise ValueError('Final answer must be a complete Python code block')
    code = match[1]
    if len(code.encode())>65536: raise ValueError('Program exceeds 64 KiB')
    return code


def validate_code(code):
    tree = ast.parse(code)
    if not any(isinstance(n,ast.FunctionDef) and n.name=='solve' for n in tree.body):
        raise ValueError('Missing top-level solve(grid) function')


def feedback(execution, demos):
    cases = []
    results = execution.get('cases',[])
    if len(results)!=len(demos): raise RuntimeError('Executor returned the wrong case count')
    for i,(case,demo) in enumerate(zip(results,demos)):
        actual=case.get('output');expected=demo['output'];error=None
        try: grid(actual)
        except (ValueError,TypeError) as failure:error=str(failure)
        item={'example':i,'executable':bool(case['executable']),'valid_grid':case['executable'] and error is None,
              'correct':case['executable'] and error is None and actual==expected}
        if case.get('error'):item['execution_error']=case['error'][:2000]
        if case.get('stdout'):item['stdout']=case['stdout'][:1000]
        if case.get('stderr'):item['stderr']=case['stderr'][:1000]
        if error:item['grid_error']=error
        if error is None and not item['correct']:
            item['actual_shape']=[len(actual),len(actual[0])]
            item['expected_shape']=[len(expected),len(expected[0])]
            differences=[]
            for y,row in enumerate(expected):
                for x,value in enumerate(row):
                    got=actual[y][x] if y<len(actual) and x<len(actual[y]) else None
                    if got!=value and len(differences)<12:
                        differences.append({'row':y,'column':x,'expected':value,'actual':got})
            item['first_mismatches']=differences
        cases.append(item)
    return {'passed':sum(c['correct'] for c in cases),'total':len(cases),
            'executed_cases':sum(c['executable'] for c in cases),'cases':cases}


def initial_state(visible):
    if (set(visible)!= {'train','test'} or not visible['train'] or not visible['test']
        or any(set(t)!= {'input'} for t in visible['test'])
        or any(set(d)!= {'input','output'} for d in visible['train'])):
        raise ValueError('Only demonstrations and answer-free query inputs are allowed')
    for pair in visible['train']:grid(pair['input']);grid(pair['output'])
    for pair in visible['test']:grid(pair['input'])
    return {'schema':2,'visible_sha256':message_digest(visible),'attempts':[], 'pending':None,
            'total_generated_tokens':0,'phase':'candidates','complete':False}


def messages_for(visible,attempts):
    messages=[{'role':'system','content':SYSTEM},
              {'role':'user','content':'Task:\n'+json.dumps(visible,ensure_ascii=False)}]
    if attempts:
        previous=attempts[-1]
        summary=[{'candidate':a['candidate'],'passed':a['feedback']['passed'],'total':a['feedback']['total']} for a in attempts]
        # Bound history by retaining the previous candidate and all compact scores,
        # not every large reasoning trace. The task itself is never truncated.
        messages += [{'role':'assistant','content':previous['response'].get('final_answer') or ''},
                     {'role':'user','content':'Execution feedback on demonstrations only:\n'+json.dumps(previous['feedback'])+
                      '\nEarlier candidate scores: '+json.dumps(summary)+
                      '\nDiagnose the rule or implementation failure, then return the complete revised Python code block.'}]
    return messages


def advance(visible,state,generate,execute,save,stop=lambda:False):
    if state['visible_sha256']!=message_digest(visible):raise RuntimeError('Task changed across resume')
    if state['complete']:return state
    if state['attempts'] and 'feedback' not in state['attempts'][-1]:
        if stop():return state
        attempt=state['attempts'][-1]
        attempt['demonstration_execution']=execute(attempt['code'],[d['input'] for d in visible['train']])
        attempt['feedback']=feedback(attempt['demonstration_execution'],visible['train'])
        if attempt['feedback']['passed']==len(visible['train']):
            state['phase']='queries';state['termination']='all_demonstrations_matched'
        save(state)
    # A crash after saving feedback but before updating phase must not refine success.
    if state['phase']=='candidates' and state['attempts'] and state['attempts'][-1].get('feedback',{}).get('passed')==len(visible['train']):
        state['phase']='queries';state['termination']='all_demonstrations_matched';save(state)
    while state['phase']=='candidates' and len(state['attempts'])<MAX_CANDIDATES:
        if stop():return state
        pending=state['pending']
        if pending is None:
            messages=messages_for(visible,state['attempts'])
            pending={'messages':messages,'messages_sha256':message_digest(messages),'prefix':[],'segments':[]}
            state['pending']=pending;save(state)
        if message_digest(pending['messages'])!=pending['messages_sha256']:raise RuntimeError('Pending conversation changed')
        response=generate(pending['messages'],pending['prefix'])
        if response['finish_reason']=='context_limit':
            state['context_exhausted_messages_sha256']=pending['messages_sha256']
            state['pending']=None;state['phase']='queries';state['termination']='context_exhausted';save(state);break
        old_prefix=list(pending['prefix'])
        tokens=response['completion_token_ids']
        if tokens[:len(old_prefix)]!=old_prefix or response['new_generated_tokens']!=len(tokens)-len(old_prefix):
            raise RuntimeError('Generation continuation does not preserve exact token prefix')
        pending['segments'].append(response)
        pending['prefix']=tokens
        state['total_generated_tokens']+=response['new_generated_tokens']
        if state['total_generated_tokens']>TOTAL_TOKENS:raise RuntimeError('Task token budget exceeded')
        if response['finish_reason'] in ('interrupted','checkpoint'):
            save(state);return state
        attempt={'candidate':len(state['attempts']),'response':response,'segments':pending['segments'],
                 'messages':pending['messages'],'messages_sha256':pending['messages_sha256']}
        # Persist generated code even if syntax checking fails.
        try:
            if response['finish_reason']!='eos':raise ValueError('Candidate reached its token cap')
            if response.get('separation_error'):raise ValueError(response['separation_error'])
            attempt['code']=extract_code(response['final_answer']);validate_code(attempt['code'])
        except (ValueError,SyntaxError,TypeError) as error:
            attempt['feedback']={'passed':0,'total':len(visible['train']),'executed_cases':0,'error':str(error)[:2000]}
        # Save candidate first; execution is recovered independently after a deadline.
        state['pending']=None;state['attempts'].append(attempt);save(state)
        if 'feedback' not in attempt:
            if stop():return state
            attempt['demonstration_execution']=execute(attempt['code'],[d['input'] for d in visible['train']])
            attempt['feedback']=feedback(attempt['demonstration_execution'],visible['train']);save(state)
        if attempt['feedback']['passed']==len(visible['train']):
            state['phase']='queries';state['termination']='all_demonstrations_matched';save(state);break
    if state['phase']=='candidates' and len(state['attempts'])==MAX_CANDIDATES:
        state['phase']='queries';state['termination']='candidate_budget_exhausted';save(state)
    if state['phase']=='queries':
        eligible=[a for a in state['attempts'] if a['feedback']['executed_cases']>0]
        selected=max(eligible,key=lambda a:(a['feedback']['passed'],-a['candidate'])) if eligible else None
        state['selected_candidate']=selected['candidate'] if selected else None
        for name,attempt in [('first',state['attempts'][0] if state['attempts'] else None),('selected',selected)]:
            key=name+'_query_execution'
            if key in state:continue
            if stop():return state
            if attempt is None or 'demonstration_execution' not in attempt:
                state[key]={'cases':[{'executable':False,'error':'No executable candidate','output':None} for _ in visible['test']]}
            elif name=='selected' and attempt['candidate']==0 and 'first_query_execution' in state:
                state[key]=state['first_query_execution']
            else:
                state[key]=execute(attempt['code'],[t['input'] for t in visible['test']])
            save(state)  # Query predictions never go into messages or selection.
        for label in ('first','selected'):
            state[label+'_output_json']={'outputs':[c.get('output') if c['executable'] else None for c in state[label+'_query_execution']['cases']]}
        state['complete']=True;state['phase']='complete';save(state)
    return state


def validate_state(visible,state):
    if state.get('schema')!=2 or state.get('visible_sha256')!=message_digest(visible):
        raise ValueError('State does not match the task')
    attempts=state['attempts']
    if len(attempts)>MAX_CANDIDATES:raise ValueError('Too many candidate programs')
    try:
        from .records import validate_record
    except ImportError:
        from records import validate_record
    seen=set();total=0
    groups=[]
    for index,attempt in enumerate(attempts):
        if attempt['candidate']!=index:raise ValueError('Candidate order changed')
        if attempt['messages_sha256']!=message_digest(attempt['messages']):raise ValueError('Candidate messages changed')
        if attempt['messages']!=messages_for(visible,attempts[:index]):raise ValueError('Candidate feedback history changed')
        if not attempt['segments'] or attempt['response']!=attempt['segments'][-1]:raise ValueError('Candidate response evidence changed')
        parsing_error=None
        try:
            if attempt['response']['finish_reason']!='eos':raise ValueError('Candidate reached its token cap')
            if attempt['response'].get('separation_error'):raise ValueError(attempt['response']['separation_error'])
            parsed=extract_code(attempt['response']['final_answer'])
            if attempt.get('code')!=parsed:raise ValueError('Saved code differs from generated answer')
            validate_code(parsed)
        except (ValueError,SyntaxError,TypeError) as failure:parsing_error=str(failure)[:2000]
        if parsing_error is not None:
            expected={'passed':0,'total':len(visible['train']),'executed_cases':0,'error':parsing_error}
            if attempt.get('feedback')!=expected or 'demonstration_execution' in attempt:raise ValueError('Invalid response was executed or failure evidence changed')
        elif 'feedback' in attempt and 'demonstration_execution' not in attempt:raise ValueError('Missing execution evidence')
        if attempt['response']['finish_reason'] not in ('eos','token_limit'):raise ValueError('Incomplete candidate marked finished')
        if 'demonstration_execution' in attempt:
            if attempt['feedback']!=feedback(attempt['demonstration_execution'],visible['train']):raise ValueError('Demo feedback changed')
        if any(r.get('task_id')!=state.get('task_id',r.get('task_id')) or r.get('arm')!=state.get('arm',r.get('arm')) or r.get('benchmark_id')!=state.get('benchmark_id',r.get('benchmark_id')) for r in attempt['segments']):raise ValueError('Generation provenance changed')
        groups.append((attempt['segments'],None))
    pending=state['pending']
    if pending:
        if pending['messages_sha256']!=message_digest(pending['messages']) or pending['messages']!=messages_for(visible,attempts):
            raise ValueError('Pending task conversation changed')
        if any(r.get('task_id')!=state.get('task_id',r.get('task_id')) or r.get('arm')!=state.get('arm',r.get('arm')) or r.get('benchmark_id')!=state.get('benchmark_id',r.get('benchmark_id')) for r in pending['segments']):raise ValueError('Pending generation provenance changed')
        groups.append((pending['segments'],pending['prefix']))
    for segments,prefix in groups:
        previous=[];prompt_length=None
        for i,r in enumerate(segments):
            validate_record(r)
            if r['finish_reason'] not in ('eos','token_limit','interrupted','checkpoint'):raise ValueError('Invalid stopping reason')
            if r['finish_reason']=='token_limit' and len(r['completion_token_ids'])!=r['token_budget']:raise ValueError('Token cap not reached')
            if r['replayed_prefix_tokens']!=len(previous) or r['generation_input_tokens']!=r['prompt_tokens']+len(previous):raise ValueError('Replay input accounting mismatch')
            if r['attempt_id'] in seen:raise ValueError('Duplicate generation segment')
            seen.add(r['attempt_id'])
            if type(r['new_generated_tokens']) is not int or r['new_generated_tokens']<1:raise ValueError('No generation progress')
            if r['completion_token_ids'][:len(previous)]!=previous or r['new_generated_tokens']!=len(r['completion_token_ids'])-len(previous):
                raise ValueError('Continuation prefix/new-token accounting mismatch')
            if i and segments[i-1]['finish_reason'] not in ('interrupted','checkpoint'):raise ValueError('Continued a completed response')
            if prompt_length is not None and prompt_length!=r['prompt_tokens']:raise ValueError('Prompt size changed during continuation')
            prompt_length=r['prompt_tokens'];previous=r['completion_token_ids'];total+=r['new_generated_tokens']
        if prefix is not None and (previous!=prefix or segments and segments[-1]['finish_reason'] not in ('interrupted','checkpoint')):
            raise ValueError('Pending prefix does not match saved interrupted tokens')
    if total!=state['total_generated_tokens'] or not 0<=total<=TOTAL_TOKENS:raise ValueError('Task token ledger mismatch')
    if state['complete']:
        if state['pending'] is not None or state['phase']!='complete':raise ValueError('Complete task has unfinished state')
        eligible=[a for a in attempts if a['feedback']['executed_cases']>0]
        best=max(eligible,key=lambda a:(a['feedback']['passed'],-a['candidate'])) if eligible else None
        if state.get('selected_candidate')!=(best['candidate'] if best else None):raise ValueError('Selection changed')
        for label in ('first','selected'):
            cases=state[label+'_query_execution']['cases']
            if len(cases)!=len(visible['test']):raise ValueError('Wrong query execution count')
            outputs={'outputs':[c.get('output') if c['executable'] else None for c in cases]}
            if state[label+'_output_json']!=outputs:raise ValueError('Saved query grids differ from execution')
    return state
