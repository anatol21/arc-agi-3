"""Cooperative task controllers: only independent generation requests are batched."""
if __package__:
    from . import controller,batched,records
else:
    import controller,batched,records

class NeedGeneration(Exception):
    def __init__(self,request):self.request=request

def request_for(task,arm,messages,prefix):
    return {'task_id':task['task_id'],'arm':arm,'messages':messages,'prefix':list(prefix)}

def drive(task,arm,state,recipe,cache,execute,save,stop):
    def generate(messages,prefix):
        req=request_for(task,arm,messages,prefix);key=batched.request_id(recipe,req)
        if key not in cache:raise NeedGeneration(req)
        response=cache[key]
        if response['finish_reason']=='context_limit':return response
        records.validate_record(response)
        if response['request_id']!=key or response['messages_sha256']!=records.digest(messages):raise RuntimeError('Wrong durable response')
        return response
    try:
        controller.advance(task['visible'],state,generate,execute,save,stop)
        controller.validate_state(task['visible'],state)
        return None
    except NeedGeneration as pending:
        controller.validate_state(task['visible'],state)
        return pending.request

def load_cache(output,recipe):
    import json
    cache={}
    for path in sorted((output/'batches').glob('*.json')):
        evidence=json.loads(path.read_text());fingerprint=evidence.pop('sha256')
        if evidence.get('engineering_only') is not False or fingerprint!=records.digest(evidence) or evidence['benchmark_id']!=recipe['benchmark_id']:raise RuntimeError('Batch evidence changed')
        if evidence['batch_id']!=path.stem:raise RuntimeError('Batch file identity changed')
        for r in evidence['rows']:
            records.validate_record(r)
            if r['benchmark_id']!=recipe['benchmark_id'] or r['batch_id']!=evidence['batch_id']:raise RuntimeError('Wrong response provenance')
            if r['request_id'] in cache:raise RuntimeError('Duplicate generation request; review before resuming')
            cache[r['request_id']]=r
    return cache
