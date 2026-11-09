"""Read-only bridge to pinned v4 loader, sandbox, data and scorer."""
import hashlib
import importlib
import json
import sys
from pathlib import Path
NAME='arc-flash-python-nothink-v2'
PROGRAM='arc-flash-program-agent-v4'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bridge(folder):
    manifest=json.loads((folder/'package-manifest.json').read_text())
    if folder.name!=NAME or manifest['name']!=NAME:raise RuntimeError('Wrong package')
    for n,h in manifest['files_sha256'].items():
        if Path(n).name!=n or sha(folder/n)!=h:raise RuntimeError('Changed package: '+n)
    program=folder.parent/PROGRAM;expected=json.loads((folder/'program-package-manifest.json').read_text())
    if json.loads((program/'package-manifest.json').read_text())!=expected:raise RuntimeError('Original program identity changed')
    for n,h in expected['files_sha256'].items():
        if sha(program/n)!=h:raise RuntimeError('Original source changed: '+n)
    sys.path.insert(0,str(program));m={}
    for n in ('setup_runtime','agent','protocol','generation','configuration','guarded_train','ple_placement','checkpoint_validation','deadline','result_io','sandbox','run_checks','probe_sandbox'):
        mod=importlib.import_module(n)
        if Path(mod.__file__).resolve()!=(program/(n+'.py')).resolve():raise RuntimeError('Conflicting runtime module '+n)
        m[n]=mod
    benchmark,original=m['setup_runtime'].ready(program)
    m['probe_sandbox'].verify_probe(program)
    return program,benchmark,original,m

def ready(folder):
    program,benchmark,original,m=bridge(folder)
    receipt=json.loads((folder/'READY.json').read_text())
    for n in ('recipe.json','tasks.jsonl','inference-config.json','template-check.json'):
        if receipt['files'].get(n)!=sha(folder/n):raise RuntimeError('Restage changed input: '+n)
    recipe=json.loads((folder/'recipe.json').read_text())
    if __package__:from . import policy
    else:import policy
    if recipe.get('policy')!=policy.POLICY or recipe.get('policy_sha256')!=sha(folder/'policy.py') or recipe.get('generation_sha256')!=sha(folder/'batched.py'):raise RuntimeError('Adaptive policy changed')
    if recipe['source_benchmark_id']!=original['benchmark_id']:raise RuntimeError('Wrong source benchmark')
    return benchmark,recipe,m

def variant_recipe(recipe,batch_size):
    if batch_size not in (1,2,4,8,16,32):raise ValueError('Supported caps: 1,2,4,8,16,32')
    r=dict(recipe,batch_size=batch_size,scheduling='global length-sorted wavefront; alternate same-arm groups; 512-token checkpoints')
    r['benchmark_id']=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
    return r
