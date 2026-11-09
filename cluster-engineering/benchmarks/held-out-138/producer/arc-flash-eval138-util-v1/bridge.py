"""Integrity-pinned access to the already staged native evaluation; no GPU imports."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

NAME = 'arc-flash-eval138-util-v1'
PARENT = 'arc-flash-eval138-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def dependencies(folder, *, require_ready=True):
    folder = Path(folder).resolve()
    own = json.loads((folder / 'package-manifest.json').read_text())
    if folder.name != NAME or own['name'] != NAME:
        raise RuntimeError('Wrong utilization package')
    for name, expected in own['files_sha256'].items():
        if Path(name).name != name or sha(folder / name) != expected:
            raise RuntimeError('Utilization source changed: ' + name)
    parent = folder.parent / PARENT
    if sha(parent / 'package-manifest.json') != own['parent_manifest_sha256']:
        raise RuntimeError('Native evaluation package changed')
    sys.path.insert(0, str(parent))
    modules = {}
    for name in ('common', 'protocol', 'configuration', 'run_benchmark',
                 'guarded_train', 'ple_placement', 'checkpoint_validation', 'deadline'):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != parent / (name + '.py'):
            raise RuntimeError('Conflicting native module: ' + name)
        modules[name] = module
    modules['common'].verify_package(parent)
    modules['common'].verify_ready(parent)
    native = json.loads((parent / 'benchmark-recipe.json').read_text())
    expected = dict(do_sample=False, num_beams=1, num_return_sequences=1,
                    max_new_tokens=8192, context=18432, enable_thinking=False, seed=42)
    if native['tasks'] != 138 or native['query_grids'] != 185 or native['global_step'] != 660:
        raise RuntimeError('Wrong parent evaluation')
    if any(native['generation'].get(k) != v for k, v in expected.items()):
        raise RuntimeError('Parent decoding conditions changed')
    prompts = modules['protocol'].rows(parent / 'prompts.jsonl')
    if len(prompts) != 138 or len({p['task_id'] for p in prompts}) != 138:
        raise RuntimeError('Wrong prompt count or duplicate tasks')
    if any(set(p) != {'task_id', 'messages', 'query_count'} or
           len(p['messages']) != 1 or set(p['messages'][0]) != {'role', 'content'} or
           p['messages'][0]['role'] != 'user' for p in prompts):
        raise RuntimeError('Unexpected prompt fields or roles')
    if require_ready:
        receipt = json.loads((folder / 'READY.json').read_text())
        if (receipt['parent_ready_sha256'] != sha(parent / 'READY.json') or
                receipt['own_manifest_sha256'] != sha(folder / 'package-manifest.json')):
            raise RuntimeError('Staged dependency identity changed')
        for name, expected in receipt['files_sha256'].items():
            if Path(name).name != name or sha(folder / name) != expected:
                raise RuntimeError('Staged utilization input changed: ' + name)
        recipe = json.loads((folder / 'recipe.json').read_text())
        unsigned = {k: v for k, v in recipe.items() if k != 'benchmark_id'}
        if recipe['benchmark_id'] != digest(unsigned):
            raise RuntimeError('Invalid recipe identity')
    else:
        recipe = None
    return parent, native, prompts, modules, recipe


def output(folder):
    return Path(folder).resolve().parent / 'runs' / NAME
