"""Private read-only bridge to the verified inference stack."""
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

NAME = 'arc-flash-cache-pilot-v1'
SOURCE = 'arc-flash-python-nothink-v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def bridge(folder):
    manifest = json.loads((folder / 'package-manifest.json').read_text())
    if folder.name != NAME or manifest['name'] != NAME:
        raise RuntimeError('Wrong pilot package')
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or sha(folder / name) != expected:
            raise RuntimeError('Pilot source changed: ' + name)
    source = folder.parent / SOURCE
    expected = json.loads((folder / 'v2-package-manifest.json').read_text())
    if json.loads((source / 'package-manifest.json').read_text()) != expected:
        raise RuntimeError('V2 package identity changed')
    for name, expected_hash in expected['files_sha256'].items():
        if sha(source / name) != expected_hash:
            raise RuntimeError('V2 source changed: ' + name)
    key = '_arc_cache_pilot_frozen_v2'
    if key not in sys.modules:
        package = types.ModuleType(key)
        package.__path__ = [str(source)]
        sys.modules[key] = package
    elif Path(sys.modules[key].__path__[0]).resolve() != source.resolve():
        raise RuntimeError('Conflicting private runtime')
    v = {name: importlib.import_module(key + '.' + name)
         for name in ('runtime', 'controller', 'records', 'telemetry')}
    _, original, modules = v['runtime'].ready(source)
    return source, original, modules, v


def ready(folder):
    source, original, modules, v = bridge(folder)
    receipt = json.loads((folder / 'READY.json').read_text())
    for name in ('recipe.json', 'references.json', 'controls.json', 'inference-config.json'):
        if receipt['files'][name] != sha(folder / name):
            raise RuntimeError('Staged input changed: ' + name)
    recipe = json.loads((folder / 'recipe.json').read_text())
    content = dict(recipe)
    identity = content.pop('benchmark_id')
    if identity != digest(content) or recipe['source_recipe_sha256'] != sha(source / 'recipe.json'):
        raise RuntimeError('Pilot provenance changed')
    if (recipe['batch_size'], recipe['segment_tokens'], recipe['segments'],
            recipe['enable_thinking'], recipe['context']) != (8, 128, 2, False, 18432):
        raise RuntimeError('Pilot bounds changed')
    return recipe, modules, v
