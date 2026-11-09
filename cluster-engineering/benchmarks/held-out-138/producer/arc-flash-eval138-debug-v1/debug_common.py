"""Fixed-prefix debug orchestration around the reviewed native evaluation."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

NAME = 'arc-flash-eval138-debug-v1'
NATIVE = 'arc-flash-eval138-v1'
PREFIX = 3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected native source: ' + old[:80])
    return source.replace(old, new)


def reviewed_source(source):
    """Change orchestration only; generate_one remains byte-identical."""
    source = replace_once(source, 'output = workspace / "runs" / OUTPUT_NAME',
                          'output = workspace / "runs" / "' + NAME + '"')
    source = replace_once(source, '    with output_lock(output):',
                          '    targets = prompts[:3]\n    with output_lock(output):')
    source = replace_once(source,
        'load_predictions(predictions, {row["task_id"] for row in prompts}, repair_tail=True)',
        'load_predictions(predictions, {row["task_id"] for row in targets}, repair_tail=True)')
    if source.count('TASKS * len(ARMS)') != 3:
        raise RuntimeError('Unexpected native completion checks')
    source = source.replace('TASKS * len(ARMS)', 'len(targets) * len(ARMS)')
    source = replace_once(source, '            for prompt in prompts:',
                          '            for prompt in targets:')
    source = replace_once(source, '            cfg = load_inference_config(run / "inference-config.json")',
        '            cfg = load_inference_config(run / "inference-config.json")\n'
        '            cfg["output_dir"] = str(output / "loader")')
    return source


def verify_package(folder):
    manifest = json.loads((folder / 'package-manifest.json').read_text())
    if folder.name != NAME or manifest['name'] != NAME:
        raise RuntimeError('Unexpected debug package')
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise RuntimeError('Debug package changed: ' + name)
    return manifest


def dependencies(folder, require_ready=True):
    folder = Path(folder).resolve()
    own = verify_package(folder)
    native = folder.parent / NATIVE
    if digest(native / 'package-manifest.json') != own['native_manifest_sha256']:
        raise RuntimeError('Native evaluation package identity changed')
    sys.path.insert(0, str(native))
    modules = {}
    for name in ('common', 'protocol', 'configuration', 'deadline'):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != native / (name + '.py'):
            raise RuntimeError('Conflicting native helper import: ' + name)
        modules[name] = module
    modules['common'].verify_package(native)
    modules['common'].verify_ready(native)
    source = reviewed_source((native / 'run_benchmark.py').read_text())
    if (folder / 'debug_runner.py').read_text() != source:
        raise RuntimeError('Debug runner differs beyond reviewed orchestration changes')
    prompts = modules['protocol'].rows(native / 'prompts.jsonl')
    expected = json.loads((folder / 'prefix-task-ids.json').read_text())
    if len(prompts) != 138 or [p['task_id'] for p in prompts[:PREFIX]] != expected:
        raise RuntimeError('Debug tasks are not the fixed first three evaluation tasks')
    recipe = json.loads((native / 'benchmark-recipe.json').read_text())
    if recipe['tasks'] != 138 or recipe['global_step'] != 660:
        raise RuntimeError('Unexpected parent benchmark')
    generation = recipe['generation']
    for key, value in dict(do_sample=False, num_beams=1, num_return_sequences=1,
                           max_new_tokens=8192, context=18432, enable_thinking=False, seed=42).items():
        if generation[key] != value:
            raise RuntimeError('Native generation condition changed: ' + key)
    receipt = dict(schema=1, package_manifest_sha256=digest(folder/'package-manifest.json'),
        native_ready_sha256=digest(native/'READY.json'),
        recipe_sha256=digest(native/'benchmark-recipe.json'), benchmark_id=recipe['benchmark_id'],
        target_task_ids=expected, target_query_grids=sum(p['query_count'] for p in prompts[:PREFIX]),
        output_name=NAME, full_output_name=NATIVE,
        report_scope='fixed-prefix pilot, not full 138-task validation accuracy')
    if require_ready and json.loads((folder/'READY.json').read_text()) != receipt:
        raise RuntimeError('Debug staging changed; stage and verify before submission')
    return native, recipe, modules, receipt
