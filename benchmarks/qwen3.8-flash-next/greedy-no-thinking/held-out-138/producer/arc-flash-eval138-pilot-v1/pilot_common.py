"""Selected diagnostic pilot orchestration around the reviewed native evaluation."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

NAME = 'arc-flash-eval138-pilot-v1'
NATIVE = 'arc-flash-eval138-v1'
PREFIX = 10


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
                          '    target_ids = json.loads((workspace / \"arc-flash-eval138-pilot-v1/task-ids.json\").read_text())\n'
                          '    by_id = {p[\"task_id\"]: p for p in prompts}\n'
                          '    targets = [by_id[task] for task in target_ids]\n    with output_lock(output):')
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
        raise RuntimeError('Unexpected pilot package')
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise RuntimeError('Pilot package changed: ' + name)
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
    if (folder / 'pilot_runner.py').read_text() != source:
        raise RuntimeError('Pilot runner differs beyond reviewed orchestration changes')
    prompts = modules['protocol'].rows(native / 'prompts.jsonl')
    expected = json.loads((folder / 'task-ids.json').read_text())
    selection = json.loads((folder/'selection.json').read_text())
    groups = selection['cohorts']
    prefix = groups['ordered_prefix']
    controls = groups['prior_api_success']
    if (len(prompts) != 138 or [p['task_id'] for p in prompts[:10]] != prefix
        or len(controls) != 2 or set(prefix) & set(controls)
        or expected != controls + prefix or len(set(expected)) != 12
        or not set(expected) <= {p['task_id'] for p in prompts}):
        raise RuntimeError('Pilot schedule/cohorts differ from the fixed selection')
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
        target_task_ids=expected, target_query_grids=sum(p['query_count'] for p in prompts if p['task_id'] in set(expected)),
        output_name=NAME, full_output_name=NATIVE, cohorts=groups,
        report_scope='selected diagnostic pilot; ordered prefix and prior-API successes reported separately')
    if require_ready and json.loads((folder/'READY.json').read_text()) != receipt:
        raise RuntimeError('Pilot staging changed; stage and verify before submission')
    return native, recipe, modules, receipt
