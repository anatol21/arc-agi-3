"""Verify this package, its frozen loader helpers, and the original benchmark."""
import hashlib
import json
from pathlib import Path

if __package__:
    from . import baseline_common
    from .result_io import atomic_json,atomic_text,results_lock,utcnow
else:
    import baseline_common
    from result_io import atomic_json,atomic_text,results_lock,utcnow

NAME='arc-flash-program-agent-v4'
WORKSPACE=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_dependencies(run):
    run=Path(run).resolve()
    manifest=json.loads((run/'package-manifest.json').read_text())
    if manifest['name']!=NAME or run.name!=NAME:raise RuntimeError('Wrong program benchmark package')
    for name,sha in manifest['files_sha256'].items():
        if Path(name).name!=name or digest(run/name)!=sha:raise RuntimeError('Changed package: '+name)
    benchmark,recipe=baseline_common.verify_baseline(run)
    baseline_common.verify_training_package(run.parent,run)
    baseline_common.verify_loader_sources(run)
    return benchmark,recipe


def ready(run):
    benchmark,original=verified_dependencies(run)
    receipt=json.loads((run/'READY.json').read_text())
    required={'recipe.json','prompts.jsonl','inference-config.json','thinking-template.json','package-manifest.json'}
    if not required.issubset(receipt['files']):raise RuntimeError('Incomplete staged program package')
    for name,sha in receipt['files'].items():
        if Path(name).name!=name or digest(run/name)!=sha:raise RuntimeError('Restage changed input: '+name)
    recipe=json.loads((run/'recipe.json').read_text())
    if recipe['source_benchmark_id']!=original['benchmark_id']:raise RuntimeError('Baseline identity changed')
    return benchmark,recipe
