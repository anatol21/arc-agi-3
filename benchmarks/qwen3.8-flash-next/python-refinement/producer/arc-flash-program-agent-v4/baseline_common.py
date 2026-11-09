"""Use the frozen v6 verification helpers; never change the training package."""
import json
from pathlib import Path

if __package__:
    from .protocol import digest
    from .result_io import atomic_json
    from experiments.cluster.run_checks import checkpoint_files, output_lock, snapshot, verify_ready as verify_training_ready, verify_snapshot
    from experiments.cluster.runtime_identity import REVISION, verify_environment, verify_model_inventory
else:
    from protocol import digest
    from result_io import atomic_json
    from run_checks import checkpoint_files, output_lock, snapshot, verify_ready as verify_training_ready, verify_snapshot
    from runtime_identity import REVISION, verify_environment, verify_model_inventory

WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "arc-flash-thinking-benchmark-v2"
OUTPUT_NAME = "arc-flash-thinking-benchmark-v2"


def verify_ready(run):
    ready = json.loads((run / "READY.json").read_text())
    if ready.get("schema_version") != 2:
        raise RuntimeError("Stage the benchmark before submitting")
    workspace = Path(ready["workspace"])
    required = {str((run / name).resolve().relative_to(workspace.resolve())) for name in
                ("run_benchmark.py", "protocol.py", "debug.sbatch", "full.sbatch", "thinking-template.json", "inference-config.json",
                 "benchmark-recipe.json", "prompts.jsonl", "answers.scorer-only.jsonl", "schema-preflight.json")}
    if not required.issubset(ready["files"]):
        raise RuntimeError("Benchmark readiness is missing runtime or data files")
    verify_snapshot(workspace, ready["files"])
    return ready


def verify_package(run):
    manifest = json.loads((run / "package-manifest.json").read_text())
    if manifest["name"] != PACKAGE_NAME:
        raise RuntimeError("Unexpected benchmark package")
    for name, sha in manifest["files_sha256"].items():
        if Path(name).name != name or digest(run / name) != sha:
            raise RuntimeError(f"Benchmark package changed: {name}")
    return manifest


def verify_training_package(workspace, run):
    previous = workspace / "arc-flash-run-v6"
    expected = json.loads((run / "training-package-manifest.json").read_text())
    for name, sha in expected["files_sha256"].items():
        if Path(name).name != name or digest(previous / name) != sha:
            raise RuntimeError(f"Staged training source changed: {name}")
    verify_training_ready(previous)
    config = json.loads((previous / "arc-flash.yml").read_text())
    verify_environment(json.loads((previous / "environment-verified.json").read_text()))
    verify_model_inventory(Path(config["base_model"]), json.loads((previous / "model-verified.json").read_text()))
    return previous, config


def verify_loader_sources(run):
    if __package__:
        from experiments.cluster.cuda_loss_context import verify_kernel_sources
    else:
        from cuda_loss_context import verify_kernel_sources
    return verify_kernel_sources(json.loads((run / "reviewed-loader-sources.json").read_text()))


def verify_baseline(run):
    baseline = run.parent / "arc-flash-dev-benchmark-v2"
    expected = json.loads((run / "baseline-package-manifest.json").read_text())
    if json.loads((baseline / "package-manifest.json").read_text()) != expected:
        raise RuntimeError("Baseline package identity changed")
    for name, sha in expected["files_sha256"].items():
        if Path(name).name != name or digest(baseline / name) != sha:
            raise RuntimeError("Baseline source changed: " + name)
    ready = json.loads((baseline / "READY.json").read_text())
    if ready.get("schema_version") != 2 or Path(ready["workspace"]).resolve() != run.parent.resolve():
        raise RuntimeError("Baseline readiness does not match the workspace")
    required = {str((baseline / name).relative_to(run.parent)) for name in
                ("run_benchmark.py", "protocol.py", "benchmark.sbatch", "inference-config.json",
                 "benchmark-recipe.json", "prompts.jsonl", "answers.scorer-only.jsonl", "schema-preflight.json")}
    if not required.issubset(ready["files"]):
        raise RuntimeError("Baseline readiness is incomplete")
    verify_snapshot(run.parent, ready["files"])
    recipe = json.loads((baseline / "benchmark-recipe.json").read_text())
    if recipe["generation"]["enable_thinking"] is not False:
        raise RuntimeError("Expected the thinking-disabled baseline")
    return baseline, recipe
