#!/usr/bin/env python3
"""Stage v6 under its output lock using verified v5 identities and the completed GPU loss diagnostic."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__:
    from .run_checks import atomic_json, digest, output_lock, snapshot, verify_ready
    from .runtime_identity import extend_environment_identity, verify_environment, verify_model_inventory, REVISION
    from .cpu_compatibility import run_preflight, validate_preflight_report
    from .prepared_data_checks import check_cached_records, check_tokenized, CACHE_KEYS
    from .cuda_loss_context import verify_kernel_sources, validate_device_evidence
else:
    from run_checks import atomic_json, digest, output_lock, snapshot, verify_ready
    from runtime_identity import extend_environment_identity, verify_environment, verify_model_inventory, REVISION
    from cpu_compatibility import run_preflight, validate_preflight_report
    from prepared_data_checks import check_cached_records, check_tokenized, CACHE_KEYS
    from cuda_loss_context import verify_kernel_sources, validate_device_evidence


def check_gpu_evidence(run, workspace):
    evidence = json.loads((run / "gpu-device-debug-evidence.json").read_text())
    validate_device_evidence(evidence)
    original = workspace / "runs/arc-flash-loss-debug-v1/debug-report.json"
    if json.loads(original.read_text()) != evidence:
        raise RuntimeError("The completed cluster GPU diagnostic differs from the reviewed evidence")
    actual = verify_kernel_sources(json.loads((run / "reviewed-kernels.json").read_text()))
    if actual != evidence["kernel_source_sha256"]:
        raise RuntimeError("Installed kernels differ from the successfully scoped GPU diagnostic")
    return original


def stage(workspace, run):
    workspace, run = workspace.resolve(), run.resolve()
    if run.parent != workspace or run.name != "arc-flash-run-v6":
        raise RuntimeError("Extract arc-flash-run-v6 directly inside the existing cluster workspace")
    output = workspace / "runs/arc-flash-v6"
    # No staged-file writes, including placement reports, precede this lock.
    with output_lock(output):
        own = json.loads((run / "package-manifest.json").read_text())
        for name, expected in own["files_sha256"].items():
            if Path(name).name != name or digest(run / name) != expected:
                raise RuntimeError(f"v6 package checksum failed: {name}")
        if (run / "READY.json").exists():
            verify_ready(run)
            config = json.loads((run / "arc-flash.yml").read_text())
            verify_environment(json.loads((run / "environment-verified.json").read_text()))
            verify_model_inventory(Path(config["base_model"]), json.loads((run / "model-verified.json").read_text()))
            validate_preflight_report(json.loads((run / "compatibility-preflight.json").read_text()))
            check_gpu_evidence(run, workspace)
            print("FULL RUN ALREADY STAGED; checks passed without rewriting files", flush=True)
            return
        if (output / "run-recipe.json").exists():
            raise RuntimeError("Existing training output cannot be re-staged without its original READY record")
        previous = workspace / "arc-flash-run-v5"
        # The old package stays untouched as the executed evidence.
        expected_previous = json.loads((run / "original-v5-manifest.json").read_text())
        for name, expected in expected_previous["files_sha256"].items():
            if Path(name).name != name or digest(previous / name) != expected:
                raise RuntimeError(f"Executed v5 source changed: {name}")
        ready = verify_ready(previous)
        config = json.loads((previous / "arc-flash.yml").read_text())
        prepared = json.loads((previous / "preparation.json").read_text())
        if prepared["config_sha256"] != digest(previous / "arc-flash.yml"):
            raise RuntimeError("v5 configuration/preparation mismatch")
        table = prepared.get("placement_preflight", {}).get("cpu_ple_parameter")
        if not table or any(table == key or table.startswith(key + ".") or key == "" for key in config["device_map"]):
            raise RuntimeError("Verified CPU PLE placement is required")
        if config.get("revision_of_model") != REVISION:
            raise RuntimeError("Model revision changed")
        probe = subprocess.run([sys.executable, "-c",
            "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
            "from run_checks import output_lock; lock=output_lock(Path(sys.argv[2])); lock.__enter__()",
            str(run), str(output)], capture_output=True, timeout=15)
        if probe.returncode == 0 or b"Another job" not in probe.stderr:
            raise RuntimeError("Filesystem failed the two-process exclusive-lock check")
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                          TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1")
        from datasets import load_from_disk
        for split, field in (("train", "datasets"), ("dev", "test_datasets")):
            path = Path(config[field][0]["path"])
            if digest(path) != prepared["splits"][split]["tokenized_sha256"]:
                raise RuntimeError("Tokenized ARC data changed")
            records = check_tokenized(path, split)
            check_cached_records(load_from_disk(str(Path(config["dataset_prepared_path"]) / CACHE_KEYS[split])), records)
            print(f"VERIFIED {split}: all source answer masks, full examples and cached rows", flush=True)
        environment = extend_environment_identity(json.loads((previous / "environment-verified.json").read_text()))
        model_identity = json.loads((previous / "model-verified.json").read_text())
        verify_model_inventory(Path(config["base_model"]), model_identity)
        print("Reused v5 content-verification proof; all model file sizes/timestamps and small-file hashes match", flush=True)
        gpu_report = check_gpu_evidence(run, workspace)
        print("GPU DEVICE EVIDENCE VERIFIED: job 7849447; four scoped losses match; isolated gradient relative error 0.35%", flush=True)
        compatibility = run_preflight()
        validate_preflight_report(compatibility)
        if any(environment["source_sha256"].get(path) != sha for path, sha in compatibility["source_sha256"].items()):
            raise RuntimeError("CPU compatibility check used source different from the staged environment")
        atomic_json(run / "compatibility-preflight.json", compatibility)
        print("INSTALLED CPU LOSS CONTRACT PASSED: False/True inferred flags; four unequal-target batches; loss, gradients, update and evaluation match", flush=True)
        config["output_dir"] = str(output)
        config.pop("logging_nan_inf_filter", None)
        prepared.update(config_sha256=None, iteration="v6", run_directory=str(run), source_job="7836256",
                        gpu_device_evidence_sha256=digest(run / "gpu-device-debug-evidence.json"),
                        compatibility_preflight_sha256=digest(run / "compatibility-preflight.json"),
                        loss_normalization="explicit sum shifted CE / accumulation-window target count; Trainer accepts loss kwargs; Accelerator accumulation one",
                        source_debug_job="7849447", gpu_fit="owner-scoped CCE forward and isolated hidden gradient verified on H100; full backbone backward and optimizer update remain unproven")
        atomic_json(run / "arc-flash.yml", config)
        prepared["config_sha256"] = digest(run / "arc-flash.yml")
        atomic_json(run / "preparation.json", prepared)
        atomic_json(run / "environment-verified.json", environment)
        atomic_json(run / "model-verified.json", model_identity)
        runtime = [run / name for name in own["files_sha256"]]
        runtime += [run / name for name in ("package-manifest.json", "arc-flash.yml", "preparation.json",
                                          "environment-verified.json", "model-verified.json", "compatibility-preflight.json")]
        runtime.append(gpu_report)
        files = dict(ready["files"])
        files.update(snapshot(workspace, runtime))
        atomic_json(run / "READY.json", {"schema_version": 2, "workspace": str(workspace), "files": files})
        verify_ready(run)
    print("FULL RUN STAGED: reuse the existing environment, data, cache and corrected model placement", flush=True)
    print("No allocation submitted. The full job will validate, perform real updates, save, and continue automatically when healthy.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    stage(args.workspace, Path(__file__).resolve().parent)
