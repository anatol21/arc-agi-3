#!/usr/bin/env python3
"""Run the full ARC fit with guards around the actual Trainer operations."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

if __package__:
    from .run_checks import atomic_json, checkpoint_files, latest_complete_checkpoint, output_lock, seal_checkpoint, verify_ready
    from .ple_placement import contains, place_small_ple_buffers
    from .runtime_guardrails import install, TimeBudgetReached
    from .runtime_identity import verify_environment, verify_model_inventory
    from .export_checkpoint import finish_export
    from .cpu_compatibility import validate_preflight_report
    from .cuda_loss_context import verify_kernel_sources, validate_device_evidence
else:
    from run_checks import atomic_json, checkpoint_files, latest_complete_checkpoint, output_lock, seal_checkpoint, verify_ready
    from ple_placement import contains, place_small_ple_buffers
    from runtime_guardrails import install, TimeBudgetReached
    from runtime_identity import verify_environment, verify_model_inventory
    from export_checkpoint import finish_export
    from cpu_compatibility import validate_preflight_report
    from cuda_loss_context import verify_kernel_sources, validate_device_evidence


def check_layout(model) -> dict:
    trainable_devices = set()
    cpu_tables = []
    trainable_count = 0
    for name, parameter in model.named_parameters():
        if parameter.device.type == "meta":
            raise RuntimeError(f"Unmaterialized model parameter: {name}")
        is_table = name.endswith("ple.ple_embedding.ngram_embedding.weight")
        if is_table:
            if parameter.device.type != "cpu" or parameter.requires_grad:
                raise RuntimeError("The frozen PLE table must remain on CPU")
            cpu_tables.append(name)
        elif parameter.device.type != "cuda":
            raise RuntimeError(f"Unexpected parameter outside the GPUs: {name}")
        if parameter.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable base parameter: {name}")
            trainable_devices.add(parameter.device.index)
            trainable_count += parameter.numel()
    if not cpu_tables or trainable_devices != {0, 1, 2}:
        raise RuntimeError("Expected a CPU PLE table and trainable adapters on all three GPUs")
    layers = {0: 0, 1: 0, 2: 0}
    for module in model.modules():
        if type(module).__name__ == "Qwen4ExpTextDecoderLayer":
            device = next(module.parameters()).device
            if device.type != "cuda" or device.index not in layers:
                raise RuntimeError("A text decoder layer is not on an allocated GPU")
            layers[device.index] += 1
    if list(layers.values()) != [16, 16, 16]:
        raise RuntimeError(f"Expected 16 decoder layers per GPU, found {layers}")
    return {"decoder_layers_per_gpu": layers, "cpu_ple_tables": cpu_tables,
            "trainable_parameters": trainable_count, "trainable_devices": sorted(trainable_devices)}


def main(config_path: Path) -> None:
    # Verify code, generated inputs, and cached Arrow data before importing GPU code.
    verify_ready(config_path.parent)
    config = json.loads(config_path.read_text())
    output = Path(config["output_dir"])
    with output_lock(output):
        atomic_json(output / "run-status.json", {"phase": "loading", "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        try:
            train(config_path, config, output)
        except BaseException as error:
            previous = json.loads((output / "run-status.json").read_text())
            previous.update(phase="time_budget_exhausted" if isinstance(error, TimeBudgetReached) else "failed",
                            error=repr(error), traceback=traceback.format_exc())
            atomic_json(output / "run-status.json", previous)
            print("RUN_STOPPED " + repr(error), flush=True)
            raise


def train(config_path: Path, config: dict, output: Path) -> None:
    prepared = json.loads((config_path.parent / "preparation.json").read_text())
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != prepared["config_sha256"]:
        raise RuntimeError("Config differs from the prepared recipe")
    preflight_path = config_path.parent / "compatibility-preflight.json"
    if prepared.get("compatibility_preflight_sha256") != hashlib.sha256(preflight_path.read_bytes()).hexdigest():
        raise RuntimeError("Missing or changed CPU compatibility preflight; stage v6 before submitting")
    validate_preflight_report(json.loads(preflight_path.read_text()))
    evidence_path = config_path.parent / "gpu-device-debug-evidence.json"
    if prepared.get("gpu_device_evidence_sha256") != hashlib.sha256(evidence_path.read_bytes()).hexdigest():
        raise RuntimeError("Missing or changed reviewed GPU device diagnostic")
    evidence = json.loads(evidence_path.read_text())
    validate_device_evidence(evidence)
    actual = verify_kernel_sources(json.loads((config_path.parent / "reviewed-kernels.json").read_text()))
    if actual != evidence["kernel_source_sha256"]:
        raise RuntimeError("CCE kernels differ from the completed GPU diagnostic")
    placement = prepared.get("placement_preflight", {})
    table = placement.get("cpu_ple_parameter")
    if not table or any(contains(key, table) for key in config["device_map"]):
        raise RuntimeError("Missing corrected CPU PLE placement; do not retry the original v1 map")
    if config.get("fsdp_config") is not None or config.get("deepspeed") or config.get("ddp"):
        raise RuntimeError("This candidate must use one process with explicit model placement")
    verify_environment(json.loads((config_path.parent / "environment-verified.json").read_text()))
    verify_model_inventory(Path(config["base_model"]), json.loads((config_path.parent / "model-verified.json").read_text()))
    recipe = output / "run-recipe.json"
    if recipe.exists() and json.loads(recipe.read_text()) != prepared:
        raise RuntimeError("Existing output was created with different data or configuration")
    atomic_json(recipe, prepared)
    checkpoint = latest_complete_checkpoint(output)
    if checkpoint is not None:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if state["global_step"] == 660:
            finish_export(output, checkpoint, config_path.parent, config)
            print("COMPLETE: validated checkpoint exported without another training update", flush=True)
            return
        if state["global_step"] > 660:
            raise RuntimeError("Checkpoint step exceeds the unchanged training recipe")
    import torch
    if torch.cuda.device_count() != 3 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Expected one process seeing exactly three allocated GPUs")
    sys.argv = ["axolotl.cli.train", str(config_path)]
    os.environ.pop("ARC_RESUME_CHECKPOINT", None)
    if checkpoint is not None:
        os.environ["ARC_RESUME_CHECKPOINT"] = str(checkpoint)
        sys.argv.append("--resume_from_checkpoint=" + str(checkpoint))
        print("Resuming " + str(checkpoint), flush=True)
    guards, restore_train = install(torch, config, prepared, output, check_layout, place_small_ple_buffers)
    try:
        runpy.run_module("axolotl.cli.train", run_name="__main__")
        if len(guards) != 1:
            raise RuntimeError("Axolotl returned without running the guarded Trainer")
        guards[0].exported()
    except BaseException as error:
        if guards:
            guard = guards[0]
            report = {"error": repr(error), "traceback": traceback.format_exc(), "context": guard.context,
                      "first_bad_activation": guard.first_bad_activation, "last_valid_checkpoint": guard.last_checkpoint,
                      "training_loop_returned": guard.train_finished}
            atomic_json(output / "run-failure.json", report)
            guard.status("time_budget_exhausted" if isinstance(error, TimeBudgetReached) else "failed", **report)
        raise
    finally:
        for guard in guards:
            guard.close()
        restore_train()


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
