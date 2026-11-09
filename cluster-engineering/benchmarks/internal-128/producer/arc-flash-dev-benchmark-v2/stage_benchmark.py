#!/usr/bin/env python3
"""CPU-only verification, inspection and preparation of the ARC dev benchmark."""
import argparse
import json
import os
from pathlib import Path

if __package__:
    from .common import (WORKSPACE, PACKAGE_NAME, OUTPUT_NAME, atomic_json, output_lock, snapshot,
                         verify_ready, verify_package, verify_training_package, verify_loader_sources)
    from .protocol import digest, split_dev, write_rows, TASKS, CONTEXT, MAX_NEW_TOKENS, ARMS
    from .inspect_run import inspect_complete_run
    from .configuration import schema_preflight
else:
    from common import (WORKSPACE, PACKAGE_NAME, OUTPUT_NAME, atomic_json, output_lock, snapshot,
                        verify_ready, verify_package, verify_training_package, verify_loader_sources)
    from protocol import digest, split_dev, write_rows, TASKS, CONTEXT, MAX_NEW_TOKENS, ARMS
    from inspect_run import inspect_complete_run
    from configuration import schema_preflight


def stage(workspace, run):
    workspace, run = Path(workspace).resolve(), Path(run).resolve()
    if run.parent != workspace or run.name != PACKAGE_NAME:
        raise RuntimeError("Extract the benchmark directly in the existing cluster workspace")
    output = workspace / "runs" / OUTPUT_NAME
    with output_lock(output):
        own = verify_package(run)
        previous, training_config = verify_training_package(workspace, run)
        loader_sources = verify_loader_sources(run)
        inspection = inspect_complete_run(workspace)
        if (run / "READY.json").exists():
            verify_ready(run)
            print("BENCHMARK ALREADY STAGED; verified unchanged inputs without rewriting", flush=True)
            return
        if (output / "benchmark-recipe.json").exists():
            raise RuntimeError("Existing predictions require their original benchmark READY record")
        data = workspace / "arc-agi3-data-v1"
        manifest_path = data / "manifest.json"
        expected = json.loads((run / "data-manifest.json").read_text())
        local_manifest = json.loads(manifest_path.read_text())
        local_manifest.pop("cluster_workspace", None)
        if local_manifest != expected:
            raise RuntimeError("ARC data manifest differs from the reviewed split")
        data_paths = [manifest_path]
        for name in ("sft/dev.jsonl", "data/dev.jsonl"):
            path = data / name
            if digest(path) != expected["files_sha256"][name]:
                raise RuntimeError(f"Development data changed: {name}")
            data_paths.append(path)
        prompts, answers = split_dev(data / "sft/dev.jsonl", data / "data/dev.jsonl", expected)
        write_rows(run / "prompts.jsonl", prompts)
        write_rows(run / "answers.scorer-only.jsonl", answers)
        checkpoint = Path(inspection["checkpoint"])
        inference = dict(training_config)
        inference.update(lora_model_dir=str(checkpoint), tokenizer_config=str(checkpoint),
                         output_dir=str(output / "loader"), gradient_checkpointing=False,
                         report_to=[], wandb_project=None, wandb_entity=None)
        # Keep model/quantization/placement/patches from the successfully executed recipe.
        atomic_json(run / "inference-config.json", inference)
        config_check = schema_preflight(run / "inference-config.json")
        atomic_json(run / "schema-preflight.json", config_check)
        print("INSTALLED CONFIG SCHEMA PASSED: string config path; no model or GPU loaded", flush=True)
        atomic_json(run / "training-inspection.json", inspection)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise RuntimeError("Completed checkpoint lacks its actual EOS/PAD token settings")
        lengths = []
        for prompt in prompts:
            token_ids = tokenizer.apply_chat_template(prompt["messages"], tokenize=True, return_dict=False,
                                                      add_generation_prompt=True, enable_thinking=False)
            if not token_ids or len(token_ids) >= CONTEXT:
                raise RuntimeError("A complete prompt exhausts the context; no truncation permitted")
            lengths.append(len(token_ids))
        recipe = {"schema_version": 1, "split": "dev", "tasks": TASKS,
                  "query_grids": sum(prompt["query_count"] for prompt in prompts),
                  "arms": list(ARMS), "checkpoint": str(checkpoint), "global_step": 660,
                  "adapter_sha256": inspection["adapter_sha256"],
                  "prompts_sha256": digest(run / "prompts.jsonl"),
                  "answers_sha256": digest(run / "answers.scorer-only.jsonl"),
                  "prompt_lengths": {"min": min(lengths), "max": max(lengths), "total": sum(lengths)},
                  "generation": {"do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                                 "max_new_tokens": MAX_NEW_TOKENS, "use_cache": True,
                                 "enable_thinking": False, "context": CONTEXT, "seed": 42,
                                 "budget_policy": "min(max_new_tokens, context minus full prompt length), identical for both arms"},
                  "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
                  "metrics": "strict task accuracy and exact-grid accuracy; invalid formats count as wrong; no answer repair",
                  "baseline": "same loaded quantized model/tokenizer/prompt/decoding; PEFT adapter disabled only",
                  "loader_source_sha256": loader_sources,
                  "optimizer_updates": 0, "ttt": False, "protected_evaluation_used": False}
        atomic_json(run / "benchmark-recipe.json", recipe)
        recipe["benchmark_id"] = digest(run / "benchmark-recipe.json")
        atomic_json(run / "benchmark-recipe.json", recipe)
        files = [run / name for name in own["files_sha256"]]
        files += [run / name for name in ("package-manifest.json", "inference-config.json", "training-inspection.json",
                                         "prompts.jsonl", "answers.scorer-only.jsonl", "benchmark-recipe.json", "schema-preflight.json")]
        # Training readiness already pins the model/data; bind adapter/export artifacts as well.
        files += [previous / "READY.json", previous / "arc-flash.yml", previous / "environment-verified.json",
                  previous / "model-verified.json"]
        files += data_paths
        files += [checkpoint / name for name in ("checkpoint-complete.json", "numerical-validation.json",
                    "adapter_config.json", "adapter_model.safetensors", "trainer_state.json", "tokenizer.json", "tokenizer_config.json")]
        files += [workspace / "runs/arc-flash-v6" / name for name in ("run-status.json", "export-validation.json")]
        atomic_json(run / "READY.json", {"schema_version": 2, "workspace": str(workspace),
                                         "files": snapshot(workspace, files)})
        verify_ready(run)
        atomic_json(output / "benchmark-status.json", {"phase": "staged", "benchmark_id": recipe["benchmark_id"],
                                                     "tasks": TASKS, "query_grids": recipe["query_grids"]})
    print("TRAINING INSPECTED " + json.dumps(inspection), flush=True)
    print(f"BENCHMARK STAGED: {TASKS} dev tasks, {recipe['query_grids']} test grids, matched finetuned/base greedy generation", flush=True)
    print("No GPU job submitted. Training files remain unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1")
    stage(args.workspace, Path(__file__).resolve().parent)
