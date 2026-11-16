#!/usr/bin/env python3
"""Build answer-only tokenized ARC data and a single-process, three-GPU candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
CONTEXT = 18432


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(rows: list[dict], tokenizer, limit: int = CONTEXT) -> list[tuple[str, dict]]:
    converted = []
    for row in rows:
        messages = row["messages"]
        if [message["role"] for message in messages] != ["user", "assistant"]:
            raise ValueError(f"Unexpected message roles: {row['task_id']}")
        json.loads(messages[1]["content"])
        full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                             enable_thinking=False)
        prefix = tokenizer.apply_chat_template(messages[:1], tokenize=True, return_dict=False,
                                               enable_thinking=False, add_generation_prompt=True)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError(f"Prompt alignment failed: {row['task_id']}")
        if len(full) > limit:
            raise ValueError(f"Complete task exceeds context; refusing truncation: {row['task_id']}")
        if tokenizer.eos_token_id not in full[len(prefix):]:
            raise ValueError(f"Assistant end token missing: {row['task_id']}")
        converted.append((row["task_id"], {"input_ids": full, "attention_mask": [1] * len(full),
                                         "labels": [-100] * len(prefix) + full[len(prefix):]}))
    return converted


def configuration(workspace: Path, run: Path, pad_token: str) -> dict:
    # An explicit map avoids automatic placement's bf16 estimates for fused MoE weights.
    placement = {"": 0, "lm_head": 2, "model.language_model.hyper_connection_mixer": 2}
    placement.update({f"model.language_model.layers.{i}": i // 16 for i in range(48)})
    return {
        "base_model": str(workspace / "models/qwen3.8-flash-next"),
        "revision_of_model": REVISION,
        "plugins": ["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"],
        "adapter": "qlora", "load_in_4bit": True, "load_in_8bit": False,
        "quantize_moe_experts": True, "ple_cpu_offload": True,
        "device_map": placement, "ddp": False,
        "datasets": [{"path": str(run / "tokenized/train.jsonl"), "ds_type": "json", "type": None}],
        "test_datasets": [{"path": str(run / "tokenized/dev.jsonl"), "ds_type": "json", "type": None}],
        "dataset_prepared_path": str(run / "prepared"), "dataset_num_proc": 1,
        "shuffle_merged_datasets": False, "shuffle_before_merging_datasets": False,
        "val_set_size": 0, "output_dir": str(workspace / "runs/arc-flash-v1"),
        "sequence_len": CONTEXT, "eval_sequence_len": CONTEXT,
        "excess_length_strategy": "raise", "sample_packing": False,
        "eval_sample_packing": False, "pad_to_sequence_len": False, "batch_flattening": False,
        "train_on_inputs": False, "chat_template_kwargs": {"enable_thinking": False},
        "special_tokens": {"pad_token": pad_token},
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0,
        # Routed experts stay quantized/frozen. Avoid billions of expert-adapter parameters
        # for this small ARC dataset. The same adapter defines fixed-weight and TTT arms.
        "lora_target_modules": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
            "linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.out_proj",
            "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert.gate_proj",
            "q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_mlp_kernel": False, "lora_qkv_kernel": False, "lora_o_kernel": False,
        "freeze_mm_modules": True, "micro_batch_size": 1, "eval_batch_size": 1,
        "gradient_accumulation_steps": 4, "num_epochs": 3,
        "optimizer": "adamw_bnb_8bit", "lr_scheduler": "cosine", "learning_rate": 5e-5,
        "warmup_ratio": 0.03, "max_grad_norm": 1.0, "bf16": True, "tf32": False,
        "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "attn_implementation": "sdpa", "curriculum_sampling": True,
        "dataloader_num_workers": 0, "dataloader_drop_last": False,
        "logging_steps": 1, "logging_nan_inf_filter": False,
        "save_first_step": True, "save_steps": 25, "save_total_limit": 3,
        "save_only_model": False, "save_safetensors": True,
        "eval_steps": 100, "seed": 42,
    }


def prepare(workspace: Path, run: Path) -> dict:
    from transformers import AutoTokenizer
    workspace = workspace.resolve()
    run = run.resolve()
    pin = (workspace / "logs/qwen-flash-revision.txt").read_text().strip()
    if pin != REVISION:
        raise ValueError("Model revision changed; refusing to reuse this run recipe")
    data = workspace / "arc-agi3-data-v1"
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["counts"] != {"train": 880, "dev": 128, "evaluation": 138} or manifest["external_synthetic_tasks"] != 0:
        raise ValueError("Data manifest differs from the confirmed ARC-only package")
    verified = {"train_dev_checksums": "checked below", "external_synthetic_tasks": 0}
    tokenizer = AutoTokenizer.from_pretrained(workspace / "models/qwen3.8-flash-next",
                                             local_files_only=True)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no end token; do not invent a new vocabulary entry")
    stats = {"revision": pin, "sequence_len": CONTEXT, "source_verified": verified,
             "splits": {}, "first_task_is_longest": True,
             "parallelism": "one process; explicit 16/16/16 decoder layers across three GPUs",
             "gpu_fit": "not yet tested"}
    for split, count, expected_tokens in (("train", 880, 2772080), ("dev", 128, 342610)):
        source = data / f"sft/{split}.jsonl"
        if digest(source) != manifest["files_sha256"][f"sft/{split}.jsonl"]:
            raise ValueError(f"{split} source checksum changed")
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        if len(rows) != count:
            raise ValueError(f"Unexpected {split} task count")
        if [row["task_id"] for row in rows] != manifest["task_ids"][split]:
            raise ValueError(f"Unexpected {split} task IDs")
        converted = convert(rows, tokenizer)
        if sum(len(record["input_ids"]) for _, record in converted) != expected_tokens:
            raise ValueError(f"{split} token count differs from the confirmed audit")
        converted.sort(key=lambda item: hashlib.sha256(f"42:{item[0]}".encode()).hexdigest())
        if split == "train":
            longest = max(range(len(converted)), key=lambda i: len(converted[i][1]["input_ids"]))
            converted.insert(0, converted.pop(longest))
        output = run / f"tokenized/{split}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        with temporary.open("w") as stream:
            for _, record in converted:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        temporary.replace(output)
        stats["splits"][split] = {"tasks": count, "total_tokens": expected_tokens,
            "max_tokens": max(len(record["input_ids"]) for _, record in converted),
            "supervised_tokens": sum(sum(label != -100 for label in record["labels"]) for _, record in converted),
            "source_sha256": digest(source), "tokenized_sha256": digest(output),
            "task_order": [task_id for task_id, _ in converted]}
    verified["train_dev_checksums"] = "passed"
    config = configuration(workspace, run, tokenizer.pad_token or tokenizer.eos_token)
    # JSON is a YAML subset, understood by Axolotl's YAML loader.
    (run / "arc-flash.yml").write_text(json.dumps(config, indent=2) + "\n")
    stats["config_sha256"] = digest(run / "arc-flash.yml")
    (run / "preparation.json").write_text(json.dumps(stats, indent=2) + "\n")
    return {"prepared": True, "train_tasks": 880, "dev_tasks": 128, "sequence_len": CONTEXT,
            "train_tokens_per_epoch": 2772080, "train_tokens_three_epochs": 8316240,
            "config": str(run / "arc-flash.yml"), "gpu_fit": "not yet tested"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path,
                        default=Path.cwd())
    parser.add_argument("--run", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.run), indent=2))
