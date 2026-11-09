#!/usr/bin/env python3
"""Generate on sanitized dev prompts only; no scorer answers or optimizer are loaded."""
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

if __package__:
    from .common import (WORKSPACE, OUTPUT_NAME, atomic_json, output_lock, verify_ready,
                         verify_package, verify_training_package, verify_loader_sources)
    from .protocol import ARMS, TASKS, CONTEXT, MAX_NEW_TOKENS, rows, append_record, load_predictions
    from .configuration import load_inference_config
else:
    from common import (WORKSPACE, OUTPUT_NAME, atomic_json, output_lock, verify_ready,
                        verify_package, verify_training_package, verify_loader_sources)
    from protocol import ARMS, TASKS, CONTEXT, MAX_NEW_TOKENS, rows, append_record, load_predictions
    from configuration import load_inference_config


def assert_adapter_mode(model, enabled):
    layers = [module for module in model.modules() if hasattr(module, "lora_A") and hasattr(module, "disable_adapters")]
    if not layers or any(bool(module.disable_adapters) == enabled or getattr(module, "merged_adapters", []) for module in layers):
        raise RuntimeError("Loaded adapter is absent, merged, or not in the requested enabled/disabled state")
    return len(layers)


def clear_generation_state(model):
    # generate receives no past_key_values and uses its default fresh dynamic cache.
    for module in model.modules():
        if hasattr(module, "rope_deltas"):
            module.rope_deltas = None


def generate_one(torch, model, tokenizer, prompt, arm, recipe, stop):
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
    token_ids = tokenizer.apply_chat_template(prompt["messages"], tokenize=True, return_dict=False,
                                              add_generation_prompt=True, enable_thinking=False)
    budget = min(MAX_NEW_TOKENS, CONTEXT - len(token_ids))
    if budget <= 0:
        raise RuntimeError("Complete prompt exceeds context; refusing truncation")
    if any(message["role"] != "user" for message in prompt["messages"]):
        raise RuntimeError("Generation accepts only the staged user prompt")
    input_device = model.get_input_embeddings().weight.device
    inputs = torch.tensor([token_ids], dtype=torch.long, device=input_device)
    config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
        max_new_tokens=budget, use_cache=True, cache_implementation=None,
        eos_token_id=recipe["eos_token_id"], pad_token_id=recipe["pad_token_id"],
        bos_token_id=tokenizer.bos_token_id, return_dict_in_generate=True,
        output_scores=False, output_hidden_states=False, output_attentions=False)

    class FiniteLogits(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if not torch.isfinite(scores).all().item():
                raise RuntimeError(f"Nonfinite greedy-generation logits: {prompt['task_id']} / {arm}")
            return scores

    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), stop(), dtype=torch.bool, device=input_ids.device)

    clear_generation_state(model)
    context = model.disable_adapter() if arm == "base" else contextlib.nullcontext()
    started = time.monotonic()
    with context:
        layers = assert_adapter_mode(model, enabled=arm == "finetuned")
        # Training's k-bit preparation can leave FP32 hidden states and a BF16 head.
        # Autocast matches the successful Trainer compute context on every CUDA device.
        with torch.inference_mode(), torch.autocast(input_device.type, dtype=torch.bfloat16, enabled=input_device.type == "cuda"):
            generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                generation_config=config, logits_to_keep=1,
                logits_processor=LogitsProcessorList([FiniteLogits()]),
                stopping_criteria=StoppingCriteriaList([Deadline()]))
    sequence = generated.sequences[0]
    if sequence.shape[0] < len(token_ids) or not torch.equal(sequence[:len(token_ids)], inputs[0]):
        raise RuntimeError("Generation did not preserve the complete input prompt")
    completion = sequence[len(token_ids):].detach().cpu().tolist()
    ended = bool(completion and completion[-1] == recipe["eos_token_id"])
    interrupted = stop() and not ended and len(completion) < budget
    if not ended and len(completion) < budget and not interrupted:
        raise RuntimeError("Generation ended without EOS, token cap or the configured deadline")
    return {"benchmark_id": recipe["benchmark_id"], "task_id": prompt["task_id"], "arm": arm,
            "text": tokenizer.decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "prompt_tokens": len(token_ids), "generated_tokens": len(completion), "token_budget": budget,
            "finish_reason": "interrupted" if interrupted else "eos" if ended else "token_limit",
            "seconds": time.monotonic() - started, "adapter_layers_checked": layers}


def run_benchmark(run):
    run = Path(run).resolve()
    workspace = run.parent
    output = workspace / "runs" / "arc-flash-eval138-debug-v1"
    verify_package(run)
    verify_ready(run)
    verify_training_package(workspace, run)
    verify_loader_sources(run)
    recipe = json.loads((run / "benchmark-recipe.json").read_text())
    prompts = rows(run / "prompts.jsonl")  # Expected outputs are never parsed or supplied to the model.
    if len(prompts) != TASKS or any(set(row) != {"task_id", "messages", "query_count"} for row in prompts):
        raise RuntimeError("Unexpected generation input schema or count")
    targets = prompts[:3]
    with output_lock(output):
        recipe_path = output / "benchmark-recipe.json"
        if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
            raise RuntimeError("Existing benchmark output belongs to a different checkpoint or protocol")
        atomic_json(recipe_path, recipe)
        predictions = output / "predictions.jsonl"
        completed = load_predictions(predictions, {row["task_id"] for row in targets}, repair_tail=True)
        if any(row.get("benchmark_id") != recipe["benchmark_id"] for row in completed.values()):
            raise RuntimeError("Prediction provenance differs from this benchmark")
        if len(completed) == len(targets) * len(ARMS):
            print("All matched predictions already exist; scoring without loading GPUs", flush=True)
            return
        status = {"phase": "loading", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "benchmark_id": recipe["benchmark_id"], "completed_predictions": len(completed),
                  "required_predictions": len(targets) * len(ARMS)}
        atomic_json(output / "benchmark-status.json", status)
        stop_requested = []
        stop_at = float(os.environ.get("ARC_BENCH_STOP_AT", time.time() + 5.5 * 3600))
        previous_handlers = {}
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            previous_handlers[number] = signal.getsignal(number)
            signal.signal(number, lambda signum, frame: stop_requested.append(signum))
        def stop():
            return bool(stop_requested) or time.time() >= stop_at
        try:
            import torch
            if torch.cuda.device_count() != 3 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
                raise RuntimeError("Benchmark requires one process and the same three-GPU layout as v6")
            from axolotl.cli.utils import load_model_and_tokenizer
            if __package__:
                from experiments.cluster.guarded_train import check_layout
                from experiments.cluster.ple_placement import place_small_ple_buffers
                from experiments.cluster.checkpoint_validation import validate_adapters
            else:
                from guarded_train import check_layout
                from ple_placement import place_small_ple_buffers
                from checkpoint_validation import validate_adapters
            torch.manual_seed(recipe["generation"]["seed"])
            torch.cuda.manual_seed_all(recipe["generation"]["seed"])
            cfg = load_inference_config(run / "inference-config.json")
            cfg["output_dir"] = str(output / "loader")
            # Use the already successful k-bit preparation/patch path, then freeze it.
            # No Trainer, optimizer, backward or merge-and-unload is invoked here.
            model, tokenizer, _ = load_model_and_tokenizer(cfg=cfg, inference=False)
            place_small_ple_buffers(model)
            layout = check_layout(model)
            if str(model.get_input_embeddings().weight.device) != "cuda:0":
                raise RuntimeError("Input embeddings differ from the verified GPU-0 placement")
            if layout["trainable_parameters"] != 33478656:
                raise RuntimeError("Reloaded adapter coverage changed")
            adapter = validate_adapters(torch, Path(recipe["checkpoint"]), model)
            if tokenizer.eos_token_id != recipe["eos_token_id"] or tokenizer.pad_token_id != recipe["pad_token_id"]:
                raise RuntimeError("Reloaded tokenizer differs from the staged training tokenizer")
            model.requires_grad_(False)
            model.eval()
            model.gradient_checkpointing_disable()
            for module in model.modules():
                config = getattr(module, "config", None)
                if config is not None and hasattr(config, "use_cache"):
                    config.use_cache = True
            atomic_json(output / "model-reload-validation.json", {"layout": layout, "adapter": adapter,
                "requires_grad_parameters": sum(parameter.requires_grad for parameter in model.parameters()),
                "optimizer_updates": 0, "baseline": "same quantized base with adapters disabled"})
            status["phase"] = "generating"
            for prompt in targets:
                for arm in ARMS:
                    if (prompt["task_id"], arm) in completed:
                        continue
                    if stop():
                        break
                    status.update(task_id=prompt["task_id"], arm=arm)
                    atomic_json(output / "benchmark-status.json", status)
                    record = generate_one(torch, model, tokenizer, prompt, arm, recipe, stop)
                    if record["finish_reason"] == "interrupted":
                        append_record(output / "interrupted-attempts.jsonl", record)
                        break
                    append_record(predictions, record)
                    completed[(prompt["task_id"], arm)] = record
                    status["completed_predictions"] = len(completed)
                    print("ARC_PREDICTION " + json.dumps({key: record[key] for key in
                        ("task_id", "arm", "prompt_tokens", "generated_tokens", "finish_reason", "seconds")}), flush=True)
                if stop():
                    break
            status.update(phase="generated" if len(completed) == len(targets) * len(ARMS) else "partial_checkpointed",
                          completed_predictions=len(completed))
            atomic_json(output / "benchmark-status.json", status)
        except BaseException as error:
            status.update(phase="failed", error=repr(error), traceback=traceback.format_exc())
            atomic_json(output / "benchmark-status.json", status)
            raise
        finally:
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)


if __name__ == "__main__":
    run = Path(__file__).resolve().parent
    try:
        run_benchmark(run)
    finally:
        # A separate CPU process loads scorer answers only after generation stops.
        if (run / "benchmark-recipe.json").exists():
            result = subprocess.run([sys.executable, str(run / "score_benchmark.py")], check=False)
            if result.returncode:
                print("SCORING_FAILED: inspect score_benchmark.py output", file=sys.stderr, flush=True)
                if sys.exc_info()[0] is None:
                    raise RuntimeError("Benchmark scoring failed")
