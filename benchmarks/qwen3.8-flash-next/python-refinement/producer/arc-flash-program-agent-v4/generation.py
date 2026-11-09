"""Same verified generation path, with exact-token continuation of one candidate."""
import contextlib
import os
import socket
import time
import uuid
if __package__:
    from .protocol import CONTEXT, MAX_NEW_TOKENS, thinking_fields, count_thinking_tokens
    from .result_io import utcnow
else:
    from protocol import CONTEXT, MAX_NEW_TOKENS, thinking_fields, count_thinking_tokens
    from result_io import utcnow

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


def generate_one(torch, model, tokenizer, prompt, arm, recipe, stop, *, prefix=None):
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
    token_ids = tokenizer.apply_chat_template(prompt["messages"], tokenize=True, return_dict=False,
                                              add_generation_prompt=True, enable_thinking=True)
    budget = min(MAX_NEW_TOKENS, CONTEXT - len(token_ids))
    if budget <= 0:
        raise RuntimeError("Complete prompt exceeds context; refusing truncation")
    prefix = list(prefix or [])
    if len(prefix) >= budget or recipe["eos_token_id"] in prefix:
        raise RuntimeError("Invalid interrupted response prefix")
    original_prompt_tokens = len(token_ids)
    token_ids = token_ids + prefix
    input_device = model.get_input_embeddings().weight.device
    inputs = torch.tensor([token_ids], dtype=torch.long, device=input_device)
    config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
        max_new_tokens=budget-len(prefix), use_cache=True, cache_implementation=None,
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
    started_at = utcnow()
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
    new_tokens = sequence[len(token_ids):].detach().cpu().tolist()
    completion = prefix + new_tokens
    ended = bool(completion and completion[-1] == recipe["eos_token_id"])
    interrupted = stop() and not ended and len(completion) < budget
    if not ended and len(completion) < budget and not interrupted:
        raise RuntimeError("Generation ended without EOS, token cap or the configured deadline")
    raw = tokenizer.decode(completion[:-1] if ended else completion, skip_special_tokens=False,
                           clean_up_tokenization_spaces=False)
    record = {"benchmark_id": recipe["benchmark_id"], "task_id": prompt["task_id"], "arm": arm,
            **thinking_fields(raw), **count_thinking_tokens(tokenizer, completion, recipe["eos_token_id"]),
            "completion_token_ids": completion,
            "thinking_delimiter_token_ids": tokenizer.encode("</think>", add_special_tokens=False),
            "eos_token_id": recipe["eos_token_id"],
            "attempt_id": uuid.uuid4().hex, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "allocation_id": os.environ.get("ARC_THINKING_ALLOCATION_ID"),
            "hostname": socket.gethostname(), "started_at": started_at, "finished_at": utcnow(),
            "new_generated_tokens": len(new_tokens), "replayed_prefix_tokens": len(prefix),
            "generation_input_tokens":len(token_ids), "prompt_tokens": original_prompt_tokens, "generated_tokens": len(completion), "token_budget": budget,
            "finish_reason": "interrupted" if interrupted else "eos" if ended else "token_limit",
            "seconds": time.monotonic() - started, "adapter_layers_checked": layers}
    record["accounting_validated"] = all(record[k] is not None for k in
        ("reasoning_tokens", "final_answer_tokens", "thinking_delimiter_tokens", "eos_tokens"))
    if record["accounting_validated"] and record["separation_error"] is None:
        at = record["reasoning_tokens"]
        end = at + record["thinking_delimiter_tokens"]
        clean = completion[:-1] if ended else completion
        if (tokenizer.decode(clean[:at], skip_special_tokens=False, clean_up_tokenization_spaces=False).strip() != record["reasoning"]
            or tokenizer.decode(clean[end:], skip_special_tokens=False, clean_up_tokenization_spaces=False).strip() != record["final_answer"]):
            raise RuntimeError("Token boundary does not match decoded thinking/final text")
    return record


