"""ARC direct-answer generation with durable token prefixes and early terminal saves."""

from contextlib import contextmanager, nullcontext
import inspect
import json
import math

from apps.cluster_dashboard.live_generation import generate_chunks
from scheduling.inference.batching import TERMINAL, checksum

# The task JSON starts directly after the final newline, as in the native benchmark.
INSTRUCTION = """Solve this ARC task. Infer the transformation from the training examples and apply it to every test input.

Return exactly one complete JSON object with this schema and no prose, markdown, explanation, or code fence:
{"outputs": [[[0]]]}

The `outputs` array must contain one rectangular 0–9 integer grid for each item in `test`, in the same order. Write every row and every cell explicitly. Never abbreviate repeated values or rows, and never use ellipses (such as `...` or `......`), placeholders, or omitted content.

Task:
"""


def checked_grid(grid):
    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:
        raise ValueError("ARC grid must have 1 to 30 rows")
    if not isinstance(grid[0], list) or not 1 <= len(grid[0]) <= 30:
        raise ValueError("ARC grid must have 1 to 30 columns")
    if any(not isinstance(row, list) or len(row) != len(grid[0]) or
           any(type(cell) is not int or not 0 <= cell <= 9 for cell in row) for row in grid):
        raise ValueError("ARC grid must be rectangular and contain integer colors 0 to 9")
    return grid


def messages_for(record):
    """Construct the one user message. Only demonstration pairs and query inputs enter it."""
    if not record.get("train") or not record.get("test"):
        raise ValueError("Expected at least one demonstration and query")
    visible = {
        "train": [{"input": checked_grid(row["input"]), "output": checked_grid(row["output"])}
                  for row in record["train"]],
        "test": [{"input": checked_grid(row["input"])} for row in record["test"]],
    }
    content = INSTRUCTION + json.dumps(visible, separators=(",", ":"))
    return [{"role": "user", "content": content}]


def prepare_tasks(tokenizer, records, *, context=18432, max_new_tokens=8192):
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if type(eos) is not int or type(pad) is not int or eos < 0 or pad < 0 or eos == pad:
        raise ValueError("Provide a tokenizer with distinct, nonnegative EOS and PAD IDs")
    if type(context) is not int or type(max_new_tokens) is not int or min(context, max_new_tokens) <= 0:
        raise ValueError("Context and output budget must be positive integers")
    prepared, seen = [], set()
    for record in records:
        task_id = record["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError("Task IDs must be unique nonempty strings")
        seen.add(task_id)
        messages = messages_for(record)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                            add_generation_prompt=True, enable_thinking=False)
        if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError(f"Invalid chat-template token IDs for {task_id}")
        budget = min(max_new_tokens, context - len(ids))
        if budget <= 0:
            raise ValueError(f"Complete prompt exhausts context for {task_id}")
        prepared.append(dict(task_id=task_id, token_ids=ids, prompt_tokens=len(ids), budget=budget,
                             context=context, messages_sha256=checksum(messages), eos=eos, pad=pad))
    if not prepared:
        raise ValueError("Empty task manifest")
    return prepared


def sampling_options(policy):
    if set(policy) != {"temperature", "top_p", "top_k", "seed"}:
        raise ValueError("Specify temperature, top_p, top_k and seed for sampling")
    for key in ("temperature", "top_p"):
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key}")
    if policy["top_p"] > 1 or type(policy["top_k"]) is not int or policy["top_k"] < 0:
        raise ValueError("Expected top_p <= 1 and nonnegative top_k")
    if type(policy["seed"]) is not int or not 0 <= policy["seed"] < 2**63:
        raise ValueError("Expected a nonnegative 63-bit seed")
    return {key: policy[key] for key in ("temperature", "top_p", "top_k")}


def workload_identity(tasks, *, model_revision, tokenizer_revision, sampling, context, max_new_tokens):
    """Input to Journal's checksum. Include loader/precision settings in the caller's identity too."""
    if not model_revision or not tokenizer_revision:
        raise ValueError("Expected explicit model and tokenizer revisions")
    if sampling is not None:
        sampling_options(sampling)
    rows = tasks.values() if isinstance(tasks, dict) else tasks
    return dict(protocol="arc-direct-answer-v1", model_revision=model_revision,
                tokenizer_revision=tokenizer_revision, sampling=sampling, context=context,
                max_new_tokens=max_new_tokens,
                tasks=[dict(task_id=t["task_id"], user_messages_sha256=t["messages_sha256"],
                            token_ids_sha256=checksum(t["token_ids"]), budget=t["budget"],
                            eos=t["eos"], pad=t["pad"]) for t in sorted(rows, key=lambda t: t["task_id"])])


@contextmanager
def sampling_rng(torch, device, seed):
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(seed)
        for index in devices:
            with torch.cuda.device(index):
                torch.cuda.manual_seed(seed)
        yield


def decode_batch(model, tokenizer, tasks, group, journal, *, stop, checkpoint_tokens=128,
                 sampling=None, on_checkpoint=None):
    """Decode one admitted group. Caller holds the journal writer lock and records attempt status."""
    import torch
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    states = journal.states()
    active, prefixes = [], []
    for key in group["task_ids"]:
        task = tasks[key]
        if (eos, pad) != (task["eos"], task["pad"]):
            raise ValueError(f"Tokenizer changed for {key}")
        saved = states.get(key)
        prefix = list(saved["tokens"]) if saved else []
        if saved:
            reason = ("eos" if prefix and prefix[-1] == eos else
                      "token_limit" if len(prefix) == task["budget"] else "checkpoint")
            if (not prefix or len(prefix) > task["budget"] or eos in prefix[:-1]
                    or saved["reason"] != reason):
                raise ValueError(f"Invalid saved stopping state for {key}")
            if reason in TERMINAL:
                continue
        active.append(task)
        prefixes.append(prefix)

    if not active:
        return "complete"
    if stop():
        return "interrupted"
    inputs = [task["token_ids"] + prefix for task, prefix in zip(active, prefixes)]
    width = max(map(len, inputs))
    ceiling = min(group["call_token_ceiling"], min(t["context"] for t in active) - width)
    if ceiling < 1:
        raise ValueError("Padded prompt and saved prefix exhaust context")
    limits = [min(task["budget"] - len(prefix), ceiling) for task, prefix in zip(active, prefixes)]
    initial_lengths = list(map(len, prefixes))
    device = model.get_input_embeddings().weight.device
    ids = torch.full((len(inputs), width), pad, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(inputs):
        ids[i, -len(row):] = torch.tensor(row, device=device, dtype=torch.long)
        mask[i, -len(row):] = 1

    class FiniteLogits(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if not torch.isfinite(scores).all().item():
                raise RuntimeError("Nonfinite model logits")
            return scores

    def commit(deltas):
        for task, prefix, delta in zip(active, prefixes, deltas, strict=True):
            if not delta:
                continue
            prefix.extend(delta)
            if eos in prefix[:-1] or len(prefix) > task["budget"]:
                raise RuntimeError(f"Invalid generated tokens for {task['task_id']}")
            reason = ("eos" if prefix[-1] == eos else
                      "token_limit" if len(prefix) == task["budget"] else "checkpoint")
            journal.checkpoint(task, prefix, reason)
            if on_checkpoint is not None:
                on_checkpoint(task["task_id"], list(prefix), reason)

    options = sampling_options(sampling) if sampling is not None else {}
    rng = nullcontext()
    if sampling is not None:
        # A changed prefix length or batch composition can change sampled draws on restart.
        layout = [[t["task_id"], n] for t, n in zip(active, initial_lengths)]
        seed = int(checksum(["sample-v1", sampling["seed"], layout])[:15], 16)
        rng = sampling_rng(torch, device, seed)
    config = GenerationConfig(do_sample=sampling is not None, num_beams=1, num_return_sequences=1,
                              max_new_tokens=max(limits), use_cache=True, cache_implementation=None,
                              bos_token_id=tokenizer.bos_token_id, eos_token_id=eos, pad_token_id=pad,
                              return_dict_in_generate=True, output_scores=False,
                              output_hidden_states=False, output_attentions=False, **options)
    # Some Qwen models store RoPE offsets on the model itself; each new group needs fresh offsets.
    for module in model.modules():
        if hasattr(module, "rope_deltas"):
            module.rope_deltas = None
    extra = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    with rng, torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                       enabled=device.type == "cuda" and model.dtype == torch.bfloat16):
        generate_chunks(model, ids, mask, generation_config=config, row_limits=limits,
                        chunk_tokens=checkpoint_tokens, stop=stop, commit=commit, save_on_terminal=True,
                        allow_sampling=sampling is not None, logits_processor=LogitsProcessorList([FiniteLogits()]),
                        **extra)
    if all(prefix and (prefix[-1] == eos or len(prefix) == task["budget"])
           for task, prefix in zip(active, prefixes)):
        return "complete"
    if stop():
        return "interrupted"
    for t, prefix, n, limit in zip(active, prefixes, initial_lengths, limits):
        if not prefix or (len(prefix) < t["budget"] and prefix[-1] != eos and len(prefix) != n + limit):
            raise RuntimeError(f"Decode ended before its physical call limit for {t['task_id']}")
    return "segment_complete"
