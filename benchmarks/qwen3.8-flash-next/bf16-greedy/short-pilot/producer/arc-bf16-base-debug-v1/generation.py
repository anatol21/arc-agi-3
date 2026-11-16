"""Greedy ARC inference with independent budgets and durable, per-task progress."""
import fcntl
import gc
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

if __package__:
    from apps.cluster_dashboard.live_generation import generate_chunks
else:
    from live_generation import generate_chunks

CONTEXT = 18432
MAX_TOKENS = 8192
CHECKPOINT_TOKENS = 128
TERMINAL = {"eos", "token_limit"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, row):
    """The old or the new complete record survives an interrupted write."""
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(row, out, ensure_ascii=False, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def prepare(tokenizer, prompt):
    task = prompt["task_id"]
    if not re.fullmatch("[0-9a-f]{8}", task) or [m.get("role") for m in prompt["messages"]] != ["user"]:
        raise ValueError("Invalid ARC task or user message")
    opts = dict(add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.apply_chat_template(prompt["messages"], tokenize=True, return_dict=False, **opts)
    budget = min(MAX_TOKENS, CONTEXT - len(ids))
    if (ids != prompt["token_ids"] or digest(ids) != prompt["prompt_sha256"]
            or digest(prompt["messages"]) != prompt["messages_sha256"]
            or prompt.get("prompt_tokens", len(ids)) != len(ids) or budget < 1):
        raise ValueError(f"NF4 prompt/token identity or budget mismatch for {task}")
    return {"task_id": task, "ids": ids, "budget": budget, "messages_sha256": prompt["messages_sha256"],
            "prompt_sha256": prompt["prompt_sha256"]}


def load_state(output, task, benchmark_id, eos):
    path = output / "states" / (task["task_id"] + ".json")
    if not path.exists():
        return None
    row = json.loads(path.read_bytes())
    sha = row.pop("sha256")
    if sha != digest(row):
        raise ValueError(f"Corrupt state: {path}")
    ids = row["completion_token_ids"]
    if (row["benchmark_id"] != benchmark_id or row["arm"] != "base" or row["enable_thinking"]
            or row["task_id"] != task["task_id"] or row["messages_sha256"] != task["messages_sha256"]
            or row["prompt_sha256"] != task["prompt_sha256"] or row["token_budget"] != task["budget"]
            or row["prompt_tokens"] != len(task["ids"]) or row["generated_tokens"] != len(ids)
            or not isinstance(ids, list) or not 0 < len(ids) <= task["budget"]
            or any(type(token) is not int or token < 0 for token in ids) or eos in ids[:-1]):
        raise ValueError(f"State identity or token ledger mismatch: {path}")
    match = "eos" if ids[-1] == eos else "token_limit" if len(ids) == task["budget"] else "checkpoint"
    if match != row["finish_reason"]:
        raise ValueError(f"Incorrect saved stopping reason: {path}")
    return row


def memory(torch, device):
    if device.type != "cuda":
        return []
    return [dict(device=i, peak_allocated_bytes=torch.cuda.max_memory_allocated(i),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(i),
                 total_bytes=torch.cuda.get_device_properties(i).total_memory)
            for i in range(torch.cuda.device_count())]


def generate_group(torch, model, tokenizer, group, states, output, benchmark_id,
                   allocation_start, stop, attempt):
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

    device = model.get_input_embeddings().weight.device
    pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
    prefixes = [list(states[t["task_id"]]["completion_token_ids"]) if t["task_id"] in states else []
                for t in group]
    inputs = [t["ids"] + prefix for t, prefix in zip(group, prefixes, strict=True)]
    remaining = [t["budget"] - len(prefix) for t, prefix in zip(group, prefixes, strict=True)]
    width = max(map(len, inputs))
    if width + max(remaining) > CONTEXT:
        raise ValueError("Padded batch would exceed context; refusing truncation")
    attempt["replayed_prefix_tokens"] = sum(map(len, prefixes))
    attempt["padding_positions"] = width * len(inputs) - sum(map(len, inputs))
    ids = torch.full((len(inputs), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, tokens in enumerate(inputs):
        ids[i, -len(tokens):] = torch.tensor(tokens, dtype=torch.long, device=device)
        mask[i, -len(tokens):] = 1

    class Finite(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if not torch.isfinite(scores).all().item():
                raise RuntimeError("Nonfinite BF16 generation logits")
            return scores

    def commit(deltas):
        for task, prefix, delta in zip(group, prefixes, deltas, strict=True):
            if not delta:
                continue
            key = task["task_id"]
            if states.get(key, {}).get("finish_reason") in TERMINAL:
                raise RuntimeError(f"Attempt to replace a finished response: {key}")
            prefix.extend(delta)
            if eos in prefix[:-1] or len(prefix) > task["budget"]:
                raise RuntimeError(f"Invalid generated token ledger: {key}")
            reason = "eos" if prefix[-1] == eos else "token_limit" if len(prefix) == task["budget"] else "checkpoint"
            row = dict(task_id=key, benchmark_id=benchmark_id, arm="base", enable_thinking=False,
                       messages_sha256=task["messages_sha256"], prompt_sha256=task["prompt_sha256"],
                       prompt_tokens=len(task["ids"]), token_budget=task["budget"],
                       completion_token_ids=list(prefix), generated_tokens=len(prefix), finish_reason=reason,
                       text=tokenizer.decode(prefix, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                       slurm_job_id=os.environ.get("SLURM_JOB_ID"), attempt_id=attempt["attempt_id"],
                       actual_batch_size=len(group), recorded_at=time.time())
            write_json(output / "states" / f"{key}.json", {**row, "sha256": digest(row)})
            states[key] = row
            if reason in TERMINAL:
                elapsed = time.time() - allocation_start
                attempt["terminal_saves"].append(dict(task_id=key, seconds_from_allocation=elapsed))
                print("BF16_TERMINAL", key, reason, len(prefix), f"{elapsed:.2f}s from allocation", flush=True)

    cfg = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
                           max_new_tokens=max(remaining), use_cache=True, cache_implementation=None,
                           bos_token_id=tokenizer.bos_token_id, eos_token_id=eos, pad_token_id=pad,
                           return_dict_in_generate=True, output_scores=False, output_hidden_states=False,
                           output_attentions=False)
    # Every invocation starts fresh. Periodic disk commits below keep its live KV cache.
    for module in model.modules():
        if hasattr(module, "rope_deltas"):
            module.rope_deltas = None
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        generate_chunks(model, ids, mask, generation_config=cfg, row_limits=remaining,
                        chunk_tokens=CHECKPOINT_TOKENS, stop=stop, commit=commit, save_on_terminal=True,
                        logits_to_keep=1, logits_processor=LogitsProcessorList([Finite()]))
    if not stop() and any(states.get(t["task_id"], {}).get("finish_reason") not in TERMINAL for t in group):
        raise RuntimeError("Unexpected nonterminal generation result without stop request")


def run_groups(torch, model, tokenizer, groups, output, benchmark_id, allocation_start, stop):
    """Return allocation metrics. A second process cannot write into an active pilot."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("states", "attempts"):
            (output / name).mkdir(exist_ok=True)
        return _run(torch, model, tokenizer, groups, output, benchmark_id, allocation_start, stop)


def _run(torch, model, tokenizer, groups, output, benchmark_id, allocation_start, stop):
    groups = [[prepare(tokenizer, prompt) for prompt in group] for group in groups]
    tasks = [t for group in groups for t in group]
    if [len(group) for group in groups] != [4, 8] or len({t["task_id"] for t in tasks}) != 12:
        raise ValueError("Expected fixed cohorts of four and eight distinct tasks")
    states = {}
    for t in tasks:
        prior = load_state(output, t, benchmark_id, tokenizer.eos_token_id)
        if prior is not None:
            states[t["task_id"]] = prior
    prior_finished = {t for t, row in states.items() if row["finish_reason"] in TERMINAL}
    attempts = []
    device = model.get_input_embeddings().weight.device
    # Preserve a reduced width on resume if an earlier attempt in the same pilot exhausted VRAM.
    cap = len(tasks)
    for path in (output / "attempts").glob("*.json"):
        a = json.loads(path.read_text())
        if a["benchmark_id"] != benchmark_id:
            raise ValueError("Attempt belongs to a different benchmark")
        if a["status"] == "oom":
            cap = min(cap, a["retry_cap"])

    for target_group in groups:
        pending = [t for t in target_group if states.get(t["task_id"], {}).get("finish_reason") not in TERMINAL]
        while pending and cap and not stop():
            group = pending[:cap]
            started = time.time()
            attempt = dict(attempt_id=uuid.uuid4().hex, benchmark_id=benchmark_id,
                           slurm_job_id=os.environ.get("SLURM_JOB_ID"), started_at=started,
                           requested_cohort=len(target_group), actual_batch_size=len(group),
                           task_ids=[t["task_id"] for t in group], terminal_saves=[], status="running")
            path = output / "attempts" / (attempt["attempt_id"] + ".json")
            write_json(path, attempt)
            print("BF16_BATCH_START", json.dumps(attempt), flush=True)
            if device.type == "cuda":
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
            try:
                generate_group(torch, model, tokenizer, group, states, output, benchmark_id, allocation_start, stop, attempt)
                attempt["status"] = "complete" if all(states.get(t["task_id"], {}).get("finish_reason") in TERMINAL for t in group) else "interrupted"
            except torch.cuda.OutOfMemoryError as err:
                attempt["status"] = "oom"
                attempt["error"] = str(err)
                cap = len(group) // 2
                attempt["retry_cap"] = cap
            except Exception as err:
                attempt["status"] = "failed"
                attempt["error"] = repr(err)
                raise
            finally:
                attempt["elapsed_seconds"] = time.time() - started
                attempt["memory"] = memory(torch, device)
                write_json(path, attempt)
                attempts.append(attempt)
                print("BF16_BATCH_END", json.dumps(attempt), flush=True)
            if attempt["status"] == "oom":
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            pending = [t for t in pending if states.get(t["task_id"], {}).get("finish_reason") not in TERMINAL]

    finished = {t for t, row in states.items() if row["finish_reason"] in TERMINAL}
    first = [record["seconds_from_allocation"] for a in attempts for record in a["terminal_saves"]]
    elapsed = time.time() - allocation_start
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    gpu_hours = gpu_count * elapsed / 3600
    return dict(completed=len(finished), new_completed=len(finished - prior_finished),
                skipped_completed=len(prior_finished), pending=len(tasks) - len(finished),
                successful_batch_sizes=[a["actual_batch_size"] for a in attempts if a["status"] == "complete"],
                oom_batch_sizes=[a["actual_batch_size"] for a in attempts if a["status"] == "oom"],
                first_saved_seconds=min(first) if first else None, allocation_elapsed_seconds=elapsed,
                gpu_hours_until_report=gpu_hours,
                new_responses_per_gpu_hour_until_report=len(finished - prior_finished) / gpu_hours if gpu_hours else None)
