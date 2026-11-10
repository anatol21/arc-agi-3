"""Length-aware continuation for the verified four-H100 BF16 base workload."""
import fcntl
import gc
import json
import os
import time
import uuid
from pathlib import Path

if __package__:
    from . import generation as native
else:
    import generation as native

WIDTHS = (1, 2, 4, 8, 12, 16)
BAND_ENDS = (4096, 8192, 10240, native.CONTEXT)
MAX_SPREAD_PERCENT = 135


def hardware_identity(torch, device):
    if device.type != "cuda":
        return {"device": str(device)}
    return {"gpus": [
        {"model": p.name, "bytes": p.total_memory, "compute_capability": [p.major, p.minor]}
        for p in (torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count()))
    ]}


def retry_width(failed_size):
    return max((w for w in WIDTHS if w < failed_size), default=0)


def choose_group(tasks, states, history):
    """Return a physical batch with full original budgets, or an empty batch if blocked."""
    # An unsafe width stays disabled in its band. Retrying rows individually retain
    # their backoff even if the saved prefix moves them into a different length band.
    success = {}
    ceilings = {}
    retry = {}
    for a in history:
        b, n = a["band_end"], a["actual_batch_size"]
        if a["status"] in {"complete", "segment_complete"}:
            success[b] = max(success.get(b, 0), n)
        elif a["status"] == "oom":
            limit = retry_width(n)
            ceilings[b] = min(ceilings.get(b, WIDTHS[-1]), max(1, limit))
            for tid in a["task_ids"]:
                retry[tid] = min(retry.get(tid, WIDTHS[-1]), limit)

    pending = []
    for t in tasks:
        tid = t["task_id"]
        row = states.get(tid, {})
        if row.get("finish_reason") in native.TERMINAL or retry.get(tid) == 0:
            continue
        prefix = len(row.get("completion_token_ids", ()))
        width = len(t["ids"]) + prefix
        remaining = t["budget"] - prefix
        # Each task can always run alone with its native remaining allowance.
        if remaining <= 0 or width + remaining > native.CONTEXT:
            raise ValueError(f"Invalid continuation lengths for {tid}")
        band = next(b for b in BAND_ENDS if width <= b)
        priority = (tid not in retry, prefix == 0, width, tid)
        pending.append((priority, band, t, width))
    if not pending:
        return [], {}

    pending.sort(key=lambda r: r[0])
    _, band, seed, seed_width = pending[0]
    safe = success.get(band, 0)
    target = 8 if safe < 8 else 12 if safe < 12 else 16
    target = min(target, ceilings.get(band, WIDTHS[-1]), retry.get(seed["task_id"], WIDTHS[-1]))
    # Choose comparable widths near the seed. The generator journals and regroups
    # surviving rows if the physical padded call reaches the context boundary.
    peers = sorted((r for r in pending if r[1] == band), key=lambda r: (
        r[2]["task_id"] != seed["task_id"], r[0][:2], abs(r[3] - seed_width), r[3], r[2]["task_id"]))
    chosen = []
    min_width = max_width = seed_width
    row_ceiling = target
    for _, _, t, width in peers:
        ceiling = min(row_ceiling, retry.get(t["task_id"], WIDTHS[-1]))
        if len(chosen) + 1 > ceiling:
            continue
        low, high = min(min_width, width), max(max_width, width)
        if high * 100 > low * MAX_SPREAD_PERCENT:
            continue
        chosen.append(t)
        min_width, max_width, row_ceiling = low, high, ceiling
        if len(chosen) == ceiling:
            break
    # Prefer an established width if there are too few suitable rows for the
    # next probe; e.g. use 8 rather than introducing an unplanned width of 11.
    if len(chosen) < target and len(chosen) >= 8:
        chosen = chosen[:max(w for w in WIDTHS if w <= len(chosen))]
    return chosen, {"band_end": band, "target": target, "previous_successful_width": safe,
                    "size_probe": len(chosen) > safe and len(chosen) in (12, 16)}


def _history(output, benchmark_id, hardware):
    history = []
    for path in sorted((output / "attempts").glob("*.json")):
        row = json.loads(path.read_bytes())
        if row["benchmark_id"] != benchmark_id:
            raise ValueError(f"Incompatible attempt in continuation directory: {path}")
        if row.get("hardware") == hardware:
            history.append(row)
    return history


def run_remaining(torch, model, tokenizer, prompts, output, benchmark_id, allocation_start, stop):
    """Run unfinished tasks until deadline, exhaustion, or all pending rows are blocked."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if isinstance(prompts, dict):
        prompts = prompts.values()
    tasks = [native.prepare(tokenizer, p) for p in prompts]
    if len(tasks) != 126 or len({t["task_id"] for t in tasks}) != 126:
        raise ValueError("Expected the 126 distinct tasks in the continuation manifest")
    with (output / "generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("states", "attempts"):
            (output / name).mkdir(exist_ok=True)
        return _run_locked(torch, model, tokenizer, tasks, output, benchmark_id, allocation_start, stop)


def execute_attempt(torch, model, tokenizer, group, states, output, benchmark_id, allocation_start,
                    stop, hardware, decision, *, sampling=None, strategy=None):
    started = time.time()
    attempt = dict(**decision, attempt_id=uuid.uuid4().hex, benchmark_id=benchmark_id, hardware=hardware,
                   slurm_job_id=os.environ.get("SLURM_JOB_ID"), started_at=started, status="running",
                   actual_batch_size=len(group), task_ids=[t["task_id"] for t in group], terminal_saves=[])
    if sampling is not None:
        attempt.update(strategy=strategy, sampling=dict(sampling))
    device = model.get_input_embeddings().weight.device
    path = output / "attempts" / (attempt["attempt_id"] + ".json")
    native.write_json(path, attempt)
    print("BF16_BATCH_START", json.dumps(attempt), flush=True)
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    try:
        options = {"segment_to_context": True}
        if sampling is not None:
            options["sampling"] = sampling
        native.generate_group(torch, model, tokenizer, group, states, output, benchmark_id,
                              allocation_start, stop, attempt, **options)
        finished_group = all(states.get(t["task_id"], {}).get("finish_reason") in native.TERMINAL for t in group)
        attempt["status"] = "complete" if finished_group else "interrupted"
        if not finished_group and attempt.get("context_limited_segment") and not stop():
            attempt["status"] = "segment_complete"
    except torch.cuda.OutOfMemoryError as err:
        attempt.update(status="oom", error=str(err), retry_cap=retry_width(len(group)))
    except Exception as err:
        attempt.update(status="failed", error=repr(err))
        raise
    finally:
        attempt["elapsed_seconds"] = time.time() - started
        attempt["memory"] = native.memory(torch, device)
        native.write_json(path, attempt)
        print("BF16_BATCH_END", json.dumps(attempt), flush=True)
    if attempt["status"] == "oom":
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return attempt


def _run_locked(torch, model, tokenizer, tasks, output, benchmark_id, allocation_start, stop):
    device = model.get_input_embeddings().weight.device
    hardware = hardware_identity(torch, device)
    history = _history(output, benchmark_id, hardware)
    states = {}
    for t in tasks:
        row = native.load_state(output, t, benchmark_id, tokenizer.eos_token_id)
        if row is not None:
            states[t["task_id"]] = row
    prior = {tid for tid, row in states.items() if row["finish_reason"] in native.TERMINAL}
    attempts = []
    while not stop():
        group, decision = choose_group(tasks, states, history)
        if not group:
            break
        attempt = execute_attempt(torch, model, tokenizer, group, states, output, benchmark_id, allocation_start,
                                  stop, hardware, decision)
        attempts.append(attempt)
        history.append(attempt)

    finished = {tid for tid, row in states.items() if row["finish_reason"] in native.TERMINAL}
    blocked = {tid for a in history if a["status"] == "oom" and a["actual_batch_size"] == 1
               for tid in a["task_ids"]} - finished
    saved = [r["seconds_from_allocation"] for a in attempts for r in a["terminal_saves"]]
    elapsed = time.time() - allocation_start
    gpu_hours = (torch.cuda.device_count() if device.type == "cuda" else 0) * elapsed / 3600
    new = len(finished - prior)
    return dict(completed=len(finished), new_completed=new, pending=len(tasks)-len(finished),
                skipped_completed=len(prior), blocked_task_ids=sorted(blocked),
                successful_batch_sizes=[a["actual_batch_size"] for a in attempts if a["status"] == "complete"],
                context_segment_batch_sizes=[a["actual_batch_size"] for a in attempts if a["status"] == "segment_complete"],
                oom_batch_sizes=[a["actual_batch_size"] for a in attempts if a["status"] == "oom"],
                first_saved_seconds=min(saved) if saved else None, allocation_elapsed_seconds=elapsed,
                gpu_hours_until_report=gpu_hours,
                new_responses_per_gpu_hour_until_report=new/gpu_hours if gpu_hours else None)
