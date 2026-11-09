"""Deterministic, CPU-only admission estimates for chunked code generation.

These bounds are deliberately higher than the prior90k-token limit. OOM is
possible and triggers bounded backoff, never below18requests. A generation chunk never changes a row's
full remaining candidate budget; it only bounds how much KV cache is retained
before prefixes are saved and independent requests are regrouped.
"""
from collections.abc import Mapping, Sequence


CONTEXT = 18_432
MAX_CANDIDATE_TOKENS = 8_192
MAX_BATCH = 64
CHUNK_TOKENS = 512
MAX_PADDING_RATIO = 1.5
MIN_BATCH = 18
MAX_PADDED_INPUT_TOKENS = 270_000
MAX_PADDED_CHUNK_END_TOKENS = 300_000

# Plain JSON values: this complete policy can be included in immutable recipes.
POLICY = {
    "name": "public_validation_dashboard_large_batch_v1",
    "context": CONTEXT,
    "max_candidate_tokens": MAX_CANDIDATE_TOKENS,
    "max_batch": MAX_BATCH,
    "min_batch": MIN_BATCH,
    "initial_target": 24,
    "target_ladder": [18,24,32,48,64],
    "chunk_tokens": CHUNK_TOKENS,
    "max_padding_ratio": MAX_PADDING_RATIO,
    "max_padded_input_tokens": MAX_PADDED_INPUT_TOKENS,
    "max_padded_chunk_end_tokens": MAX_PADDED_CHUNK_END_TOKENS,
    "ordering": "completion-first rolling seeds; closest-length fill; deterministic task-id ties",
    "input_definition": "full tokenized prompt plus saved completion prefix",
    "chunk_end_definition": "batch_size * (max input width + max min(remaining, chunk_tokens))",
    "budget_policy": "independent full candidate budgets are never shortened",
    "memory_interpretation": "heuristic admission estimates, not proof of GPU-memory fit",
    "oom_policy": "record failed cost; lower target through ladder; block arm if18OOM; never launch below18",
    "max_failed_attempts_per_allocation": 6,
    "selection": "all138 public tasks minus previously finished pairs; oldest unfinished prefix then closest-length fill",
}


def _positive_integer(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")


def _configuration(max_batch, chunk_tokens):
    _positive_integer(max_batch, "max_batch", MAX_BATCH)
    _positive_integer(chunk_tokens, "chunk_tokens", CHUNK_TOKENS)


def _row(row):
    if not isinstance(row, Mapping):
        raise ValueError("Prepared rows must be mappings")
    request = row.get("request")
    if not isinstance(request, Mapping) or not isinstance(request.get("task_id"), str) or not request["task_id"]:
        raise ValueError("Each prepared row requires a nonempty string request.task_id")
    ids = row.get("ids")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
        raise ValueError("ids must be a nonempty sequence of token IDs")
    width = len(ids)
    _positive_integer(width, "input length", CONTEXT)
    remaining = row.get("remaining")
    _positive_integer(remaining, "remaining", MAX_CANDIDATE_TOKENS)
    if width + remaining > CONTEXT:
        raise ValueError("A row's full remaining candidate budget exceeds its context")
    return width, remaining


def _metrics(rows, chunk_tokens):
    widths = [len(row["ids"]) for row in rows]
    width = max(widths)
    total = sum(widths)
    count = len(rows)
    chunk = max(min(row["remaining"], chunk_tokens) for row in rows)
    padded = count * width
    return {
        "batch_size": count,
        "chunk_tokens": chunk_tokens,
        "padded_input_width": width,
        "sum_input_tokens": total,
        "padded_input_tokens": padded,
        "padding_tokens": padded - total,
        "padding_ratio": padded / total,
        "max_chunk_new_tokens": chunk,
        "padded_chunk_end_width": width + chunk,
        "padded_chunk_end_tokens": count * (width + chunk),
        "max_full_candidate_end_tokens": max(len(row["ids"]) + row["remaining"] for row in rows),
    }


def _within(metrics, max_batch):
    # Integer cross multiplication keeps the 1.5 threshold exact.
    return (
        MIN_BATCH <= metrics["batch_size"] <= max_batch
        and 2 * metrics["padded_input_tokens"] <= 3 * metrics["sum_input_tokens"]
        and metrics["padded_input_tokens"] <= MAX_PADDED_INPUT_TOKENS
        and metrics["padded_chunk_end_tokens"] <= MAX_PADDED_CHUNK_END_TOKENS
        and metrics["padded_chunk_end_width"] <= CONTEXT
    )


def admission_metrics(group, chunk_tokens=CHUNK_TOKENS):
    """Describe a nonempty group's conservative padded-input/cache estimates.

    Invalid rows or a row whose full independent budget cannot fit its own
    context are rejected. A valid group's estimates may exceed admission bounds;
    use ``accepts`` to distinguish that case.
    """
    _configuration(MAX_BATCH, chunk_tokens)
    rows = list(group)
    if not rows:
        raise ValueError("Cannot measure an empty group")
    for row in rows:
        _row(row)
    return _metrics(rows, chunk_tokens)


def accepts(group, max_batch=MAX_BATCH, chunk_tokens=CHUNK_TOKENS):
    """Return whether valid rows fit every admission bound; reject invalid rows."""
    _configuration(max_batch, chunk_tokens)
    rows = list(group)
    if not rows:
        return False
    for row in rows:
        _row(row)
    return _within(_metrics(rows, chunk_tokens), max_batch)

