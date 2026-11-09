#!/usr/bin/env python3
"""CPU-only scoring; expected test outputs are loaded solely in this process."""
import json
from pathlib import Path

if __package__:
    from .common import OUTPUT_NAME, atomic_json, verify_ready
    from .protocol import rows, write_rows, load_predictions, score
else:
    from common import OUTPUT_NAME, atomic_json, verify_ready
    from protocol import rows, write_rows, load_predictions, score


def score_run(run):
    run = Path(run).resolve()
    verify_ready(run)
    output = run.parent / "runs" / OUTPUT_NAME
    recipe = json.loads((run / "benchmark-recipe.json").read_text())
    answers = rows(run / "answers.scorer-only.jsonl")
    predictions = load_predictions(output / "predictions.jsonl", {answer["task_id"] for answer in answers})
    if any(record.get("benchmark_id") != recipe["benchmark_id"] for record in predictions.values()):
        raise RuntimeError("Prediction provenance differs from the staged benchmark")
    summary, details = score(answers, predictions)
    summary.update(benchmark_id=recipe["benchmark_id"], global_step=recipe["global_step"],
                   adapter_sha256=recipe["adapter_sha256"], generation=recipe["generation"],
                   split="evaluation", protected_evaluation_used=True, ttt=False)
    atomic_json(output / "summary.json", summary)
    temporary = output / "per-task.jsonl.tmp"
    write_rows(temporary, details)
    temporary.replace(output / "per-task.jsonl")
    print("ARC_BENCHMARK " + json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    score_run(Path(__file__).resolve().parent)
