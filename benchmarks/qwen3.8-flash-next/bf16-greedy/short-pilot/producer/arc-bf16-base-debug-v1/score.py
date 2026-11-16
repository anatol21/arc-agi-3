"""Read query answers only in this post-generation scoring process."""
import hashlib
import json
import os
from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER))
from stage import read_plan


def main():
    plan = read_plan(FOLDER)
    output = FOLDER.parent / "runs" / FOLDER.name / plan["benchmark_id"]
    native = FOLDER.parent / "arc-flash-eval138-v1"
    sys.path.insert(0, str(native))
    from protocol import score, rows
    selected = {t for group in plan["groups"] for t in group}
    answer_path = native / "answers.scorer-only.jsonl"
    native_recipe = json.loads((native / "benchmark-recipe.json").read_text())
    if hashlib.sha256(answer_path.read_bytes()).hexdigest() != native_recipe["answers_sha256"]:
        raise ValueError("Native answer file differs from the archived scorer")
    answers = [r for r in rows(answer_path) if r["task_id"] in selected]
    if {r["task_id"] for r in answers} != selected:
        raise ValueError("Missing native scorer answers for the selected tasks")
    predictions = {}
    for path in sorted((output / "states").glob("*.json")):
        r = json.loads(path.read_text())
        sha = r.pop("sha256")
        if hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != sha:
            raise ValueError(f"Invalid journal checksum: {path.name}")
        if r["benchmark_id"] != plan["benchmark_id"] or r["task_id"] not in selected:
            raise ValueError(f"Unexpected state {path.name}")
        if r["finish_reason"] in {"eos", "token_limit"}:
            predictions[r["task_id"], "base"] = r
    summary, details = score(answers, predictions)
    comparison = json.loads((FOLDER / "nf4-reference.json").read_text())
    for r in details:
        prediction = predictions[r["task_id"], "base"]
        baseline = comparison[r["task_id"]]
        r.update(generated_tokens=prediction["generated_tokens"], nf4_exact=baseline["task_correct"],
                 nf4_generated_tokens=baseline["generated_tokens"], nf4_package=baseline["package"],
                 same_response_text=hashlib.sha256(prediction["text"].encode()).hexdigest()==baseline["response_sha256"])
    result = dict(benchmark_id=plan["benchmark_id"], model="original precision, base only",
                  nf4_reference="Historical first completion per task; batching and GPU placement may differ",
                  base=summary["arms"]["base"], tasks=details)
    target = output / "allocations" / os.environ["SLURM_JOB_ID"] / "scores.json"
    target.write_text(json.dumps(result, indent=2)+"\n")
    print("BF16_SCORE", json.dumps(result["base"]), flush=True)


if __name__ == "__main__":
    main()
