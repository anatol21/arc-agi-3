"""Verify the release on CPU. Supply the original labeled development JSONL for a fresh exact-grid rescore."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "producer" / "arc-flash-dev-benchmark-v2"
sys.path.insert(0, str(RUN))
from protocol import ARMS, load_predictions, parse_outputs, rows, score


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(labeled_dev=None):
    proof = json.loads((ROOT / "evidence-sha256.json").read_text())
    for filename, digest in proof.items():
        require(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, f"Changed evidence: {filename}")
    manifest = json.loads((RUN / "data-manifest.json").read_text())
    ids = manifest["task_ids"]["dev"]
    summary = json.loads((ROOT / "summary.json").read_text())
    require(len(ids) == len(set(ids)) == 128, "Unexpected development manifest")
    predictions = load_predictions(ROOT / "predictions.jsonl", set(ids))
    detail = rows(ROOT / "per-task.jsonl")
    index = {(r["task_id"], r["arm"]): r for r in detail}
    expected = {(i, arm) for i in ids for arm in ARMS}
    require(len(detail) == 256 and index.keys() == predictions.keys() == expected, "Incomplete or duplicate results")
    correct = {}
    for arm in ARMS:
        correct[arm] = set()
        grids = matches = malformed = limited = 0
        for task in ids:
            p, s = predictions[task, arm], index[task, arm]
            require(p["benchmark_id"] == summary["benchmark_id"], "Incompatible prediction")
            require(p["finish_reason"] == s["finish_reason"], "Finish reason differs")
            require(p["task_id"] == task and s["task_correct"] == all(s["grid_correct"]), "Inconsistent task result")
            err = None
            try:
                parse_outputs(p["text"], len(s["grid_correct"]))
            except (ValueError, TypeError, KeyError) as error:
                err = str(error)
            require(bool(err) == bool(s["format_error"]), "Parser and score record differ")
            if s["task_correct"]:
                correct[arm].add(task)
            grids += len(s["grid_correct"])
            matches += sum(s["grid_correct"])
            malformed += bool(err)
            limited += p["finish_reason"] == "token_limit"
        report = summary["arms"][arm]
        require(report["complete"] and report["solved_tasks"] == len(correct[arm]) and report["processed_tasks"] == 128 and
                report["total_grids"] == grids == 134 and report["correct_grids"] == matches and
                report["invalid_json_or_grids"] == malformed and report["token_limit_tasks"] == limited, "Summary mismatch")
    tuned, base = correct["finetuned"], correct["base"]
    comparison = {"paired_tasks": len(ids), "both_correct": len(tuned & base), "finetuned_only_correct": len(tuned - base),
                  "base_only_correct": len(base - tuned), "both_wrong": len(set(ids) - tuned - base)}
    require(comparison == summary["paired_comparison"], "Paired counts differ")
    if labeled_dev:
        require(hashlib.sha256(labeled_dev.read_bytes()).hexdigest() == manifest["files_sha256"]["data/dev.jsonl"],
                "The labeled dev file differs from the original input")
        answers = [{"task_id": r["task_id"], "outputs": [query["output"] for query in r["test"]], "source": r["source"]}
                   for r in rows(labeled_dev)]
        fresh, details = score(answers, predictions)
        require(details == detail and all(summary[k] == v for k, v in fresh.items()), "Rescore differs from published results")
    return {"paired": comparison, "both": sorted(tuned & base), "finetuned_only": sorted(tuned-base),
            "base_only": sorted(base-tuned), "rescored_with_original_labels": bool(labeled_dev)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-dev", type=Path)
    print(json.dumps(verify(parser.parse_args().labeled_dev), indent=2))
