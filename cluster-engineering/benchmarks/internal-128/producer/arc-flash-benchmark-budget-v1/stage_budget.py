#!/usr/bin/env python3
"""CPU verification only; reuse the existing v2 checkpoint, protocol and predictions."""
import json
from pathlib import Path

if __package__:
    from .budget_common import verify
else:
    from budget_common import verify


def stage(run):
    run = Path(run).resolve()
    benchmark, common = verify(run)
    recipe = json.loads((benchmark / "benchmark-recipe.json").read_text())
    with common.output_lock(run.parent / "runs/arc-flash-dev-benchmark-v2"):
        common.atomic_json(run / "READY.json", {"benchmark_id": recipe["benchmark_id"],
            "files": {path.name: common.digest(path) for path in run.iterdir()
                      if path.is_file() and path.name != "READY.json"}})
    print("ALLOCATION BUDGET READY: unchanged v2 benchmark; preserve and resume existing predictions", flush=True)
    print("No job submitted. Generation stops five minutes before the actual Slurm end time.", flush=True)


if __name__ == "__main__":
    stage(Path(__file__).resolve().parent)
