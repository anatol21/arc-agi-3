#!/usr/bin/env python3
"""Set an allocation-aware deadline and run the unchanged v2 generation program."""
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from .budget_common import verify, check_files
    from .deadline import allocation_budget
else:
    from budget_common import verify, check_files
    from deadline import allocation_budget


def launch(run):
    run = Path(run).resolve()
    benchmark, common = verify(run)
    ready = json.loads((run / "READY.json").read_text())
    check_files(run, ready["files"])
    recipe = json.loads((benchmark / "benchmark-recipe.json").read_text())
    if ready["benchmark_id"] != recipe["benchmark_id"]:
        raise RuntimeError("Benchmark identity changed after deadline staging")
    budget = allocation_budget()
    budget["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    output = run.parent / "runs/arc-flash-dev-benchmark-v2"
    common.atomic_json(output / "allocation-budget.json", budget)
    print("ALLOCATION_BUDGET " + json.dumps(budget), flush=True)
    if time.time() >= budget["stop_at"]:
        subprocess.run([sys.executable, str(benchmark / "score_benchmark.py")], check=True)
        print("No generation budget remains; scored saved predictions without loading weights", flush=True)
        return
    os.environ["ARC_BENCH_STOP_AT"] = str(budget["stop_at"])
    # Execute the exact original file: same recipe/ID/output, with all old progress.
    runpy.run_path(str(benchmark / "run_benchmark.py"), run_name="__main__")


if __name__ == "__main__":
    launch(Path(__file__).resolve().parent)
