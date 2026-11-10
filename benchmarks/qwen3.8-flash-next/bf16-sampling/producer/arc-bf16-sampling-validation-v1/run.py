"""One allocation: stage native weights, audit BF16 placement, then run the two cohorts."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
from datetime import datetime
import sys
import time
import traceback

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER))
from stage import read_plan
from cold import local_weights
from generation import run_groups
from loader import load_base


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def run_allocation(plan, drive, *, scope):
    workspace = FOLDER.parent
    output = workspace / "runs" / FOLDER.name / plan["benchmark_id"]
    job = os.environ["SLURM_JOB_ID"]
    allocation = output / "allocations" / job
    allocation.mkdir(parents=True, exist_ok=False)
    from runpy import run_path
    budget = run_path(str(workspace / "arc-flash-eval138-v1/deadline.py"))["allocation_budget"]()
    end = int(budget["allocation_end"])
    if os.environ.get("SLURM_JOB_START_TIME"):
        start = int(os.environ["SLURM_JOB_START_TIME"])
    else:
        fields = subprocess.check_output(["scontrol", "--oneliner", "show", "job", job],
                                         text=True, timeout=15).split()
        started = next(field.split("=", 1)[1] for field in fields if field.startswith("StartTime="))
        start = int(datetime.fromisoformat(started).timestamp())
    interrupted = False

    def signal_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, signal_stop)

    def stop():
        return interrupted or time.time() >= end - 60

    report = dict(job=job, allocation_start=start, allocation_end=end,
                  benchmark_id=plan["benchmark_id"], scope=scope,
                  phase="staging_weights")
    weight_staging = {}
    write(allocation / "status.json", report)
    with (output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            import torch
            if (torch.cuda.device_count() != 4 or any("H100" not in torch.cuda.get_device_name(i)
                                                       for i in range(4))):
                raise RuntimeError("BF16 inference needs four visible H100s in a single process")
            with local_weights(workspace, plan, weight_staging) as local_model:
                write(allocation / "weight-staging.json", weight_staging)
                report["phase"] = "loading"
                write(allocation / "status.json", report)
                load_start = time.monotonic()
                model, tokenizer, verified = load_base(
                    Path(local_model), Path(plan["checkpoint_tokenizer"]),
                    workspace / "arc-flash-eval138-v1", FOLDER / "tensor-inventory.csv", allocation)
                if (tokenizer.eos_token_id, tokenizer.pad_token_id) != (plan["eos_token_id"], plan["pad_token_id"]):
                    raise ValueError("EOS or padding token changed since the original NF4 benchmark")
                report.update(loader_seconds=time.monotonic()-load_start,
                              verified_at=time.time(), phase="generating")
                write(allocation / "verified-loader.json", verified)
                write(allocation / "status.json", report)
                torch.manual_seed(42)
                summary = drive(torch, model, tokenizer, output, start, stop)
                write(allocation / "generation-summary.json", summary)
                print("BF16_GENERATION_SUMMARY", json.dumps(summary), flush=True)
                report["phase"] = "generation_ended"
        except BaseException:
            report.update(phase="failed", error=traceback.format_exc())
            raise
        finally:
            report["worker_end"] = time.time()
            write(allocation / "weight-staging.json", weight_staging)
            write(allocation / "status.json", report)


def main():
    plan = read_plan(FOLDER)
    groups = [[plan["prompts"][task] for task in group] for group in plan["groups"]]

    def pilot(torch, model, tokenizer, output, start, stop):
        return run_groups(torch, model, tokenizer, groups, output, plan["benchmark_id"], start, stop)

    run_allocation(plan, pilot, scope={"groups": plan["groups"]})


if __name__ == "__main__":
    main()
