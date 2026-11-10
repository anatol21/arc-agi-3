"""Account, choose a production partition and resume the two BF16 sampling strategies."""
import argparse
from datetime import datetime
from decimal import Decimal
import json
import math
from pathlib import Path
import re
import subprocess
import sys

FOLDER = Path(__file__).resolve().parent
if __package__:
    from . import continuation_scheduler as scheduler
    from .continuation_score import checked_states
    from .generation import TERMINAL
    from .sample_cont_stage import arm_id, arm_prompts, read_plan
else:
    sys.path.insert(0, str(FOLDER))
    import continuation_scheduler as scheduler
    from continuation_score import checked_states
    from generation import TERMINAL
    from sample_cont_stage import arm_id, arm_prompts, read_plan

PARTITIONS = ("gpu_h100", "gpu_h100_short")
MAX_WAVE_MINUTES = 360
# The actual four-H100 loader reached verification after 360.56 s in pilot 8487396.
RELOAD_SECONDS = 361


def progress(root, plan):
    counts = dict(terminal=0, committed_tokens=0)
    for slug in plan["arm_order"]:
        states = checked_states(root / "arms" / slug, arm_prompts(plan, slug),
                                arm_id(plan, slug), plan["eos_token_id"])
        for record in states.values():
            if record.get("sampling") != plan["sampling"][slug]:
                raise ValueError(f"Unexpected sampling policy in a saved {slug} response")
            counts["terminal"] += record["finish_reason"] in TERMINAL
            counts["committed_tokens"] += record["generated_tokens"]
    return counts


def target(plan):
    return sum(len(plan["eligible_ids"][slug]) for slug in plan["arm_order"])


def wall_minutes(value):
    """Read Slurm's MaxTime ([days-]hh:mm:ss), rounding down."""
    if value in {"UNLIMITED", "INFINITE"}:
        return MAX_WAVE_MINUTES
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d\d):(\d\d)", value)
    if not match:
        raise ValueError(f"Unrecognized partition MaxTime: {value}")
    days, hours, minutes, _ = match.groups()
    return int(days or 0)*24*60 + int(hours)*60 + int(minutes)


def choose_slot(folder, remaining_minutes, command=scheduler.slurm, partitions=PARTITIONS):
    candidates, rejected = [], {}
    for part in partitions:
        try:
            fields = dict(token.split("=", 1) for token in
                          command(["scontrol", "-o", "show", "partition", part]).split() if "=" in token)
            if fields["PartitionName"] != part or fields["State"] != "UP":
                rejected[part] = "Partition unavailable"
                continue
            minutes = min(remaining_minutes, MAX_WAVE_MINUTES, wall_minutes(fields["MaxTime"]))
            if minutes < 1:
                rejected[part] = "No remaining whole minute"
                continue
            try:
                text = command(["sbatch", "--test-only", "--partition="+part, "--nodes=1", "--gres=gpu:4",
                                "--time="+str(minutes), "--no-requeue", "--chdir="+str(folder.parent),
                                str(folder/"sample_cont.sbatch")])
            except subprocess.TimeoutExpired:
                # A slow backfill estimate leaves the start unknown. test-only never submits a job.
                text = ""

        except subprocess.CalledProcessError as exc:
            rejected[part] = str(exc.stderr or exc)
            continue
        predicted = re.search(r"\bto start at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\b", text)
        eta = predicted[1] if predicted else None
        # Compare the same useful work, rather than rewarding short reservations for
        # ending sooner. This approximation omits future queue waits and is recalculated
        # after each actual allocation.
        capacity = minutes*60 - RELOAD_SECONDS
        work = max(0, remaining_minutes*60 - RELOAD_SECONDS)
        reloads = (max(1, math.ceil(work / capacity)) - 1) if capacity > 0 else 0
        adjusted = datetime.fromisoformat(eta).timestamp() + reloads * RELOAD_SECONDS if eta else None
        candidates.append(dict(partition=part, minutes=minutes, estimated_start=eta,
                               extra_loads=reloads, adjusted_start_epoch=adjusted))
    if not candidates:
        raise RuntimeError(f"No eligible production partition: {rejected}")
    def cost(c):
        known = c["adjusted_start_epoch"]
        return known is None, known or 0, -c["minutes"], c["partition"]
    selected = min(candidates, key=cost)
    return dict(partition=selected["partition"], minutes=selected["minutes"], partition_probes=candidates,
                rejected_partitions=rejected, reload_allowance_seconds=RELOAD_SECONDS)


def gpu_seconds(hours):
    value = Decimal(hours) * 3600
    if not value.is_finite() or value < 4*60 or value != value.to_integral_value():
        raise ValueError("GPU-hour cap must represent at least one four-GPU minute and whole GPU-seconds")
    return int(value)


def run(folder, action, job=None, *, hours=None, command=scheduler.slurm):
    workload = scheduler.Workload(read_plan, progress, target, choose_slot,
                                  "sample_cont.sbatch", "sample_cont_settle.sbatch", "bf16-sample")
    cap = gpu_seconds(hours) if hours is not None else None
    return scheduler.run(folder, action, job, command, workload=workload, cap_gpu_seconds=cap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--gpu-hours", help="Total campaign cap; required on initial launch")
    settle = sub.add_parser("settle")
    settle.add_argument("job")
    args = parser.parse_args()
    ledger = run(FOLDER, args.action, getattr(args, "job", None), hours=getattr(args, "gpu_hours", None))
    print("SAMPLED_CONTINUATION_SCHEDULER", json.dumps(ledger, allow_nan=False), flush=True)
