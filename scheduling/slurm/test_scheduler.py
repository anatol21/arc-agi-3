"""Offline regression tests for slot selection, budget settlement and restart behavior."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from checkpoint import digest, save_item
import scheduler


class FakeSlurm:
    def __init__(self):
        self.receipts = []
        self.accounts = {}
        self.eta = {
            "gpu-short": "2030-01-01T10:00:00",
            "gpu-long": "2030-01-02T10:00:00",
        }
        self.lose_reply = False

    def __call__(self, cmd):
        if cmd[0] == "scontrol":
            name = cmd[-1]
            limit = "00:30:00" if name == "gpu-short" else "1-00:00:00"
            return f"PartitionName={name} MaxTime={limit} State=UP"
        if cmd[0] == "sbatch":
            if "--test-only" in cmd:
                part = next(x for x in cmd if x.startswith("--partition=")).split("=")[
                    1
                ]
                return (
                    "sbatch: Job 1 to start at " + self.eta[part]
                    if self.eta[part]
                    else ""
                )
            job = str(len(self.receipts) + 100)
            self.receipts.append((job, cmd))
            if self.lose_reply:
                self.lose_reply = False
                raise subprocess.TimeoutExpired("sbatch", 30)
            return job
        if cmd[0] == "squeue" or "--name" in cmd:
            name = cmd[cmd.index("--name") + 1]
            return "".join(
                f"{j}|{name}\n"
                for j, args in self.receipts
                if f"--job-name={name}" in args
            )
        if cmd[0] == "sacct":
            j = cmd[cmd.index("-j") + 1]
            return self.accounts.get(j, "")
        raise AssertionError(cmd)


class SchedulingTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name).resolve()
        self.path = self.base / "campaign.json"
        shutil.copyfile(scheduler.HERE / "examples/campaign.json", self.path)
        self.config = scheduler.load_config(self.path)
        self.identity = digest(self.config)
        self.root = self.base / ".campaigns" / self.identity
        self.slurm = FakeSlurm()

    def call(self, *args):
        return scheduler.run(self.path, *args, command=self.slurm)

    def test_short_long_and_unknown(self):
        pick = lambda: scheduler.choose_partition(
            self.config, self.path, 120, self.slurm
        )["partition"]
        self.assertEqual(pick(), "gpu-short")
        self.slurm.eta["gpu-long"] = "2030-01-01T10:01:00"
        self.assertEqual(pick(), "gpu-long")
        self.slurm.eta = {p: None for p in self.slurm.eta}
        for order in [["gpu-long", "gpu-short"], ["gpu-short", "gpu-long"]]:
            self.config["partitions"] = order
            self.assertEqual(pick(), "gpu-long")
        self.slurm.eta["gpu-short"] = "2030-01-01T10:30:00"
        self.assertEqual(pick(), "gpu-short")

    def test_restart_charges_actual_usage_and_recovers_lost_receipt(self):
        self.slurm.lose_reply = True
        first = self.call("launch")
        self.assertEqual(first["waves"][0]["gpu"]["job"], "100")
        self.assertIn("--dependency=afterany:100", self.slurm.receipts[1][1])
        self.assertEqual(
            self.slurm.receipts[1][1][-3:],
            [str(self.path), "100", str(scheduler.HERE / "scheduler.py")],
        )
        self.assertEqual(len(self.slurm.receipts), 2)
        self.call("launch")
        self.assertEqual(len(self.slurm.receipts), 2)
        save_item(self.root, self.identity, "sample-A", 2, "partial", {"prefix": 2})
        self.slurm.accounts["100"] = "100|PREEMPTED|0:15|900|cpu=2,gres/gpu=1"
        self.slurm.eta["gpu-long"] = "2030-01-01T09:00:00"
        second = self.call("settle", "100")
        self.assertEqual(second["charged_gpu_seconds"], 900)
        self.assertIn("--time=105", self.slurm.receipts[2][1])
        for item in self.config["items"]:
            save_item(self.root, self.identity, item, 3, "finished", {})
        self.slurm.accounts["102"] = "102|COMPLETED|0:0|1000|cpu=2,gres/gpu=1"
        final = self.call("settle", "102")
        self.assertEqual(final["phase"], "complete")
        self.assertEqual(final["charged_gpu_seconds"], 1900)
        self.assertEqual(len(self.slurm.receipts), 4)

    def test_budget_and_failure(self):
        self.call("launch")
        self.slurm.accounts["100"] = "100|FAILED|1:0|62|gres/gpu=1"
        self.assertEqual(self.call("settle", "100")["phase"], "needs_attention")
        self.assertEqual(len(self.slurm.receipts), 2)

    def test_round_remaining_down(self):
        self.slurm.eta["gpu-long"] = "2030-01-01T08:00:00"
        self.call("launch")
        self.slurm.accounts["100"] = "100|TIMEOUT|0:15|7179|gres/gpu=1"
        self.assertEqual(self.call("settle", "100")["phase"], "budget_reached")
        self.assertEqual(len(self.slurm.receipts), 2)

    def test_unknown_intent_never_blindly_retries(self):
        def lose(cmd):
            if "--parsable" in cmd:
                raise subprocess.TimeoutExpired("sbatch", 30)
            return self.slurm(cmd)

        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "Ambiguous prior submission"):
                scheduler.run(self.path, "launch", command=lose)
        self.assertEqual(len(self.slurm.receipts), 0)

    def test_documented_local_worker(self):
        shutil.copyfile(
            scheduler.HERE / "examples/example_worker.py",
            self.base / "example_worker.py",
        )
        command = [sys.executable, str(scheduler.HERE / "worker.py"), str(self.path)]
        for _ in range(5):
            subprocess.run(command, cwd=self.base, check=True, capture_output=True)
        self.assertEqual(len(list((self.root / "items").glob("*.json"))), 4)
        self.assertEqual(self.call("status")["finished"], 4)


if __name__ == "__main__":
    unittest.main()
