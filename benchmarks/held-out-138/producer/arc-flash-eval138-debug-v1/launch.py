"""Thirty-minute fixed-prefix debug, with native deadline and durable predictions."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from debug_common import NAME, dependencies


def launch(folder):
    native, recipe, m, receipt = dependencies(folder)
    budget = m['deadline'].allocation_budget()
    output = folder.parent / 'runs' / NAME
    output.mkdir(parents=True, exist_ok=True)
    m['common'].atomic_json(output/'allocation-budget.json', dict(budget, slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    print('EVAL_DEBUG_ALLOCATION ' + json.dumps(budget), flush=True)
    os.environ['ARC_BENCH_STOP_AT'] = str(budget['stop_at'])
    try:
        if time.time() < budget['stop_at']:
            spec = importlib.util.spec_from_file_location('_eval_debug_runner', folder/'debug_runner.py')
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            runner.run_benchmark(native)
    finally:
        # Only this separate CPU process reads held-out query answers.
        result = subprocess.run([sys.executable, str(folder/'score_debug.py')], check=False)
        if result.returncode:
            print('DEBUG_SCORING_FAILED', file=sys.stderr, flush=True)
            if sys.exc_info()[0] is None:
                raise RuntimeError('Debug scoring failed')


if __name__ == '__main__':
    launch(Path(__file__).resolve().parent)
