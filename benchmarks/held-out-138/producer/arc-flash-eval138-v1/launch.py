"""Use the actual allocation end time, then run the frozen serial generator."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

if __package__:
    from experiments.cluster_benchmark_budget.deadline import allocation_budget
else:
    from deadline import allocation_budget


def launch(run):
    # Import from this new standalone package, never the old dev output directory.
    sys.path.insert(0, str(run))
    import common
    if Path(common.__file__).resolve() != (run / 'common.py').resolve():
        raise RuntimeError('Conflicting evaluation helper import')
    common.verify_package(run); common.verify_ready(run)
    budget = allocation_budget()
    output = run.parent / 'runs' / common.OUTPUT_NAME
    common.atomic_json(output / 'allocation-budget.json',
                       dict(budget, slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    print('EVALUATION_ALLOCATION ' + json.dumps(budget), flush=True)
    if time.time() >= budget['stop_at']:
        subprocess.run([sys.executable, str(run / 'score_benchmark.py')], check=True)
        return
    os.environ['ARC_BENCH_STOP_AT'] = str(budget['stop_at'])
    runpy.run_path(str(run / 'run_benchmark.py'), run_name='__main__')


if __name__ == '__main__':
    launch(Path(__file__).resolve().parent)
