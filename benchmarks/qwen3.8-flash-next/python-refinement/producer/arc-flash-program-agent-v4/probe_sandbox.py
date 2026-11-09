"""CPU-only validation of the actual installed sandbox, before any GPU request."""
import argparse
import json
import os
import subprocess
from pathlib import Path

if __package__:
    from .setup_runtime import NAME,ready,atomic_json,digest,utcnow
    from .sandbox import preflight
else:
    from setup_runtime import NAME,ready,atomic_json,digest,utcnow
    from sandbox import preflight


def probe(folder):
    _,recipe=ready(folder)
    output=folder.parent/'runs'/NAME
    evidence={'phase':'running','benchmark_id':recipe['benchmark_id'],'job_id':os.environ.get('SLURM_JOB_ID'),
              'started_at':utcnow(),'gpu_requested':False,'package_manifest_sha256':digest(folder/'package-manifest.json')}
    path=output/'cpu-sandbox-probe.json'
    atomic_json(path,evidence)
    try:
        evidence['bubblewrap_version']=subprocess.run(['bwrap','--version'],check=True,capture_output=True,text=True,timeout=10).stdout.strip()
        evidence['checks']=preflight()
        evidence['phase']='passed'
    except BaseException as failure:
        evidence.update(phase='failed',error=repr(failure));raise
    finally:
        evidence['finished_at']=utcnow();atomic_json(path,evidence)
        print('CPU_SANDBOX_PROBE '+json.dumps(evidence),flush=True)
    return evidence


def verify_probe(folder):
    _,recipe=ready(folder)
    path=folder.parent/'runs'/NAME/'cpu-sandbox-probe.json'
    evidence=json.loads(path.read_text())
    if (evidence.get('phase')!='passed' or evidence.get('benchmark_id')!=recipe['benchmark_id']
        or evidence.get('package_manifest_sha256')!=digest(folder/'package-manifest.json')):
        raise RuntimeError('CPU sandbox probe missing, failed, or stale; do not request GPUs yet')
    required=('identity_execution','stdout_capture','workspace_hidden','network_probe_blocked',
              'environment_allowlist','readonly_filesystem_probe','gpu_devices_hidden')
    if any(evidence.get('checks',{}).get(key) is not True for key in required):
        raise RuntimeError('CPU sandbox evidence is incomplete')
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    folder=Path(__file__).resolve().parent
    if args.verify:
        verify_probe(folder);print('CPU_SANDBOX_RECEIPT_VERIFIED; GPU debug may now be submitted',flush=True)
    else:probe(folder)
