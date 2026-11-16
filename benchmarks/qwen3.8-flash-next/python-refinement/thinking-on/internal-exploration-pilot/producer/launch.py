"""Idempotent four-slot reservation. Refuse duplicate submission on ambiguous sbatch results."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from protocol_pilot import RETRY_MINUTES, save

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
OUT=WORK/'runs'/HERE.name


def launch():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for block,minutes in RETRY_MINUTES.items():
            slot=OUT/(block+'-submission.json')
            if slot.exists():
                if not json.loads(slot.read_text()).get('id'):
                    raise RuntimeError('Ambiguous submission; reconcile '+str(slot))
                status=OUT/block/'allocation.json'
                job=json.loads(slot.read_text())['id']
                if not status.exists():
                    return
                end_record=json.loads(status.read_text())
                if (end_record.get('phase')!='ended' or end_record.get('job')!=job
                        or end_record.get('block')!=block or end_record.get('error')):
                    return
                tries=13 if sys.argv[1:]==[job] else 1
                for attempt in range(tries):
                    if attempt:
                        time.sleep(5)
                    result=subprocess.check_output(['sacct','-nPX','-j',job,'-o','State,ExitCode'],text=True).strip()
                    fields=result.split('|')
                    if fields[:2]==['COMPLETED','0:0'] or fields[0]=='TIMEOUT':
                        break
                else:
                    return
                continue
            save(slot,dict(block=block,reserved_gpu_hours=4*minutes/60))
            result=subprocess.check_output(['sbatch','--parsable',f'--time=00:{minutes:02}:00',str(HERE/'debug.sbatch'),block],cwd=WORK,text=True)
            job=result.strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError(result)
            save(slot,dict(block=block,reserved_gpu_hours=4*minutes/60,id=job))
            watch=subprocess.check_output(['sbatch','--parsable','--dependency=afterany:'+job,
                                           str(HERE/'watch.sbatch'),job],cwd=WORK,text=True).strip()
            save(slot,dict(block=block,reserved_gpu_hours=4*minutes/60,id=job,watcher=watch))
            print(block,job,'watcher',watch,flush=True)
            return

if __name__=='__main__':
    launch()
