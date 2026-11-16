"""Limit each new GPU reservation by accounting for every prior GPU allocation."""
import subprocess
from decimal import Decimal

PRIOR=['8526529','8534498','8543788','8544056','8544341','8544471','8544533']


def affordable_minutes(out,planned):
    import json
    ids=PRIOR+[json.loads(slot.read_text())['id'] for slot in out.glob('*-submission.json')]
    lines=subprocess.check_output(['sacct','-nPX','-j',','.join(ids),'-o','JobID,ElapsedRaw,AllocTRES,State'],text=True).splitlines()
    seconds={}
    for line in lines:
        job,elapsed,tres,state=line.split('|')[:4]
        if job not in ids:
            continue
        if 'gres/gpu=4' not in tres.split(',') or state not in {'COMPLETED','FAILED','TIMEOUT'}:
            raise RuntimeError('Unsettled or unexpected prior GPU charge '+job)
        seconds[job]=int(elapsed)*4
    if set(seconds)!=set(ids):
        raise RuntimeError('Missing GPU accounting records')
    remaining=int(Decimal('14.59')*3600)-sum(seconds.values())
    minutes=min(planned,remaining//240)
    if minutes<=0:
        raise RuntimeError('GPU-hour budget exhausted')
    return minutes
