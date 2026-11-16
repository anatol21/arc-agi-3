"""Serialize dev allocations, alternate both conditions, and charge actual GPU seconds."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'runs'/HERE.name
sys.path.insert(0,str(HERE))
from protocol_pilot import save

PRIOR={'8526529','8534498','8543788','8544056','8544341','8544471','8544533','8583418'}
CAP_SECONDS=225324  # 62.59 GPU-hours, including prior internal pilot


def accounting(ids):
    text=subprocess.check_output(['sacct','-nPX','-j',','.join(sorted(ids)),'-o','JobID,State,ElapsedRaw,AllocTRES'],text=True)
    records={}
    for line in text.splitlines():
        job,state,seconds,tres=line.split('|')[:4]
        if job in ids:
            if 'gres/gpu=4' not in tres.split(','):
                raise RuntimeError('Unexpected GPU allocation '+job)
            records[job]=(state,int(seconds)*4)
    if records.keys()!=ids:
        raise RuntimeError('GPU records pending')
    return records


def launch():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        hist=[json.loads(p.read_text()) for p in sorted(OUT.glob('slot-*.json'))]
        if any(not r.get('id') for r in hist):
            raise RuntimeError('Reconcile ambiguous submission')
        if hist and sys.argv[1:]!=[hist[-1]['id']]:
            return
        ids=PRIOR|{r['id'] for r in hist}
        for i in range(12):
            try:
                records=accounting(ids)
                if all(s in {'FAILED','TIMEOUT','COMPLETED'} for s,_ in records.values()):
                    break
            except RuntimeError:
                if i==11: raise
            time.sleep(5)
        else:
            raise RuntimeError('Allocation accounting unsettled')
        if hist:
            last=hist[-1]['id']
            last_out=OUT/'retry30' if hist[-1].get('retry30') else OUT
            end=last_out/'allocations'/last/'allocation.json'
            if not end.exists() or json.loads(end.read_text()).get('phase')!='ended':
                raise RuntimeError('Prior worker requires review: '+last)
        if hist:
            from score import score
            score(HERE,last_out)
        minutes=min(30,(CAP_SECONDS-sum(cost for _,cost in records.values()))//240)
        live=[a for a in 'AB' if sum(bool(json.loads(p.read_text())['done']) for p in (OUT/a/'tasks').glob('*.json'))<138]
        retry30=not live
        if retry30:
            from retry30 import targets
            live=[a for a in 'AB' if not all(
                (p:=OUT/'retry30'/a/'tasks'/(key+'.json')).exists() and json.loads(p.read_text())['done']
                for key in targets(OUT,a))]
        if not live or minutes<10:
            print('Stopped',live,minutes,flush=True)
            return
        arm=min(live,key=lambda a:sum(r['arm']==a and bool(r.get('retry30'))==retry30 for r in hist))
        slot=OUT/f'slot-{len(hist):04d}.json'
        row=dict(arm=arm,minutes=minutes,retry30=retry30)
        save(slot,row)
        job=subprocess.check_output(['sbatch','--parsable',f'--time=00:{minutes:02}:00',str(HERE/('retry30.sbatch' if retry30 else 'debug.sbatch')),arm],cwd=HERE.parent,text=True).strip().split(';')[0]
        if not job.isdigit():raise ValueError(job)
        row['id']=job
        save(slot,row)
        row['watcher']=subprocess.check_output(['sbatch','--parsable','--dependency=afterany:'+job,str(HERE/'watch.sbatch'),job],cwd=HERE.parent,text=True).strip().split(';')[0]
        save(slot,row)
        print(row,flush=True)

if __name__=='__main__':
    launch()
