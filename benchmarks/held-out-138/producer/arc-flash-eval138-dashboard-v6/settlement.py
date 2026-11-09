"""Settle only final, unique Slurm allocations from the two sealed campaign manifests."""
import json
import subprocess

PARENTS=('8146a5b796f8482bab9a5e245711fdf2','597616af9ee85f1592c8cddd8cc622bf')
JOBS={'8464839','8470348','8470445','8470549'}
CAP_GPU_SECONDS=6*3600


def settle(work):
    jobs=set()
    for run in PARENTS:
        m=json.loads((work/'cluster-ui'/run/'manifest.json').read_text())
        for wave in m['waves']:
            j=wave.get('job_id')
            if not j or j in jobs or not wave.get('handled') or wave['status']!='submitted':
                raise ValueError('Outstanding or ambiguous campaign reservation')
            jobs.add(j)
    if jobs!=JOBS: raise ValueError('Campaign jobs changed before settlement')
    lines=subprocess.check_output(['sacct','-X','-nP','-j',','.join(sorted(jobs)),'-o','JobID,State,ElapsedRaw,AllocTRES'],text=True)
    accounts={}
    for line in lines.splitlines():
        ident,state,elapsed,tres=line.split('|')[:4]
        j=ident.split('_')[0]
        if j not in jobs or ident!=j+'_0' or j in accounts or state!='COMPLETED':
            raise ValueError('Expected exactly one final array allocation per job')
        gpu=int(dict(s.split('=',1) for s in tres.split(','))['gres/gpu'])
        if gpu!=3 or int(elapsed)<=0: raise ValueError('Missing GPU accounting')
        accounts[j]=gpu*int(elapsed)
    if set(accounts)!=jobs: raise ValueError('Incomplete Slurm accounting')
    left=CAP_GPU_SECONDS-sum(accounts.values())
    if left<=0: raise ValueError('Campaign budget exhausted')
    return dict(cap_gpu_seconds=CAP_GPU_SECONDS,charged_gpu_seconds=accounts,available_gpu_seconds=left)


def check_failed_receipt(run,manifest):
    import hashlib
    r=json.loads((run/'workers/0/result.json').read_text())
    wave=manifest['waves'][-1]
    if r['state']!='failed' or r['identity']!=manifest['identity'] or r['attempt']!=wave['token'] or r['completed']!=6:
        raise ValueError('Different failed allocation or result')
    seals=r['checkpoint_files']
    for key,info in seals.items():
        p=(run/key).resolve()
        if not p.is_relative_to(run.resolve()) or p.stat().st_size!=info['size'] or hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError('Damaged checkpoint: '+key)
    files={str(p.relative_to(run)) for pattern in ('native/batches/batch-*.json','native/terminals/*.json','native/oom-attempts/*.json') for p in run.glob(pattern)}
    if not files<=set(seals):raise ValueError('Unsealed batch, failure or terminal receipt')
