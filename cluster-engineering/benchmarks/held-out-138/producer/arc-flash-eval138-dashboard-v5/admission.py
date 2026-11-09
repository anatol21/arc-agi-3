"""Prefer eligible large groups, backing off by actual arm, device and input-width band."""
import json
from pathlib import Path

import policy
import scheduling
from bridge import digest


def hardware(torch):
    return [[p.name,p.total_memory,p.major,p.minor] for p in
            (torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count()))]


class Admission:
    def __init__(self, root, benchmark, devices, batches):
        self.devices = devices
        self.caps, self.safe = {}, {}
        seed=json.loads(Path(__file__).with_name('continuation.json').read_text())
        for trial in seed['admission_history']:
            if trial['hardware']==devices:
                self.observe(trial['arm'],trial['length_band'],trial['size'],None,failed=trial['failed'])
        for path in (root/'oom-attempts').glob('*.json'):
            trial=json.loads(path.read_text())
            if trial['sha256']!=digest({k:v for k,v in trial.items() if k!='sha256'}) or trial['benchmark_id']!=benchmark:
                raise ValueError('Invalid OOM journal')
            if trial['hardware']==devices:
                self.observe(trial['arm'],trial['length_band'],len(trial['task_ids']),trial['new_target'],failed=True)
        for trial in batches:
            if trial['hardware']==devices:
                self.observe(trial['arm'],trial['length_band'],trial['actual_batch_size'],None,failed=False)

    def observe(self, arm, band, actual, lower, *, failed):
        key=arm,band
        if failed:
            cap=scheduling.after_oom(actual,actual) if lower is None else lower
            self.caps[key]=min(self.caps.get(key,64),cap or 0)
        else:
            self.safe[key]=max(self.safe.get(key,0),actual)

    def choose(self, prepared, cohorts, last, arm):
        if len({r['request']['task_id'] for r in prepared})!=len(prepared):
            raise ValueError('Duplicate pending task')
        seeds=sorted(prepared,key=lambda r:(not bool(r['prefix']),last.get(r['request']['task_id'],0),
                                            len(r['ids']),r['request']['task_id']))
        best=([],0,None)
        for seed in seeds:
            others=sorted((r for r in seeds if r is not seed),
                          key=lambda r:(abs(len(r['ids'])-len(seed['ids'])),not bool(r['prefix']),r['request']['task_id']))
            neighbors=[seed]+others
            for n in range(min(len(neighbors),64),len(best[0]),-1):
                group=neighbors[:n]
                width=max(len(r['ids']) for r in group)+max(min(r['remaining'],policy.CHUNK_TOKENS) for r in group)
                band=(width-1)//1024
                key=arm,band
                safe=self.safe.get(key,0)
                target=next((x for x in scheduling.LADDER if x>safe),64) if safe else 24
                target=min(target,self.caps.get(key,64))
                if n<=target and policy.accepts(group,64):
                    if n>=18: return group,target,band
                    best=group,target,band
                    break
        return best
