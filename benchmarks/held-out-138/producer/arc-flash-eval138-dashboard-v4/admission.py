"""Restore per-shape admission limits from committed trials, including failed trials."""
import json

import policy
import scheduling
from bridge import digest


def hardware(torch):
    return [[p.name,p.total_memory,p.major,p.minor] for p in
            (torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count()))]


class Admission:
    def __init__(self, root, benchmark, devices, batches):
        self.devices = devices
        self.caps = {}
        self.safe = {}
        errors = (json.loads(p.read_text()) for p in (root/'oom-attempts').glob('*.json'))
        for r in errors:
            if r['sha256'] != digest({k:v for k,v in r.items() if k!='sha256'}) or r['benchmark_id'] != benchmark:
                raise ValueError('Untrusted OOM journal')
            if r.get('hardware') == devices:
                key = (r['arm'],r['length_band'])
                cap = r['new_target'] or 0
                self.caps[key] = min(self.caps.get(key,64),cap)
        for b in batches:
            if b.get('hardware') == devices:
                key = (b['arm'],b['length_band'])
                self.safe[key] = max(self.safe.get(key,0), b['actual_batch_size'])

    def choose(self, prepared, cohorts, last, arm):
        # The seed group fixes the comparison band, including after a retry.
        group = scheduling.next_group(prepared,cohorts,last,target=24)
        if not group:
            return [], 24, None
        width = policy.admission_metrics(group)['padded_chunk_end_width']
        band = (width-1)//1024
        key = (arm,band)
        safe = self.safe.get(key,0)
        target = next((n for n in scheduling.LADDER if n>safe),64) if safe else 24
        target = min(self.caps.get(key,64),target)
        if not target:
            return [], 0, band
        return scheduling.next_group(prepared,cohorts,last,target=target),target,band

    def observe(self, arm, band, actual, lower, *, failed):
        key = (arm,band)
        if failed:
            self.caps[key] = min(self.caps.get(key,64),lower or 0)
        else:
            self.safe[key] = max(self.safe.get(key,0),actual)
