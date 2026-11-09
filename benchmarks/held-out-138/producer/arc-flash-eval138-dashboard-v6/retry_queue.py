"""A failed cohort retries before ordinary batch admission, also across restarts."""
import policy
import scheduling


class RetryQueue:
    def __init__(self, devices, failures, consumed):
        self.pending={}
        self.selected=None
        # Legacy receipts predate explicit retry links. Try the smallest outstanding
        # targets first. A live failure is always inserted ahead of those receipts.
        ordered=sorted(failures,key=lambda f:(f['new_target'] or 0,-f['ended_at']))
        for f in ordered:
            if f['hardware']==devices and f['failed_attempt_id'] not in consumed:
                self.add(f,front=False)

    def add(self, failure, *, front=True):
        arm=failure['arm']
        queue=self.pending.setdefault(arm,[])
        parent=failure.get('retry_of')
        if parent:
            queue[:]=[r for r in queue if r['failed_attempt_id']!=parent]
        if failure['new_target']:
            if front: queue.insert(0,failure)
            else: queue.append(failure)

    def choose(self, prepared, arm, caps):
        self.selected=None
        available={r['request']['task_id']:r for r in prepared}
        queue=self.pending.get(arm,[])
        while queue:
            f=queue[0]
            peers=[available[t] for t in f['task_ids'] if t in available]
            size=min(f['new_target'],len(peers))
            while size:
                group=peers[:size]
                band=(policy.admission_metrics(group)['padded_chunk_end_width']-1)//1024
                ceiling=caps.get((arm,band),64)
                if size<=ceiling and policy.accepts(group,64):
                    self.selected=f['failed_attempt_id']
                    return group,min(f['new_target'],ceiling),band
                size-=1
            queue.pop(0)
        return None

    def committed(self, arm):
        if self.selected:
            self.pending[arm]=[r for r in self.pending[arm] if r['failed_attempt_id']!=self.selected]
        self.selected=None


def arm_turns(arms, retry_arms):
    for arm in arms:
        yield arm
        while arm in retry_arms:
            retry_arms.remove(arm)
            yield arm
