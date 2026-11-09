"""Length-aware groups with a strict request floor and bounded OOM backoff."""
import policy

LADDER=(1,2,4,8,12,16,18,24,32,48,64)


def next_group(prepared, cohorts, last_served, *, target=24):
    if len(prepared)<policy.MIN_BATCH:
        return []
    if target not in LADDER or len({r['request']['task_id']for r in prepared})!=len(prepared):
        raise ValueError('Invalid target or duplicate scheduling task')
    # Revisit live prefixes first. All public tasks receive the same scheduling rule.
    seeds=sorted(prepared,key=lambda r:(not bool(r['prefix']),
        last_served.get(r['request']['task_id'],0),len(r['ids']),r['request']['task_id']))
    for seed in seeds:
        nearby=sorted((r for r in prepared if r is not seed),key=lambda r:(
            abs(len(r['ids'])-len(seed['ids'])),not bool(r['prefix']),r['request']['task_id']))
        group=[seed]+nearby[:min(target,len(prepared))-1]
        while len(group)>=policy.MIN_BATCH:
            if policy.accepts(group,policy.MAX_BATCH):
                return group
            group.pop()
    return []


def after_success(target, maximum):
    return min(maximum,LADDER[min(LADDER.index(target)+1,len(LADDER)-1)])


def after_oom(target, actual):
    choices=[n for n in LADDER if n<min(target,actual)]
    return max(choices)if choices else None
