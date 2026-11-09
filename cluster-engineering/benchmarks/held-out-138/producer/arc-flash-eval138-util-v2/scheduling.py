"""Rolling completion-first groups with closest-length fill and a learned OOM ceiling."""
import policy


def next_group(prepared, cohorts, last_served, *, input_ceiling=None):
    if not prepared:
        return []
    if len({r['request']['task_id']for r in prepared}) != len(prepared):
        raise ValueError('Duplicate scheduling task')
    api = set(cohorts['api_solved'])
    other = set(cohorts['other'])
    if api & other or any(r['request']['task_id']not in api|other for r in prepared):
        raise ValueError('Missing/overlapping scheduling cohort')
    limit = policy.MAX_PADDED_INPUT_TOKENS if input_ceiling is None else input_ceiling
    if type(limit) is not int or not policy.CONTEXT <= limit <= policy.MAX_PADDED_INPUT_TOKENS:
        raise ValueError('Invalid learned input ceiling')
    continuing=[r for r in prepared if r['prefix']]
    if continuing:
        # Revisit earlier unfinished work; do not traverse all new tasks before continuing.
        seed=min(continuing,key=lambda r:(last_served.get(r['request']['task_id'],0),
            r['request']['task_id']not in api,len(r['ids']),r['request']['task_id']))
    else:
        seed=min(prepared,key=lambda r:(r['request']['task_id']not in api,len(r['ids']),r['request']['task_id']))
    if not policy.accepts([seed],policy.MAX_BATCH):
        raise RuntimeError('A full singleton prompt cannot be admitted')
    group=[seed]
    candidates=sorted((r for r in prepared if r is not seed),key=lambda r:(
        abs(len(r['ids'])-len(seed['ids'])),r['request']['task_id']not in api,
        not bool(r['prefix']),r['request']['task_id']))
    for row in candidates:
        candidate=group+[row]
        if (policy.accepts(candidate,policy.MAX_BATCH) and
                policy.admission_metrics(candidate)['padded_input_tokens']<=limit):
            group.append(row)
    # A valid long singleton is retained even if the learned ceiling was lowered.
    return group


def reduced_ceiling(current, attempted):
    return max(policy.CONTEXT,min(current-1,int(attempted*0.8)))


def split_group(group):
    if len(group)<2:
        raise RuntimeError('Single-request OOM cannot be split')
    ordered=sorted(group,key=lambda r:(len(r['ids']),r['request']['task_id']))
    middle=len(ordered)//2
    return policy.groups(ordered[:middle],policy.MAX_BATCH)+policy.groups(ordered[middle:],policy.MAX_BATCH)
