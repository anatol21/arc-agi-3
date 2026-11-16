"""Run separately on CPU after generation. Count a solve only for all exact query grids."""
import json
from pathlib import Path
from protocol_pilot import save

def score(here, out=None):
    out=out or here.parent/'runs'/here.name
    answers={r['task_id']:r['outputs'] for r in map(json.loads,(here.parent/'arc-flash-eval138-v1/answers.scorer-only.jsonl').read_text().splitlines())}
    visible={json.loads(l)['task_id'] for l in (here/'visible.jsonl').read_text().splitlines()}
    if set(answers)!=visible or len(answers)!=138:
        raise ValueError('Scoring set mismatch')
    from retry30 import limit_reason
    summary={}
    ledger=[]
    for arm in 'AB':
        complete=[]
        for key in sorted(visible):
            p=out/arm/'tasks'/(key+'.json')
            if not p.exists():
                continue
            row=json.loads(p.read_text())
            seconds=row.get('active_seconds',0)
            total=row['total']
            ledger.append(dict(task_id=key,arm=arm,done=row['done'],
                active_seconds=seconds,generation_seconds=row.get('generation_seconds',0),
                generated_tokens=total,phase_seconds=seconds-row.get('parent_seconds',0),
                phase_tokens=total-row.get('parent_total',0),stop_reason=limit_reason(row),
                retry_reason=row.get('retry_reason'),attempts=len(row['attempts']),
                exact=(row.get('selected')==answers[key]) if row['done'] else None))
            if row['done']:
                complete.append(dict(task_id=row['task'],correct=row.get('selected')==answers[row['task']],stop_reason=row.get('stop_reason')))
        summary[arm]=dict(finished=len(complete),solved=sum(r['correct'] for r in complete),tasks=complete)
    save(out/'scores.json',summary)
    save(out/'task-usage.json',ledger)
