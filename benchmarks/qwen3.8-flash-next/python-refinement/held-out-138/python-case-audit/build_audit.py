"""Regenerate the per-task ledger, counts and grid figures from the archived observations."""
from collections import Counter
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def shape(grid):
    return (len(grid), len(grid[0])) if grid else None

def compare(query):
    saved, expected = query['saved'], query['reference']
    ps, rs = shape(saved), shape(expected)
    cells = sum(x != y for a, b in zip(saved, expected) for x, y in zip(a, b)) if ps == rs else None
    return dict(predicted_shape=ps, reference_shape=rs, wrong_cells=cells, copies_input=saved==query['input'], exact=saved==expected)

def classify(r, queries):
    if r['status'] != 'wrong' and r['status'] != 'no_grid':
        return r['status']
    if r['status']=='wrong':
        if any(q['predicted_shape'] is None for q in queries): return 'missing_query_grid'
        if any(q['predicted_shape'] != q['reference_shape'] for q in queries): return 'wrong_dimensions'
        if all(q['copies_input'] for q in queries): return 'copied_input'
        return 'wrong_cells'
    attempts = r['attempts']
    if attempts and attempts[-1]['end']=='context_limit':
        assert r['generated_tokens']==0
        return 'context_limit'
    assert r['stop_reason']=='task_time_limit'
    if not attempts:
        f = r['prefix']['flags']
        assert r['prefix']['after_boundary'] and not f['complete_fenced_block']
        if f['function_definition']: return 'unfinished_function'
        if f['fence']: return 'fence_and_comments'
        return 'no_function_or_fence'
    errors = {a['feedback'].get('error') for a in attempts}
    assert all(a['code'] is None for a in attempts)
    if errors=={'Missing top-level solve(grid) function'}: return 'unrecognized_probe'
    assert errors=={'Expected exactly one Python code block'}
    return 'wrong_code_block_format'

def turn_evidence(attempt):
    if attempt['code']:
        return 'Accepted solver; %d/%d demonstrations exact' % (attempt['feedback']['passed'],attempt['feedback']['total'])
    text = attempt['submitted_text'].strip()
    error = attempt['feedback'].get('error', '')
    if not text:
        output = 'empty final text'
    elif text in (')', ']'):
        output = 'only a closing bracket'
    elif text == chr(96)*3+'python':
        output = 'only an opening Python fence'
    elif text.count(chr(96)*3) == 1:
        output = 'one fence delimiter'
    elif chr(96)*3 not in text:
        output = 'text without a fenced block'
    else:
        output = 'fenced response'
    return f"{attempt['end']} / {error} / {output}"


def main():
    data=json.loads((HERE/'case-records.json').read_text())
    prior=json.loads((HERE/'prior-results.json').read_text())['tasks']
    rows=[]
    totals={a:Counter() for a in 'AB'}
    for r in data['rows']:
        p=prior[r['task_id']]
        q=[compare(z) for z in r['query']]
        key=classify(r,q)
        totals[r['arm']][key]+=1
        n=r['selected_candidate_number']
        best=r['attempts'][n-1]['feedback'] if n else {}
        rows.append(dict(arm=r['arm'],task_id=r['task_id'],outcome=r['status'],failure_or_status=key,
            generated_tokens=r['generated_tokens'],active_seconds=round(r['active_seconds'],2),generation_seconds=round(r['generation_seconds'],2),
            stop_reason=r['stop_reason'] or (r['attempts'][-1]['end'] if r['attempts'] else ''),
            finished_candidates=len(r['attempts']),turn_evidence=' | '.join(turn_evidence(a) for a in r['attempts']),selected_candidate=n,selected_demo_matches=best.get('passed'),
            selected_demo_total=best.get('total'),exploratory_calls=r['probes'],expected_grids=r['expected_grids'],returned_grids=r['returned_grids'],
            final_channel_prefix_tokens=r['prefix']['after_think'],query_comparisons=json.dumps(q,separators=(',',':')) if q else '',
            nf4_base_exact=p['strict']['base'],nf4_checkpoint660_exact=p['strict']['finetuned'],
            bf16_low_nucleus_exact=p['strict']['low_nucleus'],bf16_instruct_exact=p['strict']['instruct_sampling'],
            bf16_greedy_pilot_exact=p['bf16_greedy_pilot'],openrouter_exact_trials=p['openrouter_exact_trials'],
            openrouter_trials=p['openrouter_trials'],journal_sha256=r['journal_sha256']))
    assert len({(r['arm'],r['task_id']) for r in rows}) == len(rows)==276
    assert all(sum(x.values())==138 for x in totals.values())
    with (HERE/'per-task-audit.csv').open('w',newline='') as fp:
        wr=csv.DictWriter(fp,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (HERE/'counts.json').write_text(json.dumps(totals,indent=2,sort_keys=True)+'\n')
    panels={}
    for arm,task,name in [('B','e95e3d8e','solved-tile'),('B','97d7923e','wrong-cells'),('B','2b83f449','copied'),('A','898e7135','wrong-size')]:
        record=next(r for r in data['rows'] if (r['arm'],r['task_id'])==(arm,task))
        q=record['query'][0]
        if name=='solved-tile':
            tile=[row[:8] for row in q['saved'][:8]]
            assert q['saved']==q['reference'] and all(v==tile[i%8][j%8] for i,row in enumerate(q['saved']) for j,v in enumerate(row))
            labels=[('Query input (black = missing)',q['input']),('Recovered tile (8 by 8)',tile),('Saved program output',q['saved'])]
        else:
            labels=[('Query input',q['input']),('Saved program output',q['saved']),('Reference',q['reference'])]
        panels[name]=[dict(label=label,grid=grid) for label,grid in labels]
    (HERE/'figures.json').write_text(json.dumps(panels,separators=(',',':'))+'\n')

if __name__=='__main__':
    main()
