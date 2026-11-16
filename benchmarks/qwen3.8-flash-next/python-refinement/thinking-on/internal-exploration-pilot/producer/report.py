"""CPU-only scoring from committed query predictions; include unfinished assignments."""

import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
sys.path[:0]=[str(HERE),str(WORK/'arc-flash-python-nothink-v2')]
from protocol_pilot import PANEL, BLOCKS, save
from records import grid as validate_grid
from runtime import ready


def report():
    benchmark,_,m=ready(WORK/'arc-flash-python-nothink-v2')
    labels={r['task_id']:r['outputs'] for r in m['protocol'].rows(benchmark/'answers.scorer-only.jsonl')}
    output=WORK/'runs'/HERE.name
    rows=[]
    for block in BLOCKS:
        for key in PANEL[block[1]]:
            file=output/block/'tasks'/(key+'.json')
            s=json.loads(file.read_text()) if file.exists() else {}
            row=dict(block=block,arm=block[0],task=key,finished=s.get('done',False),
                     tokens=s.get('total',0),probes=len(s.get('probes',[])))
            for stage in ['first','selected']:
                grids=s.get(stage)
                if row['finished'] and grids is not None:
                    try:
                        for g in grids:
                            validate_grid(g)
                    except (ValueError,TypeError):
                        row[stage]=False
                    else:
                        row[stage]=grids==labels[key]
                else:
                    row[stage]=None
            rows.append(row)
    totals={a:dict(assigned=8,finished=sum(r['finished'] for r in rows if r['arm']==a),
                   selected_solves=[r['task'] for r in rows if r['arm']==a and r['selected']])
            for a in 'AB'}
    save(output/'panel-report.json',dict(rows=rows,arms=totals))
    print(json.dumps(totals))
if __name__=='__main__':
    report()
