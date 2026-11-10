"""Rebuild the 138-task paired report from published terminal responses."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy

ARMS = ('low_nucleus', 'instruct_sampling')
ROOT = Path(__file__).resolve().parent


def verify(responses, output, nf4, references=None):
    rows = {arm: {} for arm in ARMS}
    with (nf4 / 'evidence' / 'paired-results.csv').open() as stream:
        baseline = {r['task_id']: r for r in csv.DictReader(stream)}
        cohort = {t: int(r['query_grids']) for t, r in baseline.items()}
    if len(cohort) != 138 or sum(cohort.values()) != 185:
        raise ValueError('Unexpected public task cohort')
    for line in responses.read_text().splitlines():
        row = json.loads(line)
        t, arm = row['task_id'], row['strategy']
        if t not in cohort or t in rows[arm] or row['finish_reason'] not in ('eos','token_limit'):
            raise ValueError(f'Invalid or duplicate terminal observation {arm}/{t}')
        flags = row['grid_correct']
        if len(flags) != cohort[t] or any(type(x) != bool for x in flags) or row['task_correct'] != all(flags):
            raise ValueError(f'Invalid score {arm}/{t}')
        if hashlib.sha256(row['text'].encode()).hexdigest() != row['response_sha256']:
            raise ValueError(f'Text digest mismatch {arm}/{t}')
        rows[arm][t] = row
    for arm, data in rows.items():
        if data.keys() != cohort.keys():
            raise ValueError(f'{arm}: {len(data)}/138 terminal responses')
        if sum(x['stage']=='pilot' for x in data.values())!=12 or sum(x['stage']=='continuation' for x in data.values())!=126:
            raise ValueError(f'{arm}: unexpected pilot and continuation counts')
    if references:
        score = runpy.run_path(str(nf4 / 'producer/arc-flash-eval138-v1/protocol.py'))['score']
        answers = []
        for t in sorted(cohort):
            outputs = [ [v['output'] for v in json.loads((ref/(t+'.json')).read_bytes())['test']]
                       for ref in references if (ref/(t+'.json')).is_file() ]
            if not outputs or any(xs != outputs[0] for xs in outputs):
                raise ValueError(f'Missing or inconsistent reference for {t}')
            answers.append(dict(task_id=t,outputs=outputs[0]))
        for arm, data in rows.items():
            _, detail = score(answers,{(t,'base'):row for t,row in data.items()})
            for result in detail:
                if any(result[k] != data[result['task_id']][k] for k in ('task_correct','grid_correct','format_error')):
                    raise ValueError(f'Rescoring mismatch {arm}/{result["task_id"]}')
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arm, data in rows.items():
        policies = {json.dumps(row['sampling'],sort_keys=True) for row in data.values()}
        if len(policies) != 1: raise ValueError(f'Mixed sampling settings for {arm}')
        summary[arm] = dict(sampling=json.loads(policies.pop()), tasks=len(data), grids=sum(cohort.values()),
                            exact_task_ids=sorted(t for t,r in data.items() if r['task_correct']),
                            exact_grids=sum(sum(r['grid_correct']) for r in data.values()),
                            format_errors=sum(r['format_error'] is not None for r in data.values()),
                            token_limited=sum(r['finish_reason']=='token_limit' for r in data.values()))
    low, instruct = [set(summary[arm]['exact_task_ids']) for arm in ARMS]
    summary['paired'] = dict(both=sorted(low & instruct), low_nucleus_only=sorted(low - instruct),
                             instruct_sampling_only=sorted(instruct - low), neither=len(cohort) - len(low | instruct))
    with (output/'paired-results.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['task_id','query_grids',*ARMS,'nf4_base_greedy_exact'])
        for t,n in sorted(cohort.items()):
            writer.writerow([t,n,*(rows[a][t]['task_correct'] for a in ARMS), baseline[t]['base_correct']])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses',type=Path,default=ROOT/'responses.jsonl')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--nf4',type=Path,default=ROOT.parent/'greedy-no-thinking/held-out-138')
    parser.add_argument('--references',type=Path,nargs='+')
    args=parser.parse_args()
    verify(args.responses,args.out,args.nf4,args.references)
