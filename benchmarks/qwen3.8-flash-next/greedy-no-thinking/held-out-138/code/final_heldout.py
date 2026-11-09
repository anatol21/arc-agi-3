"""Produce a paired audit of cluster direct-answer runs using the original 138-task scorer.

Run on the cluster with the workspace as the argument. Source precedence is
chronological by run; duplicate archive/terminal representations within one run
are collapsed before comparing runs.
"""
import hashlib
import json
from pathlib import Path
import sys

ARMS = ('base', 'finetuned')
SERIAL = 'arc-flash-eval138-v1'
HISTORICAL = ['arc-flash-eval138-debug-v1', 'arc-flash-eval138-pilot-v1',
              'arc-flash-eval138-util-v1', 'arc-flash-eval138-util-v2',
              'arc-flash-eval138-util-v3', 'arc-flash-conv-rebalance-v1', SERIAL]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def aggregate(work):
    package = work/SERIAL
    sys.path.insert(0, str(package))
    import protocol
    answers_path = package/'answers.scorer-only.jsonl'
    answers = protocol.rows(answers_path)
    ids = {a['task_id'] for a in answers}
    assert len(ids)==len(answers)==138 and sum(len(a['outputs']) for a in answers)==185
    native = load(package/'benchmark-recipe.json')
    variants = {}
    for key in ('seed','do_sample','num_beams','enable_thinking','context','max_new_tokens'):
        assert native['generation'][key]==dict(seed=42,do_sample=False,num_beams=1,enable_thinking=False,
                                               context=18432,max_new_tokens=8192)[key]
    sources = []
    for name in HISTORICAL:
        root = work/'runs'/name
        if any((root/'batches').glob('batch-*.json')) or (root/'predictions.jsonl').exists():
            sources.append((name,root,name))
    for version in range(1,7):
        name=f'arc-flash-eval138-dashboard-v{version}'
        recipe = load(work/name/'recipe.json')
        for receipt in sorted((work/'cluster-ui').glob('*/native/source-receipt.json')):
            value=load(receipt)
            if value['benchmark_id']==recipe['benchmark_id']:
                assert value['fingerprints']['recipe']==sha(work/name/'recipe.json')
                assert value['fingerprints']['package']==sha(work/name/'package-manifest.json')
                sources.append((name,receipt.parent,receipt.parent.parent.name))
    evidence={}
    def remember(path):
        evidence[str(path.relative_to(work))]=sha(path)
    remember(answers_path)
    for name,root,source in sources:
        serial=(root/'predictions.jsonl').is_file()
        cfg=native if serial else load(work/name/'recipe.json')
        if not serial:
            assert cfg['source_benchmark_id']==native['benchmark_id']
            assert cfg['checkpoint']==native['checkpoint'] and cfg['adapter_sha256']==native['adapter_sha256']
            assert cfg['generation']==native['generation']
        by_pair={}
        def include(r):
            if r['finish_reason'] not in ('token_limit','eos'): return
            key=r['task_id'],r['arm']
            assert key[0] in ids and key[1] in ARMS and r['benchmark_id']==cfg['benchmark_id']
            assert not r.get('enable_thinking',False)
            if key in by_pair:
                old=by_pair[key]
                assert all(old[k]==r[k] for k in ('text','finish_reason','completion_token_ids','request_id'))
            by_pair[key]=r
        if serial:
            f=root/'predictions.jsonl'; remember(f)
            for r in protocol.rows(f):include(r)
        else:
            for idx,f in enumerate(sorted((root/'batches').glob('batch-*.json')),1):
                remember(f); b=load(f)
                assert b['sequence']==idx and b['sha256']==digest({k:v for k,v in b.items() if k!='sha256'})
                assert b['benchmark_id']==cfg['benchmark_id']
                for r in b['rows']:include(r)
            for f in sorted((root/'terminals').glob('*.json')):
                remember(f); receipt=load(f)
                assert receipt['sha256']==digest(receipt['row'])
                include(receipt['row'])
        pred={k:dict(v,text='' if v['finish_reason']=='token_limit' else v['text']) for k,v in by_pair.items()}
        _,details=protocol.score(answers,pred)
        for scored in details:
            key=scored['task_id'],scored['arm']; row=by_pair[key]
            variants.setdefault(key,[]).append(dict(**scored,package=name,run=source,
                job=row.get('slurm_job_id'),response_sha256=hashlib.sha256(row['text'].encode()).hexdigest()))
    assert set(variants)=={(t,a) for t in ids for a in ARMS}
    def summary(choices):
        solved={a:{t for (t,arm),r in choices.items() if a==arm and r['task_correct']} for a in ARMS}
        return dict(base=sorted(solved['base']),finetuned=sorted(solved['finetuned']),
                    both=sorted(solved['base']&solved['finetuned']),base_only=sorted(solved['base']-solved['finetuned']),
                    finetuned_only=sorted(solved['finetuned']-solved['base']),neither=len(ids-solved['base']-solved['finetuned']))
    first={k:rows[0] for k,rows in variants.items()}
    prefer_serial={k:next((r for r in rows if r['package']==SERIAL),rows[0]) for k,rows in variants.items()}
    any_solve={k:next((r for r in rows if r['task_correct']),rows[0]) for k,rows in variants.items()}
    return dict(tasks=len(ids),query_grids=sum(len(a['outputs']) for a in answers),completed_pairs=len(variants),
                source_order=[name for name,_,_ in sources],first_completion=summary(first),
                prefer_full_serial=summary(prefer_serial),any_observed_solve=summary(any_solve),
                repeated_pairs=sum(len(rs)>1 for rs in variants.values()),
                differing_solve_status=[dict(task_id=t,arm=a) for (t,a),rs in sorted(variants.items()) if len({r['task_correct'] for r in rs})>1],
                tasks_by_id={t:{a:variants[t,a] for a in ARMS} for t in sorted(ids)},source_sha256=evidence)


if __name__=='__main__':
    print(json.dumps(aggregate(Path(sys.argv[1]))))
