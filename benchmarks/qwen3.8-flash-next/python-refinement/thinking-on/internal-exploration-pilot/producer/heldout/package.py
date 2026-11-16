"""Assemble the verified native loader, decoder and task controller with the held-out scheduler."""
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

here=Path(__file__).resolve().parent
pilot=here.parent
root=pilot.parents[1]
art=root/'artifacts/cluster'
name='arc-python-heldout-length-v1'
with tempfile.TemporaryDirectory() as temp:
    d=Path(temp)/name
    d.mkdir()
    for f in ['decoding.py','protocol_pilot.py','memory_preflight.py','reference.json']:
        shutil.copyfile(pilot/f,d/f)
    # First pass retains its original limits. Retry tasks may generate longer turns.
    f=d/'decoding.py'
    f.write_text(f.read_text().replace('len(body)>=1536',
        "len(body)>=(3072 if s.get('retry_reason') else 1536)"))
    f=d/'protocol_pilot.py'
    f.write_text(f.read_text().replace("len(state['attempts']) == 4","len(state['attempts']) >= state.get('retry_turns',4)")
        .replace("32768-state['total']", "(32768+state.get('parent_total',0))-state['total']"))
    for f in ['launch.py','panel.py','score.py','retry30.py']:
        shutil.copyfile(here/f,d/f)
    rows=[json.loads(l) for l in (art/'arc-agi3-data-v1/evaluation/tasks.jsonl').read_text().splitlines()]
    ids=json.loads((art/'arc-flash-eval138-v1/data-manifest.json').read_text())['task_ids']
    assert [r['task_id'] for r in rows]==ids['evaluation'] and len(set(ids['evaluation']))==138
    assert not set(ids['evaluation']).intersection(ids['train']+ids['dev'])
    assert sum(len(r['test']) for r in rows)==185
    assert all(set(r)=={'task_id','train','test'} and all(set(q)=={'input'} for q in r['test']) for r in rows)
    shutil.copyfile(art/'arc-agi3-data-v1/evaluation/tasks.jsonl',d/'visible.jsonl')
    src=(pilot/'worker.py').read_text()
    prefix=src[:src.index('def prepare_panel')]
    body=src[src.index('def run(block):'):src.index('            if continuation:\n                panel(')]
    body=body.replace('block in BLOCKS',"block in ('A','B')").replace('path=OUT/block',"path=OUT/'allocations'/job")
    body=body.replace("start=end_time-(25 if continuation else RETRY_MINUTES[block])*60","start=int(os.environ['SLURM_JOB_START_TIME'])")
    body=body.replace('tasks,states=prepare_panel','tasks,states=prepare')
    body=body[:body.index('        if continuation:')]+body[body.index("        info['sandbox']"):]
    end=src[src.index('    except BaseException as exc:'):]
    body+="            if not info['greedy_control']['matches']:\n                raise RuntimeError('Native control mismatch')\n"
    body+='            run_panel(block,states,tasks,model,tok,m,path,info,report,start,end_time,controller)\n'
    (d/'worker.py').write_text(prefix+'from panel import prepare, run_panel\n\n'+body+end)
    for f in ['debug.sbatch','watch.sbatch']:
        script=(pilot/f).read_text().replace('arc-python-exploration-pilot-v4',name).replace('arc-explore','arc-heldout-py').replace('00:26:00','00:30:00')
        (d/f).write_text(script)
    (d/'retry30.sbatch').write_text((d/'debug.sbatch').read_text().replace('/worker.py','/retry30.py'))
    for p in d.glob('*.py'):
        compile(p.read_text(),str(p),'exec')
    with tarfile.open(sys.argv[1],'w:gz') as archive:
        archive.add(d,arcname=name)
