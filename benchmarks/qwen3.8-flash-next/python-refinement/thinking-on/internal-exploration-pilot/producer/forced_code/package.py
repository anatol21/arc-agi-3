"""Build a separate campaign package while reusing the native loader and controller."""
import hashlib,json,tarfile,tempfile,shutil,sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parent
with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp)/'arc-python-forced-code-v1'
    folder.mkdir()
    for name in ['worker.py','reference.json','memory_preflight.py','protocol_pilot.py','decoding.py','launch.py','report.py','debug.sbatch','watch.sbatch']:
        shutil.copy2(root/name,folder/name)
    shutil.copy2(here/'continuation.py',folder/'continuation.py')
    shutil.copy2(here/'budget.py',folder/'budget.py')
    hashes={}
    with tarfile.open(root.parent.parent/'artifacts/cluster/arc-exploration-paired-final/journals.tgz') as t:
        for item in t:
            if '/tasks/' in item.name and item.isfile():
                block,_,name=item.name.split('/')
                hashes[block+'/'+Path(name).stem]=hashlib.sha256(t.extractfile(item).read()).hexdigest()
    assert len(hashes)==16
    (folder/'parents.json').write_text(json.dumps(hashes,sort_keys=True))
    for name in ['debug.sbatch','watch.sbatch','launch.py']:
        p=folder/name
        s=p.read_text().replace('arc-python-exploration-pilot-v4',folder.name).replace('00:26:00','00:25:00')
        if name=='launch.py':
            s=s.replace('from protocol_pilot import RETRY_MINUTES, save','from protocol_pilot import BLOCKS, save\nRETRY_MINUTES=dict.fromkeys(BLOCKS,25)')
            s=s.replace('            save(slot,dict(block=block,reserved_gpu_hours=4*minutes/60))',
                        '            from budget import affordable_minutes\n            minutes=affordable_minutes(OUT,minutes)\n            save(slot,dict(block=block,reserved_gpu_hours=4*minutes/60))')
        p.write_text(s)
    with tarfile.open(sys.argv[1],'w:gz') as t:
        t.add(folder,arcname=folder.name)
