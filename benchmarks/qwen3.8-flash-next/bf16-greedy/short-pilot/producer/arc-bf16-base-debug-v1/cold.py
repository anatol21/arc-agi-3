import hashlib,json,shutil,os,tempfile,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def local_weights(workspace,plan,report):
    source=Path(plan['model_path'])
    manifest=(workspace/'arc-flash-run-v6/model-verified.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest()!=plan['model_manifest_sha256']:
        raise ValueError("Original model manifest changed")
    files=json.loads(manifest)['files']
    total=sum(info['bytes'] for info in files.values())
    tmp=Path(os.environ['TMPDIR'])
    report['staging_mode']='verified_node_local'
    if shutil.disk_usage(tmp).free<total*1.03:
        report['staging_mode']='verified_direct_fallback'
    start=time.monotonic()
    with tempfile.TemporaryDirectory(dir=tmp,prefix='arc-bf16-') as d:
        dest=Path(d)
        copying=report['staging_mode']=='verified_node_local'
        def prepare(name):
            src=source/name
            target=dest/name if copying else src
            if copying:
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src,target)
            with target.open('rb') as fp:
                sha=hashlib.file_digest(fp,'sha256').hexdigest()
            if target.stat().st_size!=files[name]['bytes'] or sha!=files[name]['sha256']:
                raise ValueError(f"Invalid original weight file: {name}")
        workers=min(32,len(os.sched_getaffinity(0)))
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(prepare,files))
        report.update(copy_workers=workers, verified_files=len(files),staged_bytes=total if copying else 0,preparation_seconds=time.monotonic()-start)
        print('BF16_WEIGHT_STAGING',json.dumps(report),flush=True)
        yield dest if copying else source
