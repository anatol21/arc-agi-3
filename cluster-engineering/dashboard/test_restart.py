import os
import tempfile
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

import runtime
import core
import sdk


class RestartTests(unittest.TestCase):
    def test_queue_order_with_unknown_runtime(self):
        cfg = core.normalize_config({'command':['work']})
        parts = [dict(name=n,time_limit='00:30:00',queue_seconds=q)
                 for n,q in [('short',79200),('dev',0),('unknown',None)]]
        for order in (parts,parts[::-1]):
            self.assertEqual([p['name'] for p in core.estimate_plans(cfg,order)],['dev','short','unknown'])

    def test_fallback_budget_and_queue_permissions(self):
        cfg=core.normalize_config({'command':['work'],'gpus':3,'budget':{'gpu_hours':1.5}})
        profile=dict(dev_permission='approved',dev_run_kinds=[cfg['kind']],normal_min_seconds=7200,queue_threshold_seconds=10800)
        rows=[dict(name=n,available=True,time_limit=t,role=r,queue_seconds=q) for n,r,q,t in
              [('main','normal',15000,'06:00:00'),('short','short',10,'02:00:00'),('dev','development',60,'00:30:00')]]
        self.assertEqual(runtime.choose_plan(dict(config=cfg,profile=profile),{'partitions':rows},8)['name'],'dev')
        rows[0]['queue_seconds']=None
        with self.assertRaises(runtime.BudgetReached):
            runtime.choose_plan(dict(config=cfg,profile=profile),{'partitions':rows},8)

    def test_retry_starts_empty_and_concurrent_reader_sees_only_published_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'weights'
            fp = {'source':'bytes'}
            def fail(stage):
                (stage/'matrix').write_bytes(b'half')
                raise InterruptedError
            with self.assertRaises(InterruptedError):
                sdk.prepare_cache(path,fp,fail)
            ready, release = Event(), Event()
            def build(stage):
                with (stage/'matrix').open('xb') as f: f.write(b'whole')
                ready.set()
                release.wait(5)
                return ['matrix']
            with ThreadPoolExecutor(2) as pool:
                first = pool.submit(sdk.prepare_cache,path,fp,build)
                try:
                    self.assertTrue(ready.wait(2))
                    self.assertFalse(sdk.cache_ready(path,fp))
                    second = pool.submit(sdk.prepare_cache,path,fp,build)
                finally: release.set()
                self.assertFalse(first.result())
                self.assertTrue(second.result())
            self.assertTrue(sdk.verify_cache(path,fp))

    def test_killed_builder_reclaims_staging_before_retry(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'weights'
            script="""import os,sys
from pathlib import Path
from sdk import prepare_cache
def build(stage):
    (stage/'matrix').write_bytes(b'incomplete')
    os._exit(19)
prepare_cache(Path(sys.argv[1]),{'id':'v'},build)
"""
            result=subprocess.run([sys.executable,'-c',script,str(path)],cwd=Path(sdk.__file__).parent)
            self.assertEqual(result.returncode,19)
            self.assertFalse(sdk.cache_ready(path,{'id':'v'}))
            def retry(stage):
                with (stage/'matrix').open('xb') as f: f.write(b'whole')
                return ['matrix']
            sdk.prepare_cache(path,{'id':'v'},retry)
            self.assertFalse((Path(d)/'.weights.staging').exists())
            self.assertTrue(sdk.verify_cache(path,{'id':'v'}))

    def test_recover_partial_files_from_previous_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'artifact'
            path.mkdir()
            (path/'.prepare.lock').touch()
            (path/'weights').write_bytes(b'half')
            def build(stage):
                with (stage/'weights').open('xb') as f: f.write(b'complete')
                return ['weights']
            self.assertFalse(sdk.prepare_cache(path,{'model':'old'},build))
            self.assertEqual((path/'weights').read_bytes(),b'complete')
            self.assertTrue(sdk.verify_cache(path,{'model':'old'}))

    def test_first_new_result_excludes_restored_work_and_partial_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=dict(WM_RUN_DIR=tmp,WM_WORKER_INDEX='0',WM_WORKER_COUNT='1',WM_IDENTITY='fixture',
                     WM_ATTEMPT='second',WM_DEADLINE='300',WM_ALLOCATION_START='100',WM_PRIOR_COMPLETED='4',SLURM_GPUS_ON_NODE='3')
            with patch.dict(os.environ,env),patch.object(sdk.time,'time',return_value=135):
                adapter=sdk.Adapter()
                for done in [4,4,5,6]: adapter.metric(completed=done)
            import json
            rows=[json.loads(line) for line in (adapter.directory/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r.get('first_saved_result_seconds') for r in rows],[None,None,35,35])
            self.assertAlmostEqual(rows[-1]['new_items_per_gpu_hour_so_far'],2*3600/(35*3))

    def test_suspect_artifact_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model'
            def build(stage):
                (stage/'matrix').write_bytes(b'valid')
                return ['matrix']
            sdk.prepare_cache(p,{},build)
            file=p/'matrix'
            st=file.stat()
            file.write_bytes(b'wrong')
            os.utime(file,ns=(st.st_atime_ns,st.st_mtime_ns))
            self.assertTrue(sdk.cache_ready(p,{}))
            self.assertFalse(sdk.verify_cache(p,{}))
            self.assertFalse(sdk.prepare_cache(p,{},build,verify=True))
            self.assertEqual(file.read_bytes(),b'valid')
