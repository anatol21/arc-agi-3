"""Read only allocated GPU telemetry; never probe other allocations."""
import os
import subprocess
import threading
import time

class Monitor:
    def __enter__(self):
        self.samples=[];self.errors=[];self.stop=threading.Event()
        self.ids=os.environ.get('SLURM_STEP_GPUS')
        if not self.ids:self.errors.append('SLURM_STEP_GPUS absent; no GPU telemetry queried');return self
        if any(not c.isdigit() and c not in ',-' for c in self.ids):
            self.errors.append('Unsupported allocated GPU ID format; no GPU telemetry queried');return self
        def run():
            while not self.stop.is_set():
                try:
                    result=subprocess.run(['nvidia-smi','-i',self.ids,'--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                    if result.returncode:raise RuntimeError(result.stderr[:300])
                    for line in result.stdout.splitlines():
                        i,u,mem,power=line.split(',');self.samples.append({'time':time.time(),'index':i.strip(),'gpu_busy_percent':float(u),'memory_MiB':float(mem),'power_W':float(power)})
                except Exception as e:self.errors.append(str(e)[:300]);break
                self.stop.wait(2)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set()
        if hasattr(self,'thread'):self.thread.join(timeout=4)
    def report(self):return {'samples':self.samples,'errors':self.errors,'metric_note':'GPU busy time fraction, not percentage of peak FLOPs'}
