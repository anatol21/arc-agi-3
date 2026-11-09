"""CUDA UUID to NVML mapping, verified on all3 devices; never trust numeric Slurm IDs."""
import os
import re
import subprocess
import threading
import time


def normalize_uuid(value):
    value = str(value)
    value = value if value.startswith('GPU-') else 'GPU-' + value
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',value):
        raise ValueError('Unsupported CUDA GPU UUID')
    return value


def cuda_gpu_uuids(torch):
    if torch.cuda.device_count() != 3:
        raise RuntimeError('Expected3 CUDA-visible GPUs')
    ids = [normalize_uuid(torch.cuda.get_device_properties(i).uuid) for i in range(3)]
    if len(set(ids)) != 3:
        raise RuntimeError('Duplicate CUDA device UUID')
    return ids


def parse_frame(stdout, uuids, now):
    result = []; seen = set()
    for line in stdout.splitlines():
        index, uid, busy, used, total, power = [s.strip() for s in line.split(',')]
        if uid not in uuids or uid in seen:
            raise ValueError('Unexpected/duplicate telemetry UUID')
        seen.add(uid)
        result.append(dict(time=now,cuda_ordinal=uuids.index(uid),physical_index=int(index),uuid=uid,
            gpu_busy_percent=float(busy),memory_MiB=float(used),total_memory_MiB=float(total),power_W=float(power)))
    if seen != set(uuids):
        raise ValueError('Telemetry omitted CUDA-visible GPU(s)')
    return result


class Monitor:
    def __init__(self, uuids=None):
        self.uuids = list(uuids or [])

    def __enter__(self):
        self.samples=[]; self.errors=[]; self.stop=threading.Event()
        if len(self.uuids) != 3 or len(set(self.uuids)) != 3:
            self.errors.append('Three distinct CUDA UUIDs required; no telemetry queried')
            return self
        def sample():
            while not self.stop.is_set():
                try:
                    result=subprocess.run(['nvidia-smi','-i',','.join(self.uuids),
                        '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                    if result.returncode:
                        raise RuntimeError(result.stderr[:300])
                    # Append a complete three-device frame only; never average a truncated frame.
                    self.samples.extend(parse_frame(result.stdout,self.uuids,time.time()))
                except Exception as error:
                    self.errors.append(str(error)[:300]);break
                self.stop.wait(2)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        if hasattr(self,'thread'):
            self.thread.join(timeout=4)

    def report(self):
        means={}
        for i in range(3):
            values=[r['gpu_busy_percent']for r in self.samples if r['cuda_ordinal']==i]
            means[str(i)]=sum(values)/len(values) if values else None
        return dict(cuda_uuid_mapping={str(i):u for i,u in enumerate(self.uuids)},
            raw_slurm_step_gpus=os.environ.get('SLURM_STEP_GPUS'),samples=self.samples,errors=self.errors,
            valid_sample_frames=len(self.samples)//3,complete_coverage=bool(self.samples) and not self.errors,
            mean_gpu_busy_percent=means,
            note='Samples during this attempted batch; not percent of peak FLOPs or whole-allocation efficiency')
