"""Sample only allocated GPU IDs; expand Slurm ranges before invoking nvidia-smi."""
import os
import re
import subprocess
import threading
import time


def expand_ids(raw):
    if not raw:
        raise ValueError('Allocated GPU IDs absent')
    result = []
    for part in raw.split(','):
        if not re.fullmatch(r'\d+(?:-\d+)?', part):
            raise ValueError('Unsupported allocated GPU ID format: ' + raw)
        bounds = [int(x) for x in part.split('-')]
        first, last = bounds[0], bounds[-1]
        if last < first or last - first > 128:
            raise ValueError('Invalid allocated GPU range')
        result.extend(range(first, last + 1))
    if len(result) != len(set(result)):
        raise ValueError('Duplicate GPU IDs')
    return result


class Monitor:
    def __enter__(self):
        self.samples = []; self.errors = []; self.stop = threading.Event()
        self.raw_ids = os.environ.get('SLURM_STEP_GPUS', '')
        try:
            self.ids = expand_ids(self.raw_ids)
            if len(self.ids) != 3:
                raise ValueError('Expected three allocated telemetry GPU IDs')
        except ValueError as e:
            self.errors.append(str(e)); return self

        def sample():
            while not self.stop.is_set():
                try:
                    result = subprocess.run(['nvidia-smi', '-i', ','.join(map(str, self.ids)),
                        '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                    if result.returncode:
                        raise RuntimeError(result.stderr[:300])
                    seen = set()
                    for line in result.stdout.splitlines():
                        i, uid, busy, used, total, power = [x.strip() for x in line.split(',')]
                        index = int(i)
                        if index not in self.ids or index in seen:
                            raise ValueError('Telemetry returned unexpected/duplicate GPU')
                        seen.add(index)
                        self.samples.append(dict(time=time.time(), physical_index=index, uuid=uid,
                            gpu_busy_percent=float(busy), memory_MiB=float(used),
                            total_memory_MiB=float(total), power_W=float(power)))
                    if seen != set(self.ids):
                        raise ValueError('Telemetry omitted allocated GPU(s)')
                except Exception as e:
                    self.errors.append(str(e)[:300]); break
                self.stop.wait(2)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=4)

    def report(self):
        means = {}
        for index in getattr(self, 'ids', []):
            values = [r['gpu_busy_percent'] for r in self.samples if r['physical_index'] == index]
            means[str(index)] = sum(values) / len(values) if values else None
        return dict(raw_step_gpu_ids=self.raw_ids, expanded_gpu_ids=getattr(self, 'ids', []),
            samples=self.samples, errors=self.errors, mean_gpu_busy_percent=means,
            note='Sampled activity, not percentage of peak FLOPs; NVML physical IDs may differ from CUDA ordinals')
