"""CPU-only verification of the existing staged evaluation and debug prefix."""
import os
from pathlib import Path
from debug_common import NAME, dependencies


def stage(folder):
    native, recipe, m, receipt = dependencies(folder, require_ready=False)
    ready = folder / 'READY.json'
    if ready.exists():
        dependencies(folder)
    else:
        m['common'].atomic_json(ready, receipt)
    print('EVAL DEBUG STAGED: first 3/138 tasks, both models; native serial direct answers; '
          'thinking off, greedy, checkpoint-660; separate resumable output', flush=True)
    print('Tasks: ' + ', '.join(receipt['target_task_ids']), flush=True)
    print('No GPU job submitted. Full evaluation job and output remain unchanged.', flush=True)


if __name__ == '__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
    stage(Path(__file__).resolve().parent)
