#!/usr/bin/env bash
set -euo pipefail
cd "${ARC_WORKSPACE:?Set ARC_WORKSPACE to the shared workspace}"
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
if [[ -n "${ARC_PYTHON_MODULE:-}" ]]; then module load "$ARC_PYTHON_MODULE"; fi
# Set the runtime library path for your local build before launch.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$PWD/.cache/triton"
unset LOCAL_RANK RANK WORLD_SIZE ACCELERATE_USE_FSDP ACCELERATE_USE_DEEPSPEED ACCELERATE_USE_TP ARC_RESUME_CHECKPOINT ARC_BENCH_STOP_AT
exec "$PWD/.venv-arc/bin/python" -I "$PWD/arc-flash-eval138-dashboard-v4/dashboard_adapter.py"
