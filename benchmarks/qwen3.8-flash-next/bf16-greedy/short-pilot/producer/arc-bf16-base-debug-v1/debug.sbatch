#!/usr/bin/env bash
#SBATCH --job-name=arc-bf16-4x8

#SBATCH --qos=normal
#SBATCH --partition=dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/arc-bf16-4x8-%j.log
set -euo pipefail
cd /path/to/arc-workspace
module load devel/miniforge/25.3.1-python-3.12
export LD_LIBRARY_PATH="/opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$PWD/.cache/triton"
unset LOCAL_RANK RANK WORLD_SIZE ACCELERATE_USE_FSDP ACCELERATE_USE_DEEPSPEED ACCELERATE_USE_TP ARC_RESUME_CHECKPOINT
pilot="$PWD/arc-bf16-base-debug-v1"
(cd "$pilot" && sha256sum -c package.sha256)
# All four GPUs on this node belong to this allocation. Sample by UUID so the
# report can be reconciled with the device names recorded inside the worker.
telemetry="$PWD/logs/arc-bf16-4x8-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$telemetry" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
status=0
srun --ntasks=1 --mem=0 --cpus-per-task="${SLURM_CPUS_ON_NODE:?}" --gres=gpu:4 .venv-arc/bin/python -I "$pilot/run.py" || status=$?
if [[ $status -eq 0 ]]; then
  .venv-arc/bin/python -I "$pilot/score.py"
fi
exit "$status"
