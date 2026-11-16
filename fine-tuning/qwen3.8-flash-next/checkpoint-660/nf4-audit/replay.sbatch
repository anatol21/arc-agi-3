#!/usr/bin/env bash
#SBATCH --job-name=arc-nf4-audit

#SBATCH --qos=normal
#SBATCH --partition=dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --time=00:13:00
#SBATCH --no-requeue
#SBATCH --output=logs/arc-nf4-audit-%j.log
set -euo pipefail
module load devel/miniforge/25.3.1-python-3.12
source .venv-arc/bin/activate
export LD_LIBRARY_PATH="/opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$PWD/.cache/triton"
unset LOCAL_RANK RANK WORLD_SIZE ACCELERATE_USE_FSDP ACCELERATE_USE_DEEPSPEED ACCELERATE_USE_TP ARC_RESUME_CHECKPOINT
srun --ntasks=1 --mem=0 --cpus-per-task="${SLURM_CPUS_ON_NODE:?}" --gres=gpu:3 .venv-arc/bin/python -I runs/arc-nf4-audit-7853809-v1/replay.py
