#!/usr/bin/env bash
#SBATCH --job-name=arc-eval138-pilot
#SBATCH --qos=normal
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=logs/arc-eval138-pilot-%j.log
#SBATCH --error=logs/arc-eval138-pilot-%j.log
set -eo pipefail
if [[ -n "${ARC_PYTHON_MODULE:-}" ]]; then module load "$ARC_PYTHON_MODULE"; fi
source .venv-arc/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$PWD/.cache/triton"
unset LOCAL_RANK RANK WORLD_SIZE ACCELERATE_USE_FSDP ACCELERATE_USE_DEEPSPEED ACCELERATE_USE_TP ARC_RESUME_CHECKPOINT
srun --ntasks=1 python arc-flash-eval138-pilot-v1/launch.py
