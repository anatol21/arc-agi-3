#!/usr/bin/env bash
#SBATCH --job-name=arc-flash-v6

#SBATCH --qos=normal
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@300

#SBATCH --output=logs/arc-flash-v6-%j.log
#SBATCH --error=logs/arc-flash-v6-%j.log
set -eo pipefail
module load devel/miniforge/25.3.1-python-3.12
source .venv-arc/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$PWD/.cache/triton"
unset LOCAL_RANK RANK WORLD_SIZE ACCELERATE_USE_FSDP ACCELERATE_USE_DEEPSPEED ACCELERATE_USE_TP ARC_RESUME_CHECKPOINT
export ARC_STOP_AT="$(( $(date +%s) + 5 * 3600 + 30 * 60 ))"
srun --ntasks=1 python arc-flash-run-v6/guarded_train.py arc-flash-run-v6/arc-flash.yml
