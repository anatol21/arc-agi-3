#!/usr/bin/env bash
#SBATCH --job-name=arc-dev-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@300
#SBATCH --output=logs/arc-dev-v2-%j.log
#SBATCH --error=logs/arc-dev-v2-%j.log
set -eo pipefail
source "${ARC_INFERENCE_ENV:?Set ARC_INFERENCE_ENV to the provisioned inference environment}/bin/activate"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$PWD/.cache/triton"
unset LOCAL_RANK RANK WORLD_SIZE ACCELERATE_USE_FSDP ACCELERATE_USE_DEEPSPEED ACCELERATE_USE_TP ARC_RESUME_CHECKPOINT
export ARC_BENCH_STOP_AT="$(( $(date +%s) + 5 * 3600 + 30 * 60 ))"
srun --ntasks=1 python arc-flash-dev-benchmark-v2/run_benchmark.py
