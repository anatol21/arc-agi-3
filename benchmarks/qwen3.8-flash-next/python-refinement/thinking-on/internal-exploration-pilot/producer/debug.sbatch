#!/bin/bash
#SBATCH --job-name=arc-explore-pilot

#SBATCH --partition=dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:26:00
#SBATCH --no-requeue
#SBATCH --output=logs/arc-explore-%j.log
set -eo pipefail
module load devel/miniforge/25.3.1-python-3.12
source .venv-arc/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
unset LOCAL_RANK RANK WORLD_SIZE ACCELERATE_USE_FSDP ACCELERATE_USE_DEEPSPEED ARC_RESUME_CHECKPOINT
python -I arc-python-exploration-pilot-v4/memory_preflight.py
exec python -I arc-python-exploration-pilot-v4/worker.py "$1"
