#!/bin/bash
#SBATCH --job-name=arc-heldout-py-watch

#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --output=logs/arc-heldout-py-watch-%j.log
set -eo pipefail
module load devel/miniforge/25.3.1-python-3.12
python arc-python-heldout-length-v1/launch.py "$1"
