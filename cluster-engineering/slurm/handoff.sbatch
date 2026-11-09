#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
exec "${SCHEDULER_PYTHON:-python3}" "$3" settle "$1" "$2"
