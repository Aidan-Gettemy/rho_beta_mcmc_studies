#!/bin/bash
#SBATCH --job-name=brho-e2-pair
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-24
#SBATCH --output=logs/%x-%A_%a.out
# One fixed-(beta, rho) pCN chain per array task.  Finished pairs are
# skipped, so resubmitting the array only reruns what is missing.
set -euo pipefail
source cluster/env.sh
python experiments/run_grid_study.py --pair-index env
