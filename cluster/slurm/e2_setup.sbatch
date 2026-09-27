#!/bin/bash
#SBATCH --job-name=brho-e2-setup
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
# Writes truth + manifest (the settings every pair task will read).
set -euo pipefail
source cluster/env.sh
python experiments/run_grid_study.py --setup-only $PROBLEM_ARGS $E2_ARGS
