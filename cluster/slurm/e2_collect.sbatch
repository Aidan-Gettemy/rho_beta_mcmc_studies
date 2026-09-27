#!/bin/bash
#SBATCH --job-name=brho-e2-collect
#SBATCH --partition=g1mig             # your partition
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source cluster/env.sh
python experiments/run_grid_study.py --collect
python reports/report_setup.py
python reports/report_grid.py
