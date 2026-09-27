#!/bin/bash
#SBATCH --job-name=brho-reports
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# All figures and tables for whatever has finished, plus SUMMARY.md.
set -euo pipefail
source cluster/env.sh
python experiments/run_curve_sampler.py --status || true
python reports/report_all.py
