#!/bin/bash
#SBATCH --job-name=brho-e3-curve
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Offline beta_bar(rho) sweep (beta = 1 solves only): n_rho x n_real solves,
# twice (truncated KLE, then full-rank Cholesky for the consistency figure).
set -euo pipefail
source cluster/env.sh
python experiments/run_beta_curve.py $PROBLEM_ARGS $E3_ARGS
python reports/report_curve.py
