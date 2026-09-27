#!/bin/bash
#SBATCH --job-name=brho-e3-curve
#SBATCH --partition=g1mig             # your partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Offline beta_bar(rho) sweep (beta = 1 solves only): n_rho x n_real solves,
# twice (truncated KLE, then full-rank Cholesky for the consistency figure).
set -euo pipefail
source cluster/env.sh
# the KLE build (eigensolve + Nystrom extension) runs on BLAS threads here
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
python experiments/run_beta_curve.py $PROBLEM_ARGS $E3_ARGS
python reports/report_curve.py
