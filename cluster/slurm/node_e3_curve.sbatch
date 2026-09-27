#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=g1mig             # your partition
#SBATCH --time=02:00:00
#SBATCH -J brho-e3
#SBATCH -o logs/brho-e3-%j.out
#
# E3, the offline beta_bar(rho) curve: one process (n_rho x n_real solves at
# beta = 1; a few seconds at 16x16, minutes at 100x100).  Uses the study's
# KLE basis from the store (the same one E2 and E4 use).
# Submit from the repo root:   sbatch cluster/slurm/node_e3_curve.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
python experiments/run_beta_curve.py $PROBLEM_ARGS $E3_ARGS
python reports/report_curve.py
