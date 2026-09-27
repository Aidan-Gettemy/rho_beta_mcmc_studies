#!/bin/bash
#SBATCH --job-name=brho-e2-setup
#SBATCH --partition=g1mig             # your partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Loads the study's KLE basis and reference field from $BRHO_ROOT/store (or
# builds and stores them), freezes a copy into the run, writes the manifest.
set -euo pipefail
source cluster/env.sh
# the KLE build (eigensolve + Nystrom extension) runs on BLAS threads here
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
python experiments/run_grid_study.py --setup-only $PROBLEM_ARGS $E2_ARGS
