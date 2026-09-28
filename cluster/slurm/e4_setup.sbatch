#!/bin/bash
#SBATCH --job-name=brho-e4-setup
#SBATCH --partition=g1mig             # your partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Loads the study's KLE basis + reference field from the store (the same
# ones E3 used: checked), freezes them and the E3 curve into the run.
# Set PRIOR_TEST=1 to set up the likelihood-off check instead.
set -euo pipefail
source cluster/env.sh
# the KLE build (eigensolve + Nystrom extension) runs on BLAS threads here
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
python experiments/run_curve_sampler.py --setup-only --n-chains "${E4_CHAINS:-16}" \
    --prior-test "${PRIOR_TEST:-0}" $PROBLEM_ARGS $E4_ARGS
