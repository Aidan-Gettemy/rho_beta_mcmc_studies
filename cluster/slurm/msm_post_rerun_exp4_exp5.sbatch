#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=02:00:00
#SBATCH -J msm-post-old
#SBATCH -o logs/msm-post-old-%j.out
#
# Post-processing only (no new chains) of the first-pass experiments 4 and 5 (saved every 100 iterations):
# new MPSRF convention (window from iteration 1,000, common 58 coordinates), contourf/jet fields from
# 10,000 samples.  Overwrites their post/ folders.  Plan: docs/msm_experiment_plan_2026-11-03.md.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
set +u
module load miniconda/24.11.1 2>/dev/null || true
ENV=/home/arg210006/.conda/envs/gd24_numba
source activate "$ENV" || { eval "$(conda shell.bash hook)" && conda activate "$ENV"; }
set -u
if [ "${CONDA_PREFIX:-}" != "$ENV" ]; then
    echo "ERROR: conda environment $ENV was not activated (python: $(command -v python))" >&2
    exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
export BRHO_ROOT=$PWD/runs/msm_local_kle_2026-11
export PYTHONPATH=$PWD:${PYTHONPATH:-}
for E in exp4_mixture exp5_sequence; do
    python experiments/run_msm_post.py --dir "$BRHO_ROOT/$E" --n-fields 10000 --mpsrf-start 1000 --seed 11
done
