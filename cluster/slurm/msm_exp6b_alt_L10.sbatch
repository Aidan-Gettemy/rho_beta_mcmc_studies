#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per chain (64 chains)
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=02:00:00
#SBATCH -J msm-exp6b
#SBATCH -o logs/msm-exp6b-%j.out
#
# Local-KLE (MSM) study, experiment 6b: alternate every 10 iterations, field-norm lift + frozen offset
# Plan: docs/msm_experiment_plan_2026-11-03.md.  Self-contained: every run flag is written below, so rereading
# this file shows how the run was made.  Submit from the repo root:
#     sbatch cluster/slurm/msm_exp6b_alt_L10.sbatch
# Settings: 64 chains, N = 100,000 iterations, step size adapted to 25% acceptance for the first M = 10,000
# iterations (each phase for sequences), then frozen; sigma_rel = 0.003; checkerboard data (512 of 1024 cell
# centres), truth = fixed-seed prior draw (seed 2026); chain seeds [7, chain] as in all experiments.  States saved
# every 50 iterations; MPSRF window from iteration 1,000; posterior fields from 10,000 samples.
# Output: runs/msm_local_kle_2026-11/exp6b_alt_L10/ (chains.npz, summary.json, post/).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ---- conda environment (explicit, checked) -------------------------------------------------
set +u
module load miniconda/24.11.1 2>/dev/null || true
ENV=/home/arg210006/.conda/envs/gd24_numba
source activate "$ENV" || { eval "$(conda shell.bash hook)" && conda activate "$ENV"; }
set -u
if [ "${CONDA_PREFIX:-}" != "$ENV" ]; then
    echo "ERROR: conda environment $ENV was not activated (python: $(command -v python))" >&2
    exit 1
fi
echo "python: $(command -v python)"; python -c "import numpy, scipy; print('numpy', numpy.__version__)"

# ---- one thread per process; compiled-kernel cache in node-local scratch ------------------------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
export BRHO_ROOT=$PWD/runs/msm_local_kle_2026-11
export PYTHONPATH=$PWD:${PYTHONPATH:-}
OUT=$BRHO_ROOT/exp6b_alt_L10

# ---- tests that must pass before sampling (no pytest needed) ------------------------------------
python tests/run_tests.py test_dirichlet test_msm_local test_msm_sampler

# ---- chains ----------------------------------------------------------------------------------------
python experiments/run_msm_chains.py \
    --kind alt \
    --chains 64 --workers 64 \
    --n-iter 100000 --adapt 10000 \
    --alt-L 10 \
    --s0 0.2 --save-every 50 \
    --sigma-rel 0.003 --data-seed 2026 --seed0 7 \
    --out "$OUT"

# ---- diagnostics: AR histograms, relative-error traces, MPSRF, field maps/pairs (10,000 samples) --
python experiments/run_msm_post.py --dir "$OUT" --n-fields 10000 --mpsrf-start 1000 --seed 11
