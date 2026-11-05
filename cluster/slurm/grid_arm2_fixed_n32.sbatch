#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per chain (64 chains)
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=00:45:00
#SBATCH -J grid-arm2_fixed_n32
#SBATCH -o logs/grid-arm2_fixed_n32-%j.out
#
# Grid-refinement study of global pCN, arm2_fixed on the 32 x 32 grid.  Design and rationale (including why
# sigma_abs = 0.0167): docs/grid_arms_experiment.md.  Self-contained: every run flag is written below.  Submit from
# the repo root:   sbatch cluster/slurm/grid_arm2_fixed_n32.sbatch      (all nine: bash cluster/submit_grid_arms.sh)
# Problem: Dirichlet 1 / 0 left / right, no flow top / bottom, sqexp kernel, correlation length 0.2, 20-mode
# Nystrom KLE, truth theta* = first 20 draws of seed 2026.  This job: 128 observations on every grid (grid cell closest to the centre of each cell of the 16 x 16 checkerboard; lower-left of the central four on 32 / 64), same noise and likelihood as arm 1.
# Settings: 64 chains, N = 100,000 iterations, step size adapted to 25% acceptance for the first M = 10,000
# iterations, then frozen; chain seeds [7, chain]; states saved every 50 iterations; MPSRF window from iteration
# 1,000; posterior fields from 10,000 samples.  Output: runs/grid_arms_2026-11/arm2_fixed_n32/ (chains.npz, summary.json, post/).
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
export BRHO_ROOT=$PWD/runs/grid_arms_2026-11
export PYTHONPATH=$PWD:${PYTHONPATH:-}
OUT=$BRHO_ROOT/arm2_fixed_n32

# ---- tests that must pass before sampling (no pytest needed) ------------------------------------
python tests/run_tests.py test_dirichlet test_gridstudy

# ---- chains ----------------------------------------------------------------------------------------
python experiments/run_grid_chains.py \
    --arm fixed --n 32 \
    --chains 64 --workers 64 \
    --n-iter 100000 --adapt 10000 \
    --s0 0.2 --save-every 50 \
    --eps 0.003 --data-seed 2026 --seed0 7 \
    --out "$OUT"

# ---- diagnostics: AR histogram, relative-error traces, MPSRF of theta, theta posterior, field maps ----
python experiments/run_grid_post.py --dir "$OUT" --n-fields 10000 --mpsrf-start 1000 --seed 11
