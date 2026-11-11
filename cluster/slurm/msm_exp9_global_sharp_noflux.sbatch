#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=24:00:00
#SBATCH -J msm-exp9-noflux
#SBATCH -o logs/msm-exp9-noflux-%j.out
#
# Global pCN reference, SHARP (per-sensor) likelihood, WITHOUT interface flux data: pilot + frozen-step restart, 500,000 iterations
# Spec: docs/dd_overlap_robin_experiment.md.  Self-contained: every run flag is written below, so rereading this file shows how the run was made.
# Submit from the repo root:   sbatch cluster/slurm/msm_exp9_global_sharp_noflux.sbatch
# TWO STAGES, to separate adaptation from convergence:
#   1. PILOT   10,000 iterations, step size adapted to 25% acceptance the whole time (64 chains) -> exp9_global_sharp_noflux_pilot
#              the frozen step size beta is the geometric mean of the saved step sizes of all chains over the last 30%
#              of the pilot, one beta per subdomain (one for the global problem);
#   2. RESTART 500,000 iterations from NEW random starting points (fresh seeds, initial state = a draw from the prior,
#              far more dispersed than the posterior), step size frozen at beta (--adapt 0), first 1,000 iterations
#              discarded for the fields and the acceptance statistics (--burn 1000).
# Settings: 64 chains, states saved every 50 iterations, per-sensor likelihood weight 1/sigma^2 (sigma_rel 0.00013258252, data noise 0.003), checkerboard data (512 of 1024 cells), truth =
# seed-2026 prior draw. MPSRF: window from iteration 1,000, curve point every 100 iterations, group convention.

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
echo "python: $(command -v python)"; python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

# ---- one thread per process; compiled-kernel cache in node-local scratch ------------------------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
export BRHO_ROOT=$PWD/runs/msm_local_kle_2026-11
export PYTHONPATH=$PWD:${PYTHONPATH:-}
OUT=$BRHO_ROOT/exp9_global_sharp_noflux
PILOT=${OUT}_pilot

# ---- tests that must pass before sampling (no pytest needed) ------------------------------------
python tests/run_tests.py test_dirichlet test_msm_local test_msm_sampler test_msm_dd

COMMON="--kind global --chains 64 --workers 64 --save-every 50 --sigma-rel 0.00013258252 --data-relerr 0.003 --data-seed 2026"

# ---- stage 1: pilot, step size adapted throughout --------------------------------------------------
python experiments/run_msm_chains.py $COMMON \
    --n-iter 10000 --adapt 10000 --burn 5000 --s0 0.2 --seed0 7 \
    --out "$PILOT"

# ---- stage 2: restart from dispersed prior draws with the step size frozen ---------------------------
python experiments/run_msm_chains.py $COMMON \
    --n-iter 500000 --adapt 0 --burn 1000 --step-from "$PILOT" --init-scale 1.0 --seed0 1007 \
    --out "$OUT"

# ---- diagnostics: AR histograms, relative-error traces, MPSRF, field maps/pairs (10,000 samples) --
python experiments/run_msm_post.py --dir "$OUT" --n-fields 10000 --mpsrf-start 1000 --mpsrf-step-its 100 --seed 11
