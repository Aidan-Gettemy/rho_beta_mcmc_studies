#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=02:00:00
#SBATCH -J msm-exp10-robin-dual
#SBATCH -o logs/msm-exp10-robin-dual-%j.out
#
# Domain-decomposed MCMC, Robin interface condition (alpha = 0.1) on the flux-measured faces + interface flux data (224 data, noise 0.3%, likelihood level 0.08): dual grid (25 subdomains), pilot + frozen-step restart
# Spec: docs/dd_overlap_robin_experiment.md.  Self-contained: every run flag is written below, so rereading this file shows how the run was made.
# Submit from the repo root:   sbatch cluster/slurm/msm_exp10_dd_robin_dual.sbatch
# TWO STAGES, to separate adaptation from convergence:
#   1. PILOT   10,000 iterations, step size adapted to 25% acceptance the whole time (64 chains) -> exp10_dd_robin_dual_pilot
#              the frozen step size beta is the geometric mean of the saved step sizes of all chains over the last 30%
#              of the pilot, one beta per subdomain (one for the global problem);
#   2. RESTART 50,000 iterations from NEW random starting points (fresh seeds, initial state = a draw from the prior,
#              far more dispersed than the posterior), step size frozen at beta (--adapt 0), first 1,000 iterations
#              discarded for the fields and the acceptance statistics (--burn 1000).
# Settings: 64 chains, states saved every 50 iterations, per-sensor likelihood weight (sigma_rel 0.00013258252, data noise 0.003), checkerboard data (512 of 1024 cells), truth =
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
OUT=$BRHO_ROOT/exp10_dd_robin_dual
PILOT=${OUT}_pilot

# ---- tests that must pass before sampling (no pytest needed) ------------------------------------
python tests/run_tests.py test_msm_dd

COMMON="--grid dual --chains 64 --workers 64 --save-every 50 --smoothing 1e-3 --neighbors 60 --sigma-rel 0.00013258252 --data-relerr 0.003 --data-seed 2026 --flux-noise-rel 0.003 --flux-sigma-rel 0.08 --robin-alpha 0.1"

# ---- stage 1: pilot, step size adapted throughout --------------------------------------------------
python experiments/run_dd_chains.py $COMMON \
    --n-iter 10000 --adapt 10000 --burn 5000 --s0 0.2 --seed0 7 \
    --out "$PILOT"

# ---- stage 2: restart from dispersed prior draws with the step size frozen ---------------------------
python experiments/run_dd_chains.py $COMMON \
    --n-iter 50000 --adapt 0 --burn 1000 --step-from "$PILOT" --init-scale 1.0 --seed0 1007 \
    --out "$OUT"

# ---- primal + dual pooled (straight average of the two grids' posterior means; sd from the union of samples, 5,000 fields per grid)
PRIMAL=$BRHO_ROOT/exp10_dd_robin_primal
[ -f "$PRIMAL/chains.npz" ] || { echo "ERROR: $PRIMAL/chains.npz missing - run the primal script first" >&2; exit 1; }
python experiments/run_dd_post.py --dirs "$PRIMAL" "$OUT" --out "$BRHO_ROOT/exp10_dd_robin_pooled/post" --n-fields 5000 --mpsrf-start 1000 --mpsrf-step-its 100 --seed 11 --ref "$BRHO_ROOT/exp9_global_sharp_flux/post/post_summary.json"
