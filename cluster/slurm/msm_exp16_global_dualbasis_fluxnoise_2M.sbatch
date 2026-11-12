#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=04:00:00
#SBATCH -J msm-exp16-global-dualbasis-fluxnoise-2M
#SBATCH -o logs/msm-exp16-global-dualbasis-fluxnoise-2M-%j.out
#
# GLOBAL likelihood (full 32x32 forward solve, 512 sharp pressure data + 224 flux data at their noise level, as exp14) but sampled in the DUAL local-KLE basis (95% energy, 114 coefficients): the unknown is the stacked local-KLE coefficient vector c (theta = B c, prior N(0, I)), not the 64 global KLE coefficients. Question answered: how much of the DD-vs-global accuracy gap comes from the local parametrisation alone (no decomposition, no interface approximation)? Pilot + frozen-step restart, 2,000,000 iterations
# Spec: docs/dd_gibbs_experiment.md (section: same-basis global).  Self-contained: every run flag is written below, so rereading this file shows how the run was made.
# Submit from the repo root:   sbatch cluster/slurm/msm_exp16_global_dualbasis_fluxnoise_2M.sbatch
# TWO STAGES, to separate adaptation from convergence:
#   1. PILOT   100,000 iterations (step-size floor lowered to 1e-4; adaptation window 500 iterations instead of 100, so each up/down decision rests on ~125 proposals instead of ~25), step size adapted to 25% acceptance the whole time (64 chains) -> exp16_global_dualbasis_fluxnoise_2M_pilot
#              the frozen step size beta is the geometric mean of the saved step sizes of all chains over the last 30%
#              of the pilot, one beta per subdomain (one for the global problem);
#   2. RESTART 2,000,000 iterations from NEW random starting points (fresh seeds, initial state = a draw from the prior,
#              far more dispersed than the posterior), step size frozen at beta (--adapt 0), first 1,000 iterations
#              discarded for the fields and the acceptance statistics (--burn 1000).
# Settings: 64 chains, pilot: states saved every 50 iterations; restart: every 200 (10,000 saved states per chain), per-sensor likelihood weight 1/sigma^2 (sigma_rel 0.00013258252, data noise 0.003), checkerboard data (512 of 1024 cells), truth =
# seed-2026 prior draw. MPSRF: window from iteration 1,000, curve point every 200 iterations, group convention.

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
OUT=$BRHO_ROOT/exp16_global_dualbasis_fluxnoise_2M
PILOT=${OUT}_pilot

# ---- tests that must pass before sampling (no pytest needed) ------------------------------------
python tests/run_tests.py test_dirichlet test_msm_local test_msm_sampler test_msm_dd

COMMON="--kind dual_raw --chains 64 --workers 64 --sigma-rel 0.00013258252 --data-relerr 0.003 --data-seed 2026 --flux-noise-rel 0.003 --flux-sigma-rel 0.00020045 --primal-dim 6 --dual-energy 0.95"

# ---- stage 1: pilot, step size adapted throughout --------------------------------------------------
python experiments/run_msm_chains.py $COMMON \
    --save-every 50 --n-iter 100000 --adapt 100000 --burn 50000 --s0 0.2 --s-min 1e-4 --adapt-every 500 --seed0 7 \
    --out "$PILOT"

# ---- stage 2: restart from dispersed prior draws with the step size frozen ---------------------------
python experiments/run_msm_chains.py $COMMON \
    --save-every 200 --n-iter 2000000 --adapt 0 --burn 1000 --step-from "$PILOT" --init-scale 1.0 --seed0 1007 \
    --out "$OUT"

# ---- diagnostics: AR histograms, relative-error traces, MPSRF, field maps/pairs (10,000 samples) --
python experiments/run_msm_post.py --dir "$OUT" --n-fields 10000 --mpsrf-start 1000 --mpsrf-step-its 200 --seed 11
