#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=08:00:00
#SBATCH -J msm-exp17-ov-h4-dual-e99
#SBATCH -o logs/msm-exp17-ov-h4-dual-e99-%j.out
#
# OVERLAPPING subdomains + Robin (alpha = 0.1) + plain Gibbs for the flux datum h: dual decomposition, halo 4, local KLE energy 0.99.
# Every subdomain is extended by 4 cells towards each interior interface.  The extended block has its own local KLE (0.99 energy),
# uses ALL sensors of its rectangle (--overlap-halo-sensors), the Robin condition on its extended boundary and the measured flux faces
# inside it (--interior-flux).  Flux data: the standard 14 primal+dual lines (224 faces), noise-level weight 0.00020045; the extended boundaries lie on them.
# The non-overlapping field is extracted from the cores and projected on the global KLE as before (theta = B c_all).
# Gibbs prior: sd of h = 0.03 x rms face flux (0.003 on the measured faces), correlation length 4 cells.
# Spec: docs/dd_overlap_robin_experiment.md.  Self-contained: every run flag is written below.
# Submit from the repo root:   sbatch cluster/slurm/msm_exp17_ov_h4_dual_e99.sbatch
# TWO STAGES: 1. PILOT   10,000 iterations, step size adapted throughout (64 chains) -> exp17_ov_h4_dual_e99_pilot; the frozen step beta is the
#                geometric mean of the saved steps of all chains over the last 30% (one per subdomain);
#             2. RESTART 100,000 iterations from NEW dispersed prior draws, step frozen (--adapt 0), first 1,000 discarded (--burn 1000).
# Settings: 64 chains, saved every 50 iterations, per-sensor likelihood (sigma_rel 0.00013258252, noise 0.003), checkerboard data, truth = seed-2026 draw.
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
OUT=$BRHO_ROOT/exp17_ov_h4_dual_e99
PILOT=${OUT}_pilot

# ---- tests that must pass before sampling (no pytest needed) ------------------------------------
python tests/run_tests.py test_msm_dd

COMMON="--grid dual --chains 64 --workers 64 --save-every 50 --smoothing 1e-3 --neighbors 60 --sigma-rel 0.00013258252 --data-relerr 0.003 --data-seed 2026 --flux-noise-rel 0.003 --flux-sigma-rel 0.00020045 --robin-alpha 0.1 --h-prior-rel 0.03 --h-prior-meas-rel 0.003 --h-corr-len 4 --interior-flux --overlap-halo 4 --overlap-energy 0.99 --overlap-halo-sensors"

# ---- stage 1: pilot, step size adapted throughout --------------------------------------------------
python experiments/run_dd_chains.py $COMMON \
    --n-iter 10000 --adapt 10000 --burn 5000 --s0 0.2 --seed0 7 \
    --out "$PILOT"

# ---- stage 2: restart from dispersed prior draws with the step size frozen ---------------------------
python experiments/run_dd_chains.py $COMMON \
    --n-iter 100000 --adapt 0 --burn 1000 --step-from "$PILOT" --init-scale 1.0 --seed0 1007 \
    --out "$OUT"

# ---- post-processing: fields, errors, MPSRF; reference = the converged global run (flux at the noise level)
python experiments/run_dd_post.py --dirs "$OUT" --out "$OUT/post" --n-fields 5000 --mpsrf-start 1000 --mpsrf-step-its 100 --seed 11 --ref "$BRHO_ROOT/exp14_global_sharp_fluxnoise_10M/post/post_summary.json"
