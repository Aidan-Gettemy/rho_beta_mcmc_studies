#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=03:00:00
#SBATCH -J msm-exp18-noise-f1-dual
#SBATCH -o logs/msm-exp18-noise-f1-dual-%j.out
#
# EXP. 18 -- noise scaling of the NON-overlapping DD: dual decomposition, ALL noise levels divided by f = 1.0.
#   data noise 0.003 (nominal 0.003), likelihood sigma_rel 0.000132583 (nominal 0.00013258252), flux noise 0.003,
#   flux weight per datum 0.00020045 (nominal 0.00020045 = noise/sqrt(224)), Gibbs prior sd of h on the measured faces 0.003 (nominal 0.003).
# Everything else as the matched exp. 17 settings WITHOUT overlap: Robin alpha 0.1, plain Gibbs for h (sd 0.03 x rms face flux,
# correlation length 4 cells), interior-flux likelihood, the standard 14 flux lines (224 faces), local basis 99% local KLE energy, truth = seed-2026 draw.
# Reference: the converged global run at the NOMINAL noise (exp. 14, error 0.0045).  f = 1 is the matched non-overlapping control.
# Sharper data -> smaller pCN steps -> longer chains: 64 chains, pilot 10000 (adapted, floor 0.001, window 100), restart 100000 iterations
# with the step frozen, saved every 50.  If MPSRF is still above 1.2 at the end, extend (more iterations) before reading the errors.
# Submit from the repo root:   sbatch cluster/slurm/msm_exp18_noise_f1_dual.sbatch      (or: bash cluster/submit_noise_scaling.sh)
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
OUT=$BRHO_ROOT/exp18_noise_f1_dual
PILOT=${OUT}_pilot

# ---- tests that must pass before sampling (no pytest needed) ------------------------------------
python tests/run_tests.py test_msm_dd

COMMON="--grid dual --chains 64 --workers 64 --smoothing 1e-3 --neighbors 60 --sigma-rel 0.00013258252 --data-relerr 0.003 --data-seed 2026 --flux-noise-rel 0.003 --flux-sigma-rel 0.00020045 --robin-alpha 0.1 --h-prior-rel 0.03 --h-prior-meas-rel 0.003 --h-corr-len 4 --interior-flux --dual-energy 0.99 --s-min 0.001 --adapt-every 100"

# ---- stage 1: pilot, step size adapted throughout --------------------------------------------------
python experiments/run_dd_chains.py $COMMON \
    --save-every 50 --n-iter 10000 --adapt 10000 --burn 5000 --s0 0.2 --seed0 7 \
    --out "$PILOT"

# ---- stage 2: restart from dispersed prior draws with the step size frozen ---------------------------
python experiments/run_dd_chains.py $COMMON \
    --save-every 50 --n-iter 100000 --adapt 0 --burn 1000 --step-from "$PILOT" --init-scale 1.0 --seed0 1007 \
    --out "$OUT"

# ---- post-processing: fields, errors, MPSRF; reference = the converged global run at the nominal noise (exp. 14)
python experiments/run_dd_post.py --dirs "$OUT" --out "$OUT/post" --n-fields 5000 --mpsrf-start 1000 --mpsrf-step-its 100 --seed 11 --ref "$BRHO_ROOT/exp14_global_sharp_fluxnoise_10M/post/post_summary.json"
