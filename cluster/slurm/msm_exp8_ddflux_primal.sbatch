#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=00:40:00
#SBATCH -J msm-ddflux-primal
#SBATCH -o logs/msm-ddflux-primal-%j.out
#
# Domain-decomposed MCMC with spline interface (Xu, Liao & Li 2022) PLUS interface FLUX data (224 data, every other face of the 14 interface lines, noise 0.3%, likelihood level 0.08), primal 4x4 grid (16 subdomains, 6 local modes each). Test A of the plan is the post-processing at the end of this script.
# Spec: docs/dd_flux_experiment.md.  Self-contained: every run flag is written below, so rereading this file shows
# how the run was made.  Submit from the repo root:
#     sbatch cluster/slurm/msm_exp8_ddflux_primal.sbatch
# Settings: 64 independent chains per subdomain, N = 50,000 iterations, step size adapted to 25% acceptance for the
# first M = 5,000 iterations, then frozen; local KLE of 95% energy (primal); sigma_rel = 0.003; checkerboard data
# (512 of 1024 cell centres), truth = fixed-seed prior draw (seed 2026); chain seeds [7, grid, block, chain].
# Interface pressure: thin-plate spline through all sensors (smoothing 1e-3, 60 neighbours), Dirichlet on the faces.
# States saved every 50 iterations; MPSRF per subdomain from iteration 1,000, sup over subdomains.
# Output: runs/msm_local_kle_2026-11/exp8_ddflux_primal/ (chains.npz, summary.json), post/ (test A)
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
OUT=$BRHO_ROOT/exp8_ddflux_primal

# ---- tests that must pass before sampling (no pytest needed) ------------------------------------
python tests/run_tests.py test_msm_dd

# ---- chains (64 chains x all subdomains of the primal grid, 50,000 iterations) ---------------------
python experiments/run_dd_chains.py \
    --grid primal \
    --chains 64 --workers 64 \
    --n-iter 50000 --adapt 5000 \
    --s0 0.2 --save-every 50 \
    --smoothing 1e-3 --neighbors 60 \
    --flux-noise-rel 0.003 --flux-sigma-rel 0.08 \
    --sigma-rel 0.003 --data-seed 2026 --seed0 7 \
    --out "$OUT"

# ---- test A: primal only. Posterior fields from 5,000 assembled global fields (one independent saved state per subdomain)
python experiments/run_dd_post.py --dirs "$OUT" --out "$OUT/post" --n-fields 5000 --mpsrf-start 1000 --seed 11 \
    --ref "$BRHO_ROOT/exp8_global_flux/post/post_summary.json"   # sd ratio to the global+flux run (run msm_exp8_global_flux.sbatch first; optional)
