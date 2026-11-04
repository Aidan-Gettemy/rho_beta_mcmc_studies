#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per chain (64 chains)
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=02:00:00
#SBATCH -J msm-exp3a_primal_pinv
#SBATCH -o logs/msm-exp3a_primal_pinv-%j.out
#
# Local-KLE (MSM) study, experiment 3a: primal MSM whitened with the pseudo-inverse (null directions dropped).
# Spec: docs/msm_local_kle_experiment.md.  Self-contained: every run flag is written below, so
# rereading this file shows how the run was made.  Submit from the repo root:
#     sbatch cluster/slurm/msm_exp3a_primal_pinv.sbatch
# Settings: 64 chains, N = 100,000 iterations, step size adapted to 25% acceptance for the first
# M = 10,000 iterations (each phase for experiment 5), then frozen; sigma_rel = 0.003; checkerboard
# data (512 of 1024 cell centres), truth = fixed-seed prior draw (seed 2026); chain seeds [7, chain]
# are the same in all six experiments.  Output: runs/msm_local_kle_2026-11/exp3a_primal_pinv/
#   chains.npz, summary.json (raw, stay on the cluster) and post/ (figures + post_summary.json).
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
OUT=$BRHO_ROOT/exp3a_primal_pinv

# ---- tests that must pass before sampling (cheap, no PDE solves except the solver tests) -------
python tests/run_tests.py test_dirichlet test_msm_local

# ---- chains --------------------------------------------------------------------------------------
python experiments/run_msm_chains.py \
    --kind primal_pinv \
    --chains 64 --workers 64 \
    --n-iter 100000 --adapt 10000 \
    --s0 0.2 --save-every 50 \
    --sigma-rel 0.003 --data-seed 2026 --seed0 7 \
    --out "$OUT"

# ---- diagnostics: AR histograms, relative-error traces, MPSRF, posterior fields -------------------
python experiments/run_msm_post.py --dir "$OUT" --n-fields 5000 --seed 11
