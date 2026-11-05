#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=00:40:00
#SBATCH -J paper_e7_smooth_sub
#SBATCH -o logs/paper_e7_smooth_sub-%j.out
#
# Paper-MSM study (docs/paper_msm_experiment.md).  MSM with the paper's smoothing, one subdomain at a time (5 coefficients, stored neighbours)
# Submit from the repo root:   sbatch cluster/slurm/paper_e7_smooth_sub.sbatch      (all: bash cluster/submit_paper_msm.sh)
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
echo "python: $(command -v python)"; python -c "import numpy, scipy; print('numpy', numpy.__version__)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
export PYTHONPATH=$PWD:${PYTHONPATH:-}
python tests/run_tests.py test_dirichlet test_gridstudy test_msmpaper
OUT=runs/paper_msm_2026-11/e7_smooth_sub
# 150,000 iterations, step size adapted to 25% acceptance for the first M = 30,000, then frozen; 64 chains,
# seeds [7, chain]; states saved every 50 iterations; MPSRF window from iteration 1,000; fields from 10,000 samples.
python experiments/run_paper_chains.py --exp e7_smooth_sub --chains 64 --workers 64 --seed0 7 --save-every 50 \
    --eps 0.003 --data-seed 2026 --out "$OUT"
python experiments/run_paper_post.py --dir "$OUT" --n-fields 10000 --mpsrf-start 1000 --seed 11
