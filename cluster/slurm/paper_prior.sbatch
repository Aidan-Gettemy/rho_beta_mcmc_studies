#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=00:30:00
#SBATCH -J paper_prior
#SBATCH -o logs/paper_prior-%j.out
#
# Paper-MSM study (docs/paper_msm_experiment.md).  prior test (likelihood off), all eight field constructions, s = 0.2 and 0.7
# Submit from the repo root:   sbatch cluster/slurm/paper_prior.sbatch      (all: bash cluster/submit_paper_msm.sh)
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
python experiments/run_paper_prior.py --out runs/paper_msm_2026-11/prior_test --chains 64 --n-iter 40000 --workers 64
