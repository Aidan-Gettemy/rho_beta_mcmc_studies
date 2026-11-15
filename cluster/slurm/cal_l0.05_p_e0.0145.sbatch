#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time=04:00:00
#SBATCH -J msm-cal_l0.05_p_e0.0145
#SBATCH -o logs/msm-cal_l0.05_p_e0.0145-%j.out
#
# PHASE 0b noise calibration, global pCN, pressure only: ell 0.05, grid 64x64, 2048 checkerboard sensors, eps = 0.0145.
# likelihood sigma_rel = eps/sqrt(2048) = 0.000320408.  Target: mean posterior sd of log k of about 0.4 (Laplace estimate eps ~ 0.0145).
# 16 chains, pilot 50000 adapted iterations (floor 1e-4, window 500), restart 500000 with the step frozen, saved every 50; MPSRF window from 1000.
# Submit from the repo root:   sbatch cluster/slurm/cal_l0.05_p_e0.0145.sbatch
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
echo "python: $(command -v python)"; python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
export BRHO_ROOT=$PWD/runs/msm_local_kle_2026-11
export PYTHONPATH=$PWD:${PYTHONPATH:-}:$PWD/experiments
OUT=$BRHO_ROOT/cal_l0.05_p_e0.0145
PILOT=${OUT}_pilot

COMMON="--kind global --chains 16 --workers 16 --ell 0.05 --grid 64 --kle-energy 0.95 --sigma-rel 0.00032040776 --data-relerr 0.0145 --data-seed 2026"

python experiments/run_msm_chains.py $COMMON --save-every 50 --n-iter 50000 --adapt 50000 --burn 25000 --s0 0.2 --s-min 0.0001 --adapt-every 500 --seed0 7 --out "$PILOT"
python experiments/run_msm_chains.py $COMMON --save-every 50 --n-iter 500000 --adapt 0 --burn 1000 --step-from "$PILOT" --init-scale 1.0 --seed0 1007 --out "$OUT"
python experiments/run_msm_post.py --dir "$OUT" --n-fields 2000 --mpsrf-start 1000 --mpsrf-step-its 5000 --seed 11
