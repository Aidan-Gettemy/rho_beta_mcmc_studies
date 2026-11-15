#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=g1mig
#SBATCH --time=01:00:00
#SBATCH -J msm-cal_l0.1_p_e0.0248
#SBATCH -o logs/msm-cal_l0.1_p_e0.0248-%j.out
#
# PRESSURE-ONLY noise calibration, global pCN: ell 0.1, grid 32x32, 512 checkerboard sensors, eps = 0.0248.
# likelihood sigma_rel = eps/sqrt(512) = 0.00109602.  Target: mean posterior sd of log k of about 0.5 (the pressure+flux cells target 0.25).
# 64 chains, pilot 20000 adapted iterations (floor 1e-4, window 500), restart 200000 with the step frozen, saved every 50; MPSRF window from 1000.
# Submit from the repo root:   sbatch cluster/slurm/cal_l0.1_p_e0.0248.sbatch
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
OUT=$BRHO_ROOT/cal_l0.1_p_e0.0248
PILOT=${OUT}_pilot

COMMON="--kind global --chains 64 --workers 64 --ell 0.1 --grid 32 --kle-energy 0.966 --sigma-rel 0.0010960155 --data-relerr 0.0248 --data-seed 2026"

python experiments/run_msm_chains.py $COMMON --save-every 50 --n-iter 20000 --adapt 20000 --burn 10000 --s0 0.2 --s-min 0.0001 --adapt-every 500 --seed0 7 --out "$PILOT"
python experiments/run_msm_chains.py $COMMON --save-every 50 --n-iter 200000 --adapt 0 --burn 1000 --step-from "$PILOT" --init-scale 1.0 --seed0 1007 --out "$OUT"
python experiments/run_msm_post.py --dir "$OUT" --n-fields 2000 --mpsrf-start 1000 --mpsrf-step-its 5000 --seed 11
