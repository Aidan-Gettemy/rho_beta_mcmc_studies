#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem=64G
#SBATCH --partition=g1mig
#SBATCH --time=02:00:00
#SBATCH -J dd_end_s6_t2027_d
#SBATCH -o logs/dd_end_s6_t2027_d-%j.out
#
# dd_end study, stage 6: extra truth on the 4 x 4 column
# Cells (labels of the sheet): d.  Truth seed 2027.  eps frozen per ell (experiments/dd_end_cells.py).
# 64 chains per sampler; pilot with the step adapted, then a restart with the step frozen; PROVISIONAL iteration counts
# (pilot 30000, restart 500000 global / 300000 DD): extend any run with sup-MPSRF above 1.2 before reading it.
# Submit from the repo root:   sbatch cluster/slurm/dd_end_s6_t2027_d.sbatch
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
export PYTHONPATH=$PWD:${PYTHONPATH:-}:$PWD/experiments
python tests/run_tests.py test_msm_dd

# ---- d: global pCN reference
OUT=runs/msm_local_kle_2026-11/dd_end/l0.1/pf/d_t2027
python experiments/run_msm_chains.py --kind global --ell 0.1 --grid 32 --kle-energy 0.966 --sigma-rel 0.0088388348 --data-relerr 0.2 --data-seed 2027 --flux-noise-rel 0.2 --flux-sigma-rel 0.013363062 \
    --chains 64 --workers 64 --save-every 16 --n-iter 30000 --adapt 30000 --burn 15000 --s0 0.2 --s-min 0.0001 --adapt-every 500 --seed0 7 \
    --out "${OUT}_pilot"
python experiments/run_msm_chains.py --kind global --ell 0.1 --grid 32 --kle-energy 0.966 --sigma-rel 0.0088388348 --data-relerr 0.2 --data-seed 2027 --flux-noise-rel 0.2 --flux-sigma-rel 0.013363062 \
    --chains 64 --workers 64 --save-every 16 --n-iter 500000 --adapt 0 --burn 25000 --step-from "${OUT}_pilot" --init-scale 1.0 --seed0 1007 \
    --out "$OUT"
python experiments/run_msm_post.py --dir "$OUT" --n-fields 2000 --mpsrf-start 0 --mpsrf-log --auto-burn --seed 11
