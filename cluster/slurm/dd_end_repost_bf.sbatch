#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=g1mig
#SBATCH --time=01:00:00
#SBATCH -J dd_end_repost_bf
#SBATCH -o logs/dd_end_repost_bf-%j.out
#
# dd_end stage 2: re-run ONLY the post-processing of cells b and f with a longer burn-in (their chains are kept on the cluster).
# The burn-in is set to the iteration from which the sup-MPSRF stays below 1.2 (b: ~6.3k -> 7000; f: ~356k -> 360000).  No sampling.
# Submit from the repo root:   sbatch cluster/slurm/dd_end_repost_bf.sbatch
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

R=runs/msm_local_kle_2026-11/dd_end
for pair in "l0.2/pf/b_t2026 7000" "l0.05/pf/f_t2026 360000"; do
    set -- $pair
    test -f "$R/$1/chains.npz" || { echo "ERROR: no chains in $R/$1" >&2; exit 1; }
    python experiments/run_msm_post.py --dir "$R/$1" --n-fields 2000 --mpsrf-start 0 --mpsrf-log --burn "$2" --seed 11
done
echo done
