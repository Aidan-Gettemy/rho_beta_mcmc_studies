#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem=64G
#SBATCH --partition=g1mig
#SBATCH --time=01:00:00
#SBATCH -J dd_end_s4_t2026_l0.05_pf_b2_pooled
#SBATCH -o logs/dd_end_s4_t2026_l0.05_pf_b2_pooled-%j.out
#
# dd_end study, stage 4 rerun (the one-job version timed out at 12 h): pooled post of l-4-DD and l-4-DD+ (run after both)
# Cells (labels of the sheet): l-4-DD, l-4-DD+.  Truth seed 2026.  eps frozen per ell (experiments/dd_end_cells.py).
# 64 chains per sampler; pilot with the step adapted, then a restart with the step frozen; PROVISIONAL iteration counts
# (shortened for this rerun: DD pilot 30000, restart 60000 with burn 12000 (20%); --auto-burn extends it if needed): extend any run with sup-MPSRF above 1.2 before reading it.
# Submit from the repo root:   sbatch cluster/slurm/dd_end_s4_t2026_l0.05_pf_b2_pooled.sbatch   (or bash cluster/submit_dd_end_s4_pf_b2_split.sh)
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

python experiments/run_dd_post.py --dirs "runs/msm_local_kle_2026-11/dd_end/l0.05/pf/l-4-DD_t2026" "$OUT" --out "$OUT/post_pooled" --n-fields 2000 --mpsrf-start 0 --mpsrf-log --auto-burn --seed 11 --ref runs/msm_local_kle_2026-11/dd_end/l0.05/pf/f_t2026/post/post_summary.json
