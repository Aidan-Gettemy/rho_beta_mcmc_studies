#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem=64G
#SBATCH --partition=g1mig
#SBATCH --time=06:00:00
#SBATCH -J dd_end_s4_t2026_l0.05_pf_b8
#SBATCH -o logs/dd_end_s4_t2026_l0.05_pf_b8-%j.out
#
# dd_end study, stage 4: MSM-basis global, DD primal, DD dual + pooled (ell 0.05)
# Cells (labels of the sheet): l-64-DD, l-64-DD+.  Truth seed 2026.  eps frozen per ell (experiments/dd_end_cells.py).
# 64 chains per sampler; pilot with the step adapted, then a restart with the step frozen; PROVISIONAL iteration counts
# (pilot 50000, restart 1500000 global / 150000 DD): extend any run with sup-MPSRF above 1.2 before reading it.
# Submit from the repo root:   sbatch cluster/slurm/dd_end_s4_t2026_l0.05_pf_b8.sbatch
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

# ---- l-64-DD: DD, primal blocks
OUT=runs/msm_local_kle_2026-11/dd_end/l0.05/pf/l-64-DD_t2026
python experiments/run_dd_chains.py --grid primal --n 64 --ell 0.05 --kle-energy 0.95 --blocks 8 --primal-local-dim 0 --dual-energy 0.95 --sigma-rel 0.0019887378 --data-relerr 0.09 --data-seed 2026 --flux-noise-rel 0.09 --flux-sigma-rel 0.0042521003 --robin-alpha 0.1 --h-prior-rel 0.09486833 --h-prior-meas-rel 0.09 --h-corr-len 4.0 --interior-flux \
    --chains 64 --workers 64 --save-every 50 --n-iter 50000 --adapt 50000 --burn 25000 --s0 0.2 --s-min 0.0001 --adapt-every 500 --seed0 7 \
    --out "${OUT}_pilot"
python experiments/run_dd_chains.py --grid primal --n 64 --ell 0.05 --kle-energy 0.95 --blocks 8 --primal-local-dim 0 --dual-energy 0.95 --sigma-rel 0.0019887378 --data-relerr 0.09 --data-seed 2026 --flux-noise-rel 0.09 --flux-sigma-rel 0.0042521003 --robin-alpha 0.1 --h-prior-rel 0.09486833 --h-prior-meas-rel 0.09 --h-corr-len 4.0 --interior-flux \
    --chains 64 --workers 64 --save-every 50 --n-iter 150000 --adapt 0 --burn 30000 --step-from "${OUT}_pilot" --init-scale 1.0 --seed0 1007 \
    --out "$OUT"
python experiments/run_dd_post.py --dirs "$OUT" --out "$OUT/post" --n-fields 2000 --mpsrf-start 0 --mpsrf-log --auto-burn --seed 11 --ref runs/msm_local_kle_2026-11/dd_end/l0.05/pf/f_t2026/post/post_summary.json

# ---- l-64-DD+: DD, dual blocks (+ pooled with the primal run)
OUT=runs/msm_local_kle_2026-11/dd_end/l0.05/pf/l-64-DD+_t2026
python experiments/run_dd_chains.py --grid dual --n 64 --ell 0.05 --kle-energy 0.95 --blocks 8 --primal-local-dim 0 --dual-energy 0.95 --sigma-rel 0.0019887378 --data-relerr 0.09 --data-seed 2026 --flux-noise-rel 0.09 --flux-sigma-rel 0.0042521003 --robin-alpha 0.1 --h-prior-rel 0.09486833 --h-prior-meas-rel 0.09 --h-corr-len 4.0 --interior-flux \
    --chains 64 --workers 64 --save-every 50 --n-iter 50000 --adapt 50000 --burn 25000 --s0 0.2 --s-min 0.0001 --adapt-every 500 --seed0 7 \
    --out "${OUT}_pilot"
python experiments/run_dd_chains.py --grid dual --n 64 --ell 0.05 --kle-energy 0.95 --blocks 8 --primal-local-dim 0 --dual-energy 0.95 --sigma-rel 0.0019887378 --data-relerr 0.09 --data-seed 2026 --flux-noise-rel 0.09 --flux-sigma-rel 0.0042521003 --robin-alpha 0.1 --h-prior-rel 0.09486833 --h-prior-meas-rel 0.09 --h-corr-len 4.0 --interior-flux \
    --chains 64 --workers 64 --save-every 50 --n-iter 150000 --adapt 0 --burn 30000 --step-from "${OUT}_pilot" --init-scale 1.0 --seed0 1007 \
    --out "$OUT"
python experiments/run_dd_post.py --dirs "$OUT" --out "$OUT/post" --n-fields 2000 --mpsrf-start 0 --mpsrf-log --auto-burn --seed 11 --ref runs/msm_local_kle_2026-11/dd_end/l0.05/pf/f_t2026/post/post_summary.json
python experiments/run_dd_post.py --dirs "runs/msm_local_kle_2026-11/dd_end/l0.05/pf/l-64-DD_t2026" "$OUT" --out "$OUT/post_pooled" --n-fields 2000 --mpsrf-start 0 --mpsrf-log --auto-burn --seed 11 --ref runs/msm_local_kle_2026-11/dd_end/l0.05/pf/f_t2026/post/post_summary.json
