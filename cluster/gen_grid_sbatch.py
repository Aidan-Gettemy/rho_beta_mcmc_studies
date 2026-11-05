"""Writes the nine self-contained sbatch scripts of the grid-refinement study (cluster/slurm/grid_*.sbatch) and
cluster/submit_grid_arms.sh.   python cluster/gen_grid_sbatch.py   (run from the repo root; the outputs are committed)."""
import os

ARMS = [("arm1_check", "check", "checkerboard of the grid (128 / 512 / 2048 observations), fixed NORMALISED noise: "
         "sigma_p = 0.003 ||p_obs|| / sqrt(n_obs), relative-error likelihood"),
        ("arm2_fixed", "fixed", "128 observations on every grid (grid cell closest to the centre of each cell of the "
         "16 x 16 checkerboard; lower-left of the central four on 32 / 64), same noise and likelihood as arm 1"),
        ("arm3_abs", "abs", "checkerboard of the grid (128 / 512 / 2048 observations), FIXED per-observation noise "
         "sigma_abs = 0.003 ||p_obs^(16x16)|| = 0.0167, Gaussian likelihood sum r^2 / (2 sigma_abs^2)")]
TIME = {16: "00:30:00", 32: "00:45:00", 64: "01:30:00"}
N_ITER, ADAPT = 100000, 10000

TEMPLATE = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per chain (64 chains)
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time={time}
#SBATCH -J grid-{tag}
#SBATCH -o logs/grid-{tag}-%j.out
#
# Grid-refinement study of global pCN, {arm_id} on the {n} x {n} grid.  Design and rationale (including why
# sigma_abs = 0.0167): docs/grid_arms_experiment.md.  Self-contained: every run flag is written below.  Submit from
# the repo root:   sbatch cluster/slurm/grid_{tag}.sbatch      (all nine: bash cluster/submit_grid_arms.sh)
# Problem: Dirichlet 1 / 0 left / right, no flow top / bottom, sqexp kernel, correlation length 0.2, 20-mode
# Nystrom KLE, truth theta* = first 20 draws of seed 2026.  This job: {desc}.
# Settings: 64 chains, N = {n_iter:,} iterations, step size adapted to 25% acceptance for the first M = {adapt:,}
# iterations, then frozen; chain seeds [7, chain]; states saved every 50 iterations; MPSRF window from iteration
# 1,000; posterior fields from 10,000 samples.  Output: runs/grid_arms_2026-11/{tag}/ (chains.npz, summary.json, post/).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ---- conda environment (explicit, checked) -------------------------------------------------
set +u
module load miniconda/24.11.1 2>/dev/null || true
ENV=/home/arg210006/.conda/envs/gd24_numba
source activate "$ENV" || {{ eval "$(conda shell.bash hook)" && conda activate "$ENV"; }}
set -u
if [ "${{CONDA_PREFIX:-}}" != "$ENV" ]; then
    echo "ERROR: conda environment $ENV was not activated (python: $(command -v python))" >&2
    exit 1
fi
echo "python: $(command -v python)"; python -c "import numpy, scipy; print('numpy', numpy.__version__)"

# ---- one thread per process; compiled-kernel cache in node-local scratch ------------------------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
export BRHO_ROOT=$PWD/runs/grid_arms_2026-11
export PYTHONPATH=$PWD:${{PYTHONPATH:-}}
OUT=$BRHO_ROOT/{tag}

# ---- tests that must pass before sampling (no pytest needed) ------------------------------------
python tests/run_tests.py test_dirichlet test_gridstudy

# ---- chains ----------------------------------------------------------------------------------------
python experiments/run_grid_chains.py \\
    --arm {arm} --n {n} \\
    --chains 64 --workers 64 \\
    --n-iter {n_iter} --adapt {adapt} \\
    --s0 0.2 --save-every 50 \\
    --eps 0.003 --data-seed 2026 --seed0 7 \\
    --out "$OUT"

# ---- diagnostics: AR histogram, relative-error traces, MPSRF of theta, theta posterior, field maps ----
python experiments/run_grid_post.py --dir "$OUT" --n-fields 10000 --mpsrf-start 1000 --seed 11
"""
os.makedirs("cluster/slurm", exist_ok=True)
jobs = []
for arm_id, arm, desc in ARMS:
    for n in (16, 32, 64):
        tag = f"{arm_id}_n{n}"
        txt = TEMPLATE.format(time=TIME[n], tag=tag, arm_id=arm_id, n=n, desc=desc, arm=arm, n_iter=N_ITER, adapt=ADAPT)
        open(f"cluster/slurm/grid_{tag}.sbatch", "w").write(txt)
        jobs.append(f"grid_{tag}")
sub = ["#!/bin/bash",
       "# Submit the nine jobs of the grid-refinement study (docs/grid_arms_experiment.md).",
       "# Run from the repo root:   bash cluster/submit_grid_arms.sh        (add DRY=1 to only print the sbatch commands)",
       "set -euo pipefail", 'cd "$(dirname "$0")/.."', "mkdir -p logs", "JOBS=("] + [f"  {j}" for j in jobs] + [
       ")", 'for j in "${JOBS[@]}"; do', "    f=cluster/slurm/$j.sbatch",
       '    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }',
       '    if [ -n "${DRY:-}" ]; then echo "sbatch $f"; else echo -n "$j: "; sbatch "$f"; fi', "done", ""]
open("cluster/submit_grid_arms.sh", "w").write("\n".join(sub))
os.chmod("cluster/submit_grid_arms.sh", 0o755)
print("\n".join(jobs))
