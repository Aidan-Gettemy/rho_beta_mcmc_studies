"""Writes the eight self-contained sbatch scripts of the paper-MSM study (cluster/slurm/paper_*.sbatch), the prior-test
script and cluster/submit_paper_msm.sh.   python cluster/gen_paper_sbatch.py   (run from the repo root; outputs committed)."""
import os
import sys

sys.path.insert(0, os.getcwd())
from brho.msmpaper import EXPS  # noqa: E402

DESC = {"e0_global_all": "control: global pCN, all 20 thetas at once",
        "e1_global_one": "global pCN, one theta at a time",
        "e2_raw_all": "MSM, 20 local coefficients, no projection (discontinuous field), all at once",
        "e3_proj_all": "MSM with projection onto the global KLE (theta = B c), all at once",
        "e4_proj_one": "MSM with projection, one local coefficient at a time",
        "e5_smooth_all": "MSM with the paper's interface smoothing (plain average, radius 0.1), all at once",
        "e6_smooth_one": "MSM with the paper's smoothing, one local coefficient at a time",
        "e7_smooth_sub": "MSM with the paper's smoothing, one subdomain at a time (5 coefficients, stored neighbours)"}

HEAD = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time={time}
#SBATCH -J {name}
#SBATCH -o logs/{name}-%j.out
#
# Paper-MSM study (docs/paper_msm_experiment.md).  {desc}
# Submit from the repo root:   sbatch cluster/slurm/{name}.sbatch      (all: bash cluster/submit_paper_msm.sh)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
export PYTHONPATH=$PWD:${{PYTHONPATH:-}}
python tests/run_tests.py test_dirichlet test_gridstudy test_msmpaper
"""
CHAIN = """OUT=runs/paper_msm_2026-11/{exp}
# {n_iter:,} iterations, step size adapted to 25% acceptance for the first M = {adapt:,}, then frozen; 64 chains,
# seeds [7, chain]; states saved every 50 iterations; MPSRF window from iteration 1,000; fields from 10,000 samples.
python experiments/run_paper_chains.py --exp {exp} --chains 64 --workers 64 --seed0 7 --save-every 50 \\
    --eps 0.003 --data-seed 2026 --out "$OUT"
python experiments/run_paper_post.py --dir "$OUT" --n-fields 10000 --mpsrf-start 1000 --seed 11
"""
PRIOR = """python experiments/run_paper_prior.py --out runs/paper_msm_2026-11/prior_test --chains 64 --n-iter 40000 --workers 64
"""
os.makedirs("cluster/slurm", exist_ok=True)
names = []
for exp, spec in EXPS.items():
    name = f"paper_{exp}"
    txt = HEAD.format(time="00:40:00", name=name, desc=DESC[exp]) + CHAIN.format(exp=exp, n_iter=spec["n_iter"], adapt=spec["adapt"])
    open(f"cluster/slurm/{name}.sbatch", "w").write(txt)
    names.append(name)
open("cluster/slurm/paper_prior.sbatch", "w").write(
    HEAD.format(time="00:30:00", name="paper_prior", desc="prior test (likelihood off), all eight field constructions, s = 0.2 and 0.7") + PRIOR)
names.append("paper_prior")
sub = ["#!/bin/bash", "# Submit the paper-MSM study (docs/paper_msm_experiment.md).  Repo root: bash cluster/submit_paper_msm.sh   (DRY=1 prints only)",
       "set -euo pipefail", 'cd "$(dirname "$0")/.."', "mkdir -p logs", "JOBS=("] + [f"  {j}" for j in names] + [
       ")", 'for j in "${JOBS[@]}"; do', "    f=cluster/slurm/$j.sbatch",
       '    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }',
       '    if [ -n "${DRY:-}" ]; then echo "sbatch $f"; else echo -n "$j: "; sbatch "$f"; fi', "done", ""]
open("cluster/submit_paper_msm.sh", "w").write("\n".join(sub))
os.chmod("cluster/submit_paper_msm.sh", 0o755)
print("\n".join(names))
