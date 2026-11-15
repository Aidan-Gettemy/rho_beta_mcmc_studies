#!/usr/bin/env python3
"""Pressure-only noise calibration of the global pCN for the dd_end study: find eps with mean posterior sd of log k = 0.5
(prior sd ~ 0.98; pressure + flux cells target 0.25).  A Laplace scan gave eps ~ 0.012 / 0.033 / 0.032 at ell 0.2 / 0.1 / 0.05
and a short 8-chain pCN run at ell 0.2 gave sd 0.531 at 0.0124, so each ell gets three eps (x0.75, x1, x1.3) to bracket the target.
64 chains per job (the official chain count).  Run from the repo root:  python3 cluster/gen_calibration_p_sbatch.py"""
import os

T = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={chains}
#SBATCH --mem=64G
#SBATCH --partition=g1mig
#SBATCH --time={time}
#SBATCH -J msm-{tag}
#SBATCH -o logs/msm-{tag}-%j.out
#
# PRESSURE-ONLY noise calibration, global pCN: ell {ell}, grid {grid}x{grid}, {nobs} checkerboard sensors, eps = {eps}.
# likelihood sigma_rel = eps/sqrt({nobs}) = {sigma:.6g}.  Target: mean posterior sd of log k of about 0.5 (the pressure+flux cells target 0.25).
# {chains} chains, pilot {pilot} adapted iterations (floor 1e-4, window 500), restart {restart} with the step frozen, saved every 50; MPSRF window from 1000.
# Submit from the repo root:   sbatch cluster/slurm/{tag}.sbatch
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
echo "python: $(command -v python)"; python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
export BRHO_ROOT=$PWD/runs/msm_local_kle_2026-11
export PYTHONPATH=$PWD:${{PYTHONPATH:-}}:$PWD/experiments
OUT=$BRHO_ROOT/{tag}
PILOT=${{OUT}}_pilot

COMMON="--kind global --chains {chains} --workers {chains} --ell {ell} --grid {grid} --kle-energy {energy} --sigma-rel {sigma:.8g} --data-relerr {eps} --data-seed 2026"

python experiments/run_msm_chains.py $COMMON --save-every 50 --n-iter {pilot} --adapt {pilot} --burn {half} --s0 0.2 --s-min 0.0001 --adapt-every 500 --seed0 7 --out "$PILOT"
python experiments/run_msm_chains.py $COMMON --save-every 50 --n-iter {restart} --adapt 0 --burn 1000 --step-from "$PILOT" --init-scale 1.0 --seed0 1007 --out "$OUT"
python experiments/run_msm_post.py --dir "$OUT" --n-fields 2000 --mpsrf-start 1000 --mpsrf-step-its 5000 --seed 11
"""
os.makedirs("cluster/slurm", exist_ok=True)
CASES = {0.2: dict(grid=16, nobs=128, energy=0.966, eps0=0.0124, pilot=20000, restart=400000, time="02:00:00"),
         0.1: dict(grid=32, nobs=512, energy=0.966, eps0=0.033, pilot=30000, restart=600000, time="04:00:00"),
         0.05: dict(grid=64, nobs=2048, energy=0.95, eps0=0.032, pilot=50000, restart=1000000, time="06:00:00")}
names = []
for ell, c in CASES.items():
    for f in (0.75, 1.0, 1.3):
        j = dict(c, ell=ell, eps=float(f"{c['eps0'] * f:.3g}"), chains=64)
        j["sigma"] = j["eps"] / j["nobs"] ** 0.5; j["half"] = j["pilot"] // 2
        j["tag"] = f"cal_l{ell}_p_e{j['eps']}"
        open(f"cluster/slurm/{j['tag']}.sbatch", "w").write(T.format(**j)); names.append(j["tag"])
open("cluster/submit_calibration_p.sh", "w").write("#!/bin/bash\n# submit the pressure-only calibration jobs (run from the repo root)\n" + "".join(f"sbatch cluster/slurm/{n}.sbatch\n" for n in names))
print("\n".join(names))
