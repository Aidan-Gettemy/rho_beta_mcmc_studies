#!/usr/bin/env python3
"""Phase 0b noise calibration of the global pCN, PRESSURE + FLUX (same eps on both), for the dd_end study:
find eps with mean posterior sd of log k = 0.25 (prior sd ~ 0.98).  eps = per-datum noise relative to the rms pressure;
likelihood sigma_rel = eps / sqrt(n_sensors).  Case: ell 0.05 on 64x64 (2048 sensors, 448 flux faces, KLE energy 0.95 -> r = 208) at eps 0.1, 0.2, 0.4 (ell 0.1 gave sd 0.153 at 0.1 and 0.336 at 0.3).
Run from the repo root:  python3 cluster/gen_calibration_sbatch.py"""
import os

T = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={chains}
#SBATCH --mem=32G
#SBATCH --partition=g1mig
#SBATCH --time={time}
#SBATCH -J msm-{tag}
#SBATCH -o logs/msm-{tag}-%j.out
#
# PHASE 0b noise calibration, global pCN, pressure + flux (flux weight eps/sqrt({nflux}) per datum, 14 flux lines every other face): ell {ell}, grid {grid}x{grid}, {nobs} checkerboard sensors, eps = {eps}.
# likelihood sigma_rel = eps/sqrt({nobs}) = {sigma:.6g}.  Target: mean posterior sd of log k of about 0.25.
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

COMMON="--kind global --chains {chains} --workers {chains} --ell {ell} --grid {grid} --kle-energy {energy} --sigma-rel {sigma:.8g} --data-relerr {eps} --data-seed 2026 --flux-noise-rel {eps} --flux-sigma-rel {fsig:.8g}"

python experiments/run_msm_chains.py $COMMON --save-every 50 --n-iter {pilot} --adapt {pilot} --burn {half} --s0 0.2 --s-min 0.0001 --adapt-every 500 --seed0 7 --out "$PILOT"
python experiments/run_msm_chains.py $COMMON --save-every 50 --n-iter {restart} --adapt 0 --burn 1000 --step-from "$PILOT" --init-scale 1.0 --seed0 1007 --out "$OUT"
python experiments/run_msm_post.py --dir "$OUT" --n-fields 2000 --mpsrf-start 1000 --mpsrf-step-its 5000 --seed 11
"""
os.makedirs("cluster/slurm", exist_ok=True)
jobs = [dict(ell=0.05, grid=64, nobs=2048, nflux=448, energy=0.95, eps=e, chains=16, pilot=50000, restart=300000, time="03:00:00") for e in (0.1, 0.2, 0.4)]
names = []
for j in jobs:
    j["sigma"] = j["eps"] / j["nobs"] ** 0.5; j["half"] = j["pilot"] // 2; j["fsig"] = j["eps"] / j["nflux"] ** 0.5
    j["tag"] = f"cal_l{j['ell']}_pf_e{j['eps']}"
    open(f"cluster/slurm/{j['tag']}.sbatch", "w").write(T.format(**j)); names.append(j["tag"])
open("cluster/submit_calibration.sh", "w").write("#!/bin/bash\n# submit the Phase 0b calibration jobs (run from the repo root)\n" + "".join(f"sbatch cluster/slurm/{n}.sbatch\n" for n in names))
print("\n".join(names))
