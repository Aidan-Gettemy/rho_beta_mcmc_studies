#!/usr/bin/env python3
"""Write the exp. 18 sbatch scripts: NON-overlapping domain decomposition (Robin alpha = 0.1 + plain Gibbs for h, interior flux, noise-level
flux weights) with ALL noise levels divided by a factor f in {1, 3, 6.5}, primal (10 modes) and dual (99% energy).  Question: does the DD with
f times more precise data reach the sd / error of the global sampler at the nominal noise (exp. 14)?  f = 1 is the matched control.
Run from the repo root:  python3 cluster/gen_noise_sbatch.py"""
import os

TEMPLATE = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem={mem}
#SBATCH --partition=g1mig
#SBATCH --time={time}
#SBATCH -J msm-{tag}
#SBATCH -o logs/msm-{tag}-%j.out
#
# EXP. 18 -- noise scaling of the NON-overlapping DD: {grid} decomposition, ALL noise levels divided by f = {f}.
#   data noise {relerr:.6g} (nominal 0.003), likelihood sigma_rel {sigma:.6g} (nominal 0.00013258252), flux noise {relerr:.6g},
#   flux weight per datum {fsig:.6g} (nominal 0.00020045 = noise/sqrt(224)), Gibbs prior sd of h on the measured faces {relerr:.6g} (nominal 0.003).
# Everything else as the matched exp. 17 settings WITHOUT overlap: Robin alpha 0.1, plain Gibbs for h (sd 0.03 x rms face flux,
# correlation length 4 cells), interior-flux likelihood, the standard 14 flux lines (224 faces), local basis {basis}, truth = seed-2026 draw.
# Reference: the converged global run at the NOMINAL noise (exp. 14, error 0.0045).  f = 1 is the matched non-overlapping control.
# Sharper data -> smaller pCN steps -> longer chains: {chains} chains, pilot {pilot} (adapted, floor {smin}, window {every}), restart {restart} iterations
# with the step frozen, saved every {save}.  If MPSRF is still above 1.2 at the end, extend (more iterations) before reading the errors.
# Submit from the repo root:   sbatch cluster/slurm/msm_{name}.sbatch      (or: bash cluster/submit_noise_scaling.sh)
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
echo "python: $(command -v python)"; python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

# ---- one thread per process; compiled-kernel cache in node-local scratch ------------------------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
export BRHO_ROOT=$PWD/runs/msm_local_kle_2026-11
export PYTHONPATH=$PWD:${{PYTHONPATH:-}}
OUT=$BRHO_ROOT/{name}
PILOT=${{OUT}}_pilot

# ---- tests that must pass before sampling (no pytest needed) ------------------------------------
python tests/run_tests.py test_msm_dd

COMMON="--grid {grid} --chains {chains} --workers 64 --smoothing 1e-3 --neighbors 60 --sigma-rel {sigma:.8g} --data-relerr {relerr:.8g} --data-seed 2026 --flux-noise-rel {relerr:.8g} --flux-sigma-rel {fsig:.8g} --robin-alpha 0.1 --h-prior-rel 0.03 --h-prior-meas-rel {relerr:.8g} --h-corr-len 4 --interior-flux {basisflag} --s-min {smin} --adapt-every {every}"

# ---- stage 1: pilot, step size adapted throughout --------------------------------------------------
python experiments/run_dd_chains.py $COMMON \\
    --save-every 50 --n-iter {pilot} --adapt {pilot} --burn {pilotburn} --s0 0.2 --seed0 7 \\
    --out "$PILOT"

# ---- stage 2: restart from dispersed prior draws with the step size frozen ---------------------------
python experiments/run_dd_chains.py $COMMON \\
    --save-every {save} --n-iter {restart} --adapt 0 --burn 1000 --step-from "$PILOT" --init-scale 1.0 --seed0 1007 \\
    --out "$OUT"

# ---- post-processing: fields, errors, MPSRF; reference = the converged global run at the nominal noise (exp. 14)
python experiments/run_dd_post.py --dirs "$OUT" --out "$OUT/post" --n-fields 5000 --mpsrf-start 1000 --mpsrf-step-its {mstep} --seed 11 --ref "$BRHO_ROOT/exp14_global_sharp_fluxnoise_10M/post/post_summary.json"
"""

# f: (label, chains, pilot its, restart its, save every, mpsrf step, step floor, adapt window, mem, time)
SETTINGS = {
    1.0: ("1", 64, 10000, 100000, 50, 100, 1e-3, 100, "32G", "03:00:00"),
    3.0: ("3", 32, 30000, 1000000, 200, 1000, 1e-4, 500, "64G", "06:00:00"),
    6.5: ("6p5", 32, 50000, 3000000, 400, 2000, 1e-4, 500, "64G", "12:00:00"),
}
os.makedirs(os.path.join("cluster", "slurm"), exist_ok=True)
names = []
for f, (lab, chains, pilot, restart, save, mstep, smin, every, mem, time) in SETTINGS.items():
    for grid in ("primal", "dual"):
        name = f"exp18_noise_f{lab}_{grid}"
        basis = "10 modes (primal-local-dim 10)" if grid == "primal" else "99% local KLE energy"
        basisflag = "--primal-local-dim 10" if grid == "primal" else "--dual-energy 0.99"
        path = os.path.join("cluster", "slurm", f"msm_{name}.sbatch")
        with open(path, "w") as fh:
            fh.write(TEMPLATE.format(name=name, tag=name.replace("_", "-"), grid=grid, f=f, relerr=0.003 / f, sigma=0.00013258252 / f,
                                     fsig=0.00020045 / f, basis=basis, basisflag=basisflag, chains=chains, pilot=pilot,
                                     pilotburn=pilot // 2, restart=restart, save=save, mstep=mstep, smin=smin, every=every,
                                     mem=mem, time=time))
        names.append(name); print(path)
with open(os.path.join("cluster", "submit_noise_scaling.sh"), "w") as fh:
    fh.write("#!/bin/bash\n# Submit the six exp. 18 jobs (independent).  Run from the repo root:  bash cluster/submit_noise_scaling.sh\nset -euo pipefail\n")
    for n in names:
        fh.write(f"sbatch cluster/slurm/msm_{n}.sbatch\n")
print("cluster/submit_noise_scaling.sh")
