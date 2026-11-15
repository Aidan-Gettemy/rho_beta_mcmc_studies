#!/usr/bin/env python3
"""Write the sbatch scripts and the manifest of the dd_end study (cells and frozen eps: experiments/dd_end_cells.py).

    python3 cluster/gen_dd_end_sbatch.py --stage 2                 # global references (cells a..f), truth 2026
    python3 cluster/gen_dd_end_sbatch.py --stage 3                 # MSM-basis global, DD primal and DD dual + pooled, ell 0.2 and 0.1
    python3 cluster/gen_dd_end_sbatch.py --stage 4                 # the same at ell 0.05
    python3 cluster/gen_dd_end_sbatch.py --stage 6 --truth 2027    # extra truth on the 4 x 4 column (all ell)
    python3 cluster/gen_dd_end_sbatch.py --stage 7                 # noise sweep (ell 0.1, 4 x 4): eps x 0.5 and x 2
    python3 cluster/gen_dd_end_sbatch.py --stage 8                 # overlap test j-16-DD-H

Every job is self-contained (all flags in the script, conda environment checked).  Official runs use 64 parallel chains per sampler
(per block for the DD).  Each run is the tune-then-restart protocol: a pilot with the step size adapted (target 25%, floor 1e-4,
window 500), then a restart from fresh prior draws with the step frozen; one state in 50 saved; MPSRF window from iteration 1000.
Iteration counts and time limits (ITERS) are PROVISIONAL: set them from the timing of the stage-2 reference runs and extend any run
whose sup-MPSRF is above 1.2 (slow convergence is acceptable; report it).

Output folders: runs/msm_local_kle_2026-11/dd_end/l<ell>/<p|pf>/<label>_t<truth>[_e<factor>]/   (DD+ cells: the dual run, with the
pooled primal + dual post in post_pooled).  The manifest runs/msm_local_kle_2026-11/dd_end/manifest.csv lists every run.
Return only the small files (summary.json, post/): see docs/dd_end_runbook.md."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import dd_end_cells as C      # noqa: E402

ROOT = "runs/msm_local_kle_2026-11/dd_end"
CHAINS = 64
# provisional: (pilot, restart for the global samplers, restart for the DD samplers, hours per run)
ITERS = {0.2: (20000, 300000, 200000, 2), 0.1: (30000, 500000, 300000, 4), 0.05: (50000, 1500000, 500000, 12)}

HEADER = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # one worker process per (subdomain, chain) task, 64 at a time
#SBATCH --mem=64G
#SBATCH --partition=g1mig
#SBATCH --time={time}
#SBATCH -J {tag}
#SBATCH -o logs/{tag}-%j.out
#
# dd_end study, stage {stage}: {desc}
# Cells (labels of the sheet): {labels}.  Truth seed {truth}.  eps frozen per ell (experiments/dd_end_cells.py).
# {chains} chains per sampler; pilot with the step adapted, then a restart with the step frozen; PROVISIONAL iteration counts
# (pilot {pilot}, restart {restart_g} global / {restart_d} DD): extend any run with sup-MPSRF above 1.2 before reading it.
# Submit from the repo root:   sbatch cluster/slurm/{tag}.sbatch
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
export PYTHONPATH=$PWD:${{PYTHONPATH:-}}:$PWD/experiments
python tests/run_tests.py test_msm_dd
"""

RUN = """
# ---- {label}: {what}
OUT={out}
python experiments/{script} {flags} \\
    --chains {chains} --workers 64 --save-every 50 --n-iter {pilot} --adapt {pilot} --burn {half} --s0 0.2 --s-min 0.0001 --adapt-every 500 --seed0 7 \\
    --out "${{OUT}}_pilot"
python experiments/{script} {flags} \\
    --chains {chains} --workers 64 --save-every 50 --n-iter {restart} --adapt 0 --burn 1000 --step-from "${{OUT}}_pilot" --init-scale 1.0 --seed0 1007 \\
    --out "$OUT"
"""
POST_MSM = "python experiments/run_msm_post.py --dir \"$OUT\" --n-fields 2000 --mpsrf-start 1000 --mpsrf-step-its 5000 --seed 11\n"
POST_DD = ("python experiments/run_dd_post.py --dirs {dirs} --out \"{outpost}\" --n-fields 2000 --mpsrf-start 1000 --mpsrf-step-its 5000 "
           "--seed 11 --ref {ref}\n")


def run_dir(cell, truth, factor=1.0):
    p = "pf" if cell["flux"] else "p"
    tag = f"{cell['label']}_t{truth}" + (f"_e{factor:g}" if factor != 1.0 else "")
    return f"{ROOT}/l{cell['ell']}/{p}/{tag}"


def global_ref(cell, truth, factor=1.0):
    g = [c for c in C.cells() if c["ell"] == cell["ell"] and c["flux"] == cell["flux"] and c["method"] == "pcn"][0]
    return run_dir(g, truth, factor) + "/post/post_summary.json"


def block(cell, truth, factor=1.0, dual_primal_out=None):
    """Shell text of one run (pilot, restart, post) for a cell."""
    pilot, rg, rd, _ = ITERS[cell["ell"]]
    script, flags = C.flags(cell, truth_seed=truth, eps_factor=factor)
    m = cell["method"]
    out = run_dir(cell, truth, factor)
    txt = RUN.format(label=cell["label"], what={"pcn": "global pCN reference", "msm": "global pCN in the MSM basis", "dd": "DD, primal blocks",
                                                 "ddplus": "DD, dual blocks (+ pooled with the primal run)", "ddH": "DD, primal blocks with overlap"}[m],
                     out=out, script=script, flags=flags, chains=CHAINS, pilot=pilot, half=pilot // 2,
                     restart=rg if m in ("pcn", "msm") else rd)
    if m in ("pcn", "msm"):
        return txt + POST_MSM
    ref = global_ref(cell, truth, factor)
    if m == "ddplus":
        prim = dict(cell, method="dd", label=C.label(cell["ell"], cell["flux"], "dd", cell["nb"]))
        pdir = run_dir(prim, truth, factor)
        return (txt + POST_DD.format(dirs='"$OUT"', outpost="$OUT/post", ref=ref)
                + POST_DD.format(dirs=f'"{pdir}" "$OUT"', outpost="$OUT/post_pooled", ref=ref))
    return txt + POST_DD.format(dirs='"$OUT"', outpost="$OUT/post", ref=ref)


def select(stage, truth):
    cs = C.cells()
    if stage == 2:
        return [c for c in cs if c["method"] == "pcn"]
    if stage == 3:
        return [c for c in cs if c["method"] in ("msm", "dd", "ddplus") and c["ell"] in (0.2, 0.1)]
    if stage == 4:
        return [c for c in cs if c["method"] in ("msm", "dd", "ddplus") and c["ell"] == 0.05]
    if stage == 6:      # extra truths: the 4 x 4 column at every ell
        return [c for c in cs if c["method"] != "ddH" and (c["method"] == "pcn" or c["nb"] == 4)]
    if stage == 7:      # noise sweep: ell 0.1, 4 x 4: pCN, DD, DD with dual and pooling
        return [c for c in cs if c["ell"] == 0.1 and c["method"] in ("pcn", "dd", "ddplus") and (c["method"] == "pcn" or c["nb"] == 4)]
    if stage == 8:
        return [c for c in cs if c["method"] == "ddH"]
    raise SystemExit("stage must be 2, 3, 4, 6, 7 or 8 (stage 5, dual and pooled, is part of stages 3 and 4)")


def groups(stage, cells):
    """One job per (ell, data, blocks) family; ell 0.05 and the pCN references get one job per run (long runs)."""
    out = {}
    for c in cells:
        if c["ell"] == 0.05 or c["method"] == "pcn":
            key = (c["label"],)
        else:
            key = (c["ell"], c["flux"], c["nb"])
        out.setdefault(key, []).append(c)
    return list(out.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--truth", type=int, default=C.DATA_SEED)
    a = ap.parse_args()
    cells = select(a.stage, a.truth)
    factors = (0.5, 2.0) if a.stage == 7 else (1.0,)
    if a.stage == 6 and a.truth == C.DATA_SEED:
        raise SystemExit("stage 6 needs the extra truth seeds: --truth 2027 and --truth 2028")
    os.makedirs("cluster/slurm", exist_ok=True)
    names, rows = [], []
    for g in groups(a.stage, cells):
        for f in factors:
            ell = g[0]["ell"]
            pilot, rg, rd, hours = ITERS[ell]
            suffix = f"_e{f:g}" if f != 1.0 else ""
            tag = f"dd_end_s{a.stage}_t{a.truth}_" + ("_".join(sorted({c["label"] for c in g})) if len(g) == 1 else
                                                    f"l{ell}_{'pf' if g[0]['flux'] else 'p'}_b{g[0]['nb']}") + suffix
            tag = tag.replace("+", "plus")
            # a DD+ cell needs its primal run (pooled estimate): add the primal DD cell of the same family when missing
            have = {c["label"] for c in g}
            extra = [dict(c, method="dd", label=C.label(c["ell"], c["flux"], "dd", c["nb"])) for c in g
                     if c["method"] == "ddplus" and C.label(c["ell"], c["flux"], "dd", c["nb"]) not in have]
            order = {"pcn": 0, "msm": 1, "dd": 2, "ddplus": 3, "ddH": 4}
            run_cells = sorted(g + extra, key=lambda c: order[c["method"]])
            time = hours * len(run_cells)
            time = f"{min(time, 48):02d}:00:00"
            txt = HEADER.format(time=time, tag=tag, stage=a.stage, truth=a.truth, chains=CHAINS, pilot=pilot, restart_g=rg, restart_d=rd,
                                desc=("noise sweep, eps x " + f"{f:g}") if f != 1.0 else {2: "global pCN references", 3: "MSM-basis global, DD primal, DD dual + pooled (ell 0.2, 0.1)",
                                                                                        4: "MSM-basis global, DD primal, DD dual + pooled (ell 0.05)", 6: "extra truth on the 4 x 4 column",
                                                                                        7: "noise sweep", 8: "overlap test"}[a.stage],
                                labels=", ".join(c["label"] for c in run_cells))
            for c in run_cells:
                txt += block(c, a.truth, f)
                rows.append([c["label"], c["ell"], c["flux"], c["method"], c["nb"], a.truth, f, run_dir(c, a.truth, f), tag, "planned"])
            open(f"cluster/slurm/{tag}.sbatch", "w").write(txt)
            names.append(tag)
    sub = f"cluster/submit_dd_end_s{a.stage}_t{a.truth}.sh"
    open(sub, "w").write("#!/bin/bash\n# submit the dd_end stage %d jobs (run from the repo root)\nset -euo pipefail\n" % a.stage
                         + "".join(f"sbatch cluster/slurm/{n}.sbatch\n" for n in names))
    os.makedirs(ROOT, exist_ok=True)
    mpath = f"{ROOT}/manifest.csv"
    old = list(csv.reader(open(mpath))) if os.path.exists(mpath) else []
    keep = [r for r in old[1:] if r[7] not in {x[7] for x in rows}]
    with open(mpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "ell", "flux", "method", "blocks", "truth", "eps_factor", "dir", "job", "status"])
        w.writerows(keep + rows)
    print(f"{len(names)} jobs, {len(rows)} runs; submit with: bash {sub}")
    print("\n".join(names))


if __name__ == "__main__":
    main()
