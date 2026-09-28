"""
Extend a finished (or stopped) run to more iterations, keeping every sample
already drawn: raises the run length in the manifest, then the usual
RESUME=1 launch continues each chain exactly from its checkpoint (RNG state,
frozen step sizes, kept samples).  Burn-in, thinning, step sizes and all
other settings are unchanged; the extension is recorded in the manifest
("extensions": from, to, time, note).

    python experiments/extend_run.py --exp e5_soft    --to 400000
    python experiments/extend_run.py --exp e1_blocked --to 600000
    then:  RESUME=1 sbatch cluster/slurm/node_e5_soft.sbatch
           RESUME=1 sbatch cluster/slurm/mpi_e1_naive.sbatch

Supported: E5 (SoftSamplerConfig.n_total), E4 (SamplerConfig.n_total) and
E1 (NaiveConfig.n_iter).  Exactness is tested (tests: resume/extension).
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import argparse
import glob
import os
import time

import numpy as np

from brho import io

KEYS = (("SoftSamplerConfig", "n_total"), ("SamplerConfig", "n_total"),
        ("NaiveConfig", "n_iter"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="e.g. e5_soft, e1_blocked")
    ap.add_argument("--to", type=int, required=True, help="new total iterations per chain")
    ap.add_argument("--note", default="", help="why (stored in the manifest)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    man = io.read_manifest(a.exp)
    found = [(c, k) for c, k in KEYS if c in man["configs"]]
    if not found:
        raise SystemExit(f"{a.exp}: no supported sampler config in the manifest")
    cls, key = found[0]
    old = int(man["configs"][cls][key])
    if a.to <= old:
        raise SystemExit(f"{a.exp}: --to {a.to} must exceed the current {key} = {old}")

    paths = glob.glob(os.path.join(io.results_dir(a.exp), "chain*.npz"))
    its = sorted({int(np.load(p)["it"]) for p in paths})
    if not paths:
        raise SystemExit(f"{a.exp}: no chain files in {io.results_dir(a.exp)}")
    print(f"{a.exp}: {len(paths)} chains at iteration(s) {its}; {cls}.{key} {old} -> {a.to}")
    if its != [old]:
        print("  note: not every chain is at the old end; RESUME=1 finishes them too")
    burn = man["configs"][cls].get("n_burn")
    print(f"  unchanged: n_burn {burn}, thin {man['configs'][cls].get('thin')}, step sizes "
          "(frozen at the end of burn-in)")
    if a.dry_run:
        return
    man["configs"][cls][key] = a.to
    man.setdefault("extensions", []).append(
        dict(key=f"{cls}.{key}", old=old, new=a.to, note=a.note,
             time=time.strftime("%Y-%m-%d %H:%M:%S")))
    man["status"] = f"extending to {a.to}"
    io.save_json(os.path.join(io.results_dir(a.exp), "manifest.json"), man)
    print(f"  manifest updated; now resubmit that run with RESUME=1")


if __name__ == "__main__":
    main()
