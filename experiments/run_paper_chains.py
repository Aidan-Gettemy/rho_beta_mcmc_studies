"""
Chains of one experiment of the paper-MSM study (brho/msmpaper.py, docs/paper_msm_experiment.md).
One process per chain, the same chain seeds in every experiment.  Writes to <out>/ :
    chains.npz    relerr, acc, move, phi (n_iter); c (native state) and z (common coordinates) every save_every
                  iterations; s_hist; eta_thin (field every field_every iterations)
    summary.json  settings, data summary, per-group final step sizes and acceptance, relative-error statistics

    python experiments/run_paper_chains.py --exp e7_smooth_sub --chains 64 --out runs/paper_msm_2026-11/e7_smooth_sub
(n_iter, adaptation length and the initial step size default to the values of the experiment in brho.msmpaper.EXPS)
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from brho import msmpaper as M

_G = {}


def _init(args):
    _G["S"] = M.PaperSetup()
    _G["data"] = M.PaperData(_G["S"], args.eps, args.data_seed)
    _G["a"] = args


def _one(chain):
    a = _G["a"]
    t = time.time()
    o = M.run_chain(_G["S"], _G["data"], a.exp, a.seed0, chain, a.n_iter, a.adapt, a.s0, a.save_every, a.field_every,
                    0.25, a.s_min)
    o["_sec"] = time.time() - t
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=list(M.EXPS))
    ap.add_argument("--chains", type=int, default=64)
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--adapt", type=int, default=None)
    ap.add_argument("--s0", type=float, default=None)
    ap.add_argument("--s-min", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--field-every", type=int, default=500)
    ap.add_argument("--eps", type=float, default=0.003)
    ap.add_argument("--data-seed", type=int, default=2026)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = M.EXPS[a.exp]
    a.n_iter = spec["n_iter"] if a.n_iter is None else a.n_iter
    a.adapt = spec["adapt"] if a.adapt is None else a.adapt
    a.s0 = spec["s0"] if a.s0 is None else a.s0
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers or min(a.chains, os.cpu_count()), _init, (a,)) as pool:
        res = pool.map(_one, range(a.chains))
    keys = [k for k in res[0] if not k.startswith("_")]
    arr = {k: np.stack([o[k] for o in res]) for k in keys}
    np.savez_compressed(os.path.join(a.out, "chains.npz"), **arr)
    _init(a)
    S = _G["S"]
    ng = arr["s_hist"].shape[2]
    M_, n = a.adapt, a.n_iter
    acc, rel = arr["acc"], arr["relerr"]
    grp = np.arange(n) % ng
    post = np.arange(n) >= M_
    acc_group = [float(acc[:, post & (grp == g)].mean()) for g in range(ng)]
    sh = arr["s_hist"][:, -1, :]
    summ = dict(settings=vars(a), spec=spec, data=_G["data"].summary(), wall_sec=time.time() - t0,
                sec_per_chain=float(np.mean([o["_sec"] for o in res])), n_groups=ng,
                post_acc_mean=float(acc[:, post].mean()), post_acc_chain_min=float(acc[:, post].mean(1).min()),
                post_acc_chain_max=float(acc[:, post].mean(1).max()),
                post_acc_group_min=min(acc_group), post_acc_group_max=max(acc_group),
                post_relerr_median_of_chain_medians=float(np.median(np.median(rel[:, M_:], 1))),
                post_relerr_last=float(np.median(rel[:, -1])),
                s_final_median_per_group=[float(x) for x in np.median(sh, 0)],
                s_final_min=float(sh.min()), s_final_max=float(sh.max()),
                s_at_floor_chains=int((sh <= a.s_min * 1.0001).any(1).sum()))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("settings", "spec")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
