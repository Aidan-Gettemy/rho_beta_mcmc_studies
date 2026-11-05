"""
Run the global-pCN chains of one configuration (arm, grid) of the grid-refinement study
(brho/gridstudy.py, docs/grid_arms_experiment.md).  One process per chain (multiprocessing), same chain
seeds in every configuration.  Writes to <out>/ :
    chains.npz     relerr, acc, move, phi (n_iter), theta (n_iter/save_every, 20), s_hist
    summary.json   settings, data summary, final step sizes, acceptance, relative-error statistics

    python experiments/run_grid_chains.py --arm abs --n 64 --chains 64 --n-iter 100000 --adapt 10000 \
        --out runs/grid_arms_2026-11/arm3_abs_n64
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from brho import gridstudy as G

_G = {}


def _init(args):
    S = G.GridSetup(args.n)
    _G["S"] = S
    _G["data"] = G.GridData(S, args.arm, args.eps, args.sigma_abs, args.data_seed)
    _G["a"] = args


def _one(chain):
    a = _G["a"]
    t = time.time()
    o = G.run_chain(_G["S"], _G["data"], a.seed0, chain, a.n_iter, a.adapt, a.s0, a.save_every, 0.25, a.s_min)
    o["_sec"] = time.time() - t
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=G.ARMS)
    ap.add_argument("--n", type=int, required=True, choices=G.GRIDS)
    ap.add_argument("--chains", type=int, default=64)
    ap.add_argument("--n-iter", type=int, default=100000)
    ap.add_argument("--adapt", type=int, default=10000)
    ap.add_argument("--s0", type=float, default=0.2)
    ap.add_argument("--s-min", type=float, default=1e-4, help="lower bound of the adapted pCN step size")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--eps", type=float, default=G.EPS, help="relative noise level (arms check, fixed; and the sigma_abs recipe)")
    ap.add_argument("--sigma-abs", type=float, default=None, help="arm abs: absolute noise sd (default eps * ||p_obs|| at 16 x 16)")
    ap.add_argument("--data-seed", type=int, default=G.SEED)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers or min(a.chains, os.cpu_count()), _init, (a,)) as pool:
        res = pool.map(_one, range(a.chains))
    keys = [k for k in res[0] if not k.startswith("_")]
    arr = {k: np.stack([o[k] for o in res]) for k in keys}
    np.savez_compressed(os.path.join(a.out, "chains.npz"), **arr)
    _init(a)
    n, M = a.n_iter, a.adapt
    acc, rel = arr["acc"][:, M:], arr["relerr"][:, M:]
    sh = arr["s_hist"][:, :, 0]
    summ = dict(settings=vars(a), data=_G["data"].summary(), wall_sec=time.time() - t0,
                sec_per_chain=float(np.mean([o["_sec"] for o in res])),
                post_acc_mean=float(acc.mean()), post_acc_chain_min=float(acc.mean(1).min()),
                post_acc_chain_max=float(acc.mean(1).max()),
                post_relerr_median_of_chain_medians=float(np.median(np.median(rel, 1))),
                post_relerr_chain_median_max=float(np.median(rel, 1).max()),
                post_relerr_last=float(np.median(arr["relerr"][:, -1])),
                s_final_median=float(np.median(sh[:, -1])), s_final_min=float(sh[:, -1].min()),
                s_final_max=float(sh[:, -1].max()),
                s_at_floor_chains=int((sh[:, -1] <= a.s_min * 1.0001).sum()))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "settings"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
