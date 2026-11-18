#!/usr/bin/env python3
"""Stage 9 short test (global pCN on the 3 x 1 strip): forward-model checks, then a few chains per correlation length.
Reports the pCN acceptance, MPSRF vs iteration, the posterior-mean relative L2 error of the permeability a = exp(log a) (the paper's
metric: ||mean - truth|| / ||truth||) and the posterior sd of log a.   python experiments/strip_global_test.py --L 2 1 0.5 --chains 8 --iters 20000"""
import argparse, os, sys, time, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from brho.strip import StripProblem, run_pcn
from brho.mpsrf import get_mpsrf

def work(args):
    L, k, n_iter, n_adapt = args
    pb = StripProblem(L)
    return run_pcn(pb, [9, int(L * 10), k], n_iter, n_adapt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, nargs="+", default=[2, 1, 0.5])
    ap.add_argument("--chains", type=int, default=8); ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--adapt", type=int, default=4000); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for L in a.L:
        t = time.time(); pb = StripProblem(L)
        with Pool(min(a.chains, os.cpu_count())) as p:
            R = p.map(work, [(L, k, a.iters, a.adapt) for k in range(a.chains)])
        th = np.stack([r[0] for r in R]).astype(float)                 # (m, n_save, r), save_every 10
        acc = [r[1] for r in R]; step = [r[2] for r in R]
        n = th.shape[1]; grid = np.unique(np.round(np.geomspace(20, n, 25)).astype(int))
        curve = [(int(g) * 10, get_mpsrf(th[:, :g])[0]) for g in grid]
        conv = next((it for it, v in curve if v < 1.2), None)
        post = th[:, a.adapt // 10:].reshape(-1, pb.r)                 # post-adaptation samples
        logk = post @ pb.T.T
        mean_a = np.exp(logk).mean(0); a_true = np.exp(pb.logk_true)
        err = np.linalg.norm(mean_a - a_true) / np.linalg.norm(a_true)
        err_logmean = np.linalg.norm(logk.mean(0) - pb.logk_true) / np.linalg.norm(pb.logk_true)
        sd = logk.std(0)
        res[L] = dict(r=pb.r, energy=pb.energy, sigma=pb.sigma, acc=float(np.mean(acc)), step=float(np.mean(step)), mpsrf_final=curve[-1][1],
                      conv_iter=conv, err_a=float(err), err_logk_mean=float(err_logmean), sd_logk_mean=float(sd.mean()),
                      prior_sd_logk=float(np.sqrt((pb.T ** 2).sum(1)).mean()), truth_in_90=float(np.mean(np.abs(logk.mean(0) - pb.logk_true) < 1.645 * sd)),
                      seconds=time.time() - t)
        print(L, json.dumps(res[L], indent=None), flush=True)
        print("   MPSRF curve:", " ".join(f"{it}:{v:.2f}" for it, v in curve[::3]), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
