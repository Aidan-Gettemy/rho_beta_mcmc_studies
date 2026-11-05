"""
Prior test of the paper-MSM study: likelihood OFF (every proposal accepted, step size fixed), so the chain samples the
prior induced by each field construction.  For every experiment and step size s it reports, against the target
Gaussian prior of the global KLE (covariance C = T T^T, unit variance up to truncation):
  - mean pointwise variance of the field in the interface band / interior (target: the C diagonal there),
  - mean correlation of the pairs of cells that face each other ACROSS the interfaces, versus the same pairs in C,
  - the same two numbers for the neighbouring-cell correlation inside subdomains.
The step-size dependence (s = 0.2 and 0.7) shows the state dependence of the one-subdomain-at-a-time smoothing.

    python experiments/run_paper_prior.py --out runs/paper_msm_2026-11/prior_test [--chains 16 --n-iter 40000]
"""
import _common  # noqa: F401
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brho import msmpaper as M  # noqa: E402

_G = {}


def _init():
    _G["S"] = M.PaperSetup()
    _G["data"] = None


def _one(job):
    exp, s, chain, n_iter, burn = job
    S = _G["S"]
    o = M.run_chain(S, None, exp, 101, chain, n_iter, 0, s, save_every=10, field_every=10, prior=True)
    E = o["eta_thin"][burn // 10:].astype(float)
    return E.sum(0), E.T @ E, len(E)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chains", type=int, default=16)
    ap.add_argument("--n-iter", type=int, default=40000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--steps", type=float, nargs="+", default=[0.2, 0.7])
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    S = M.PaperSetup()
    C = S.T @ S.T.T
    P = S.points
    # pairs of cells facing each other across an interface (grid neighbours in different subdomains)
    n = S.n
    pairs_x, pairs_in = [], []
    for i in range(n):
        for j in range(n):
            k = i * n + j
            for di, dj in ((0, 1), (1, 0)):
                ii, jj = i + di, j + dj
                if ii < n and jj < n:
                    k2 = ii * n + jj
                    (pairs_x if S.sub_of_cell[k] != S.sub_of_cell[k2] else pairs_in).append((k, k2))
    pairs_x, pairs_in = np.array(pairs_x), np.array(pairs_in)

    def pair_corr(Cm, pr):
        d = np.diag(Cm)
        return float(np.mean(Cm[pr[:, 0], pr[:, 1]] / np.sqrt(d[pr[:, 0]] * d[pr[:, 1]])))
    res = dict(target=dict(var_band=float(np.diag(C)[S.band].mean()), var_interior=float(np.diag(C)[~S.band].mean()),
                           corr_across=pair_corr(C, pairs_x), corr_within=pair_corr(C, pairs_in),
                           n_pairs_across=len(pairs_x)), runs={})
    jobs = [(e, s, ch, a.n_iter, a.burn) for e in M.EXPS for s in a.steps for ch in range(a.chains)]
    with Pool(a.workers or min(len(jobs), os.cpu_count()), _init) as pool:
        out = pool.map(_one, jobs, chunksize=1)
    fig, axs = plt.subplots(len(a.steps), len(M.EXPS), figsize=(2.0 * len(M.EXPS), 2.1 * len(a.steps)), squeeze=False)
    k = 0
    for e in M.EXPS:
        for si, s in enumerate(a.steps):
            part = out[k:k + a.chains]; k += a.chains
            nn = sum(p[2] for p in part)
            mu = sum(p[0] for p in part) / nn
            Cm = sum(p[1] for p in part) / nn - np.outer(mu, mu)
            v = np.diag(Cm)
            res["runs"][f"{e}@s={s}"] = dict(var_band=float(v[S.band].mean()), var_interior=float(v[~S.band].mean()),
                                              corr_across=pair_corr(Cm, pairs_x), corr_within=pair_corr(Cm, pairs_in),
                                              n_samples=int(nn))
            ax = axs[si, list(M.EXPS).index(e)]
            ax.imshow(v.reshape(n, n), origin="lower", vmin=0.4, vmax=1.1, cmap="viridis")
            ax.set_title(f"{e[:2]} s={s}", fontsize=8); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "prior_variance.png"), dpi=130)
    json.dump(res, open(os.path.join(a.out, "prior_summary.json"), "w"), indent=1)
    for key, v in res["runs"].items():
        print(f"{key:24s} var band {v['var_band']:.3f} int {v['var_interior']:.3f} | corr across {v['corr_across']:.3f} within {v['corr_within']:.3f}")
    print("target", res["target"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
