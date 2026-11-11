"""Stage 0 of the subspace-transplant study: which coefficients do the data constrain?

    python experiments/coef_variance_by_index.py --exp-tag pcn --start-iteration 14000          # real run, results/e6_glam_pcn
    python experiments/coef_variance_by_index.py --out-dir <dir with chains/> --n-sub 12 --start-iteration 12000   # any run

The state is n_sub blocks of m whitened coefficients plus the whitened rho.  The prior variance of every coordinate is 1, so over the
tail of independent chains a coordinate whose pooled variance stays near 1 is not constrained by the data.  Per block the script
smooths the variance over the coefficient index, finds where it first reaches --thr (default 0.8), adds a safety margin and writes
a mask: coordinates from that index on are 'keep own' at a restart (the chain keeps its own value), the first ones are installed
from the anchor.  The rho coordinate is always installed from the anchor.

Writes  tables/<study>/coef_variance_by_index.csv   (block, index, variance, smoothed, mean)
        tables/<study>/keep_mask.npz                (m0 .. m<n_sub>: True = keep own)       -> run_glam.py --restart-keep-mask
        figures/<study>/coef_variance_by_index.png
Use chains that have NOT been restarted from shared anchors (plain pCN, or the sample after exchanges stopped): after a GLAM restart the
unconstrained coordinates are copies of the anchor's and their variance collapses.
"""
import _common  # noqa: F401
import argparse
import os

import numpy as np

from brho import io
from brho.glam.post import load_chain, n_iterations


def tail_moments(out, nranks, start, stride, n_sub):
    """Pooled per-coordinate mean and variance over all chains of the samples from `start` on (every `stride`-th iteration)."""
    s1 = s2 = None
    n = 0
    chain_means = []
    for r in range(nranks):
        ch = load_chain(out, r)
        N = n_iterations(ch)
        # iteration -> record index, as in expand(): record j holds iterations start[j] .. start[j] + count[j] - 1
        idx = np.empty(N, dtype=np.int64)
        for j, (s, c) in enumerate(zip(ch["start"], ch["count"])):
            idx[s:s + c] = j
        sel = idx[start::stride]
        X = np.hstack([ch["theta"][sel], ch["w"][sel][:, None]])
        if s1 is None:
            s1 = np.zeros(X.shape[1]); s2 = np.zeros(X.shape[1])
        s1 += X.sum(0); s2 += (X ** 2).sum(0); n += len(X)
        chain_means.append(X.mean(0))
        del ch, X
    mean = s1 / n
    var = s2 / n - mean ** 2
    between = np.var(np.asarray(chain_means), axis=0, ddof=1)
    return mean, var, between


def propose_mask(var, n_sub, m, thr=0.8, smooth=15, margin=1.5):
    """Boolean keep-own mask per block (True from the first index where the smoothed variance reaches thr, times margin) and the cut-offs."""
    masks, cuts = [], []
    k = np.ones(smooth) / smooth
    for b in range(n_sub):
        v = var[b * m:(b + 1) * m]
        sm = np.convolve(np.pad(v, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), k, mode="valid")
        above = np.nonzero(sm >= thr)[0]
        cut = int(above[0]) if len(above) else m
        cut = min(m, int(np.ceil(cut * margin)))
        mask = np.zeros(m, dtype=bool); mask[cut:] = True
        masks.append(mask); cuts.append(cut)
    masks.append(np.zeros(1, dtype=bool))                      # rho: always installed from the anchor
    return masks, cuts


def save_outputs(tables, figures, var, mean, between, masks, cuts, n_sub, m, smooth, title=""):
    os.makedirs(tables, exist_ok=True); os.makedirs(figures, exist_ok=True)
    rows = []
    k = np.ones(smooth) / smooth
    for b in range(n_sub):
        v = var[b * m:(b + 1) * m]
        sm = np.convolve(np.pad(v, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), k, mode="valid")
        for j in range(m):
            rows.append(dict(block=b, index=j, variance=v[j], smoothed=sm[j], mean=mean[b * m + j], between_chain_var=between[b * m + j]))
    _write_csv(os.path.join(tables, "coef_variance_by_index.csv"), rows)
    np.savez(os.path.join(tables, "keep_mask.npz"), **{f"m{b}": mk for b, mk in enumerate(masks)})
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    V = var[:n_sub * m].reshape(n_sub, m)
    for b in range(n_sub):
        ax[0].plot(V[b], lw=0.7, alpha=0.7)
    ax[0].axhline(1.0, c="k", ls=":"); ax[0].set_xlabel("coefficient index within block"); ax[0].set_ylabel("pooled tail variance (prior = 1)")
    ax[0].set_title(title)
    ax[1].bar(range(n_sub), cuts, color="tab:blue"); ax[1].set_xlabel("block"); ax[1].set_ylabel(f"installed from the anchor: first K of {m}")
    ax[1].set_title(f"mean K = {np.mean(cuts):.0f}; {100 * (1 - np.mean(cuts) / m):.0f}% of coefficients kept per chain")
    fig.tight_layout(); fig.savefig(os.path.join(figures, "coef_variance_by_index.png"), dpi=150); plt.close(fig)


def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--root", default=None, help="BRHO_ROOT (default: $BRHO_ROOT, else runs/cluster if it exists)")
    ap.add_argument("--out-dir", default=None, help="folder with chains/ (default results/e6_glam[_tag])")
    ap.add_argument("--n-sub", type=int, default=12)
    ap.add_argument("--chains", type=int, default=24)
    ap.add_argument("--start-iteration", type=int, default=14000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--smooth", type=int, default=15)
    ap.add_argument("--margin", type=float, default=1.5)
    args = ap.parse_args()
    if args.root:
        os.environ["BRHO_ROOT"] = args.root
    elif "BRHO_ROOT" not in os.environ and os.path.isdir(os.path.join(io.REPO, "runs", "cluster", "results")):
        os.environ["BRHO_ROOT"] = os.path.join(io.REPO, "runs", "cluster")
    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")
    out = args.out_dir or io.results_dir(exp)
    mean, var, between = tail_moments(out, args.chains, args.start_iteration, args.stride, args.n_sub)
    m = (len(var) - 1) // args.n_sub
    masks, cuts = propose_mask(var, args.n_sub, m, args.thr, args.smooth, args.margin)
    tables, figures = io.tables_dir(exp), io.figures_dir(exp)
    save_outputs(tables, figures, var, mean, between, masks, cuts, args.n_sub, m, args.smooth, title=f"{exp}, tail from {args.start_iteration}")
    print(f"{exp}: {len(var)} coordinates, m = {m} per block; pooled variance: median {np.median(var[:-1]):.3f}, "
          f"fraction above {args.thr}: {np.mean(var[:-1] >= args.thr):.2f}; rho variance {var[-1]:.3f}")
    print("cut-offs (first K coefficients installed from the anchor), per block:", cuts)
    print("written:", tables, "and", figures)


if __name__ == "__main__":
    main()
