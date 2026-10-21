"""Trace plots of a GLAM run: theta entries versus iteration for several chains (the main diagnostic).

  python experiments/plot_glam_traces.py --exp-tag run1                          # defaults: 2 entries, 8 chains
  python experiments/plot_glam_traces.py --exp-tag run1 --entries 0:0 3:5 --chains all --stride 20

--entries  block:index pairs; block 0..n_sub-1 are the subdomain thetas, block n_sub is the rho coordinate
           (w, log rho = mu + s w); index is the entry inside the block.  The rho panel shows rho itself.
--chains   'all', or a list of chain (rank) numbers; default: the first --max-chains.
Writes figures/e6_glam_<tag>/traces_theta_<block>_<index>.png (one figure per entry, one line per chain),
traces_overview.png (the entries on stacked panels with Phi and rho), and
traces_phi.png.  Dashed vertical lines: end of window 1 (step sizes frozen) and window1 + window2; dotted grey
lines: the anchor exchanges.  Chain files hold the full state at every change, rejected moves as repeats.
"""
import _common  # noqa: F401
import os
import pickle

import numpy as np

from brho import io
from brho.glam import post


def expand(ch, n_iter, stride):
    t = np.arange(0, n_iter, stride)
    j = np.searchsorted(ch["start"], t, side="right") - 1
    return t, j


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--entries", nargs="+", default=["0:0", "1:0"])
    ap.add_argument("--chains", nargs="+", default=None)
    ap.add_argument("--max-chains", type=int, default=8)
    ap.add_argument("--stride", type=int, default=0, help="plot every stride-th iteration (default: ~2000 points)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp = io.EXP_GLAM + (f"_{a.exp_tag}" if a.exp_tag else "")
    res = io.results_dir(exp)
    man = io.load_json(os.path.join(res, "manifest.json"))
    cfg = man["configs"]["GlamConfig"]
    P, n_iter = man["ranks"], cfg["n_iter"]
    if a.chains is None:
        chains = list(range(min(P, a.max_chains)))
    elif a.chains == ["all"]:
        chains = list(range(P))
    else:
        chains = [int(c) for c in a.chains]
    stride = a.stride or max(1, n_iter // 2000)
    out = a.out or io.figures_dir(exp)
    os.makedirs(out, exist_ok=True)
    ex = []
    pk = os.path.join(res, "exchanges.pkl")
    if os.path.exists(pk):
        ex = [e["iteration"] for e in pickle.load(open(pk, "rb"))]
    data = {}
    for r in chains:
        ch = post.load_chain(res, r)
        n = min(n_iter, post.n_iterations(ch))
        t, j = expand(ch, n, stride)
        data[r] = (t, ch["theta"][j], ch["w"][j], ch["phi"][j])
    any_theta = next(iter(data.values()))[1]
    ndim = any_theta.shape[1]
    bdp = os.path.join(res, "block_dims.json")
    if os.path.exists(bdp):
        dims = io.load_json(bdp)["block_dims"]
    else:                                   # older runs: equal blocks
        n_sub = man.get("args", {}).get("toy_n_sub", 12)
        dims = [ndim // n_sub] * n_sub
    if sum(dims) != ndim:
        raise SystemExit(f"block sizes {dims} do not add up to the stored theta length {ndim}")
    n_sub = len(dims)
    offs = np.concatenate([[0], np.cumsum(dims)])

    def column(block, idx):
        if idx >= dims[block]:
            raise SystemExit(f"block {block} has only {dims[block]} entries")
        return offs[block] + idx

    def guides(ax):
        ax.axvline(cfg["window1"], color="k", ls="--", lw=0.8)
        ax.axvline(cfg["window1"] + cfg["window2"], color="k", ls="--", lw=0.8)
        for e in ex:
            ax.axvline(e, color="0.8", ls=":", lw=0.5, zorder=0)
        ax.set_xlim(0, n_iter)

    def series(r, ent):
        t, th, w, ph = data[r]
        b, i = (int(x) for x in ent.split(":"))
        if b >= n_sub:
            return t, np.exp(cfg["mu_logrho"] + cfg["s_logrho"] * w), r"$\rho$"
        return t, th[:, column(b, i)], rf"$\theta_{{{b},{i}}}$"

    cm = plt.get_cmap("tab20")
    for ent in a.entries:
        fig, ax = plt.subplots(figsize=(9, 3.4))
        for k, r in enumerate(chains):
            t, y, lab = series(r, ent)
            ax.plot(t, y, lw=0.7, color=cm(k % 20), label=f"chain {r}")
        guides(ax); ax.set_xlabel("iteration"); ax.set_ylabel(lab)
        if len(chains) <= 10:
            ax.legend(ncol=5, fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(out, "traces_theta_" + ent.replace(":", "_") + ".png"), dpi=150); plt.close(fig)

    has_rho = any(int(e.split(":")[0]) >= n_sub for e in a.entries)
    npan = len(a.entries) + (1 if has_rho else 2)
    fig, axs = plt.subplots(npan, 1, figsize=(9, 2.2 * npan), sharex=True)
    for ax, ent in zip(axs, a.entries):
        for k, r in enumerate(chains):
            t, y, lab = series(r, ent)
            ax.plot(t, y, lw=0.6, color=cm(k % 20))
        ax.set_ylabel(lab); guides(ax)
    for k, r in enumerate(chains):
        t, th, w, ph = data[r]
        if not has_rho:
            axs[-2].plot(t, np.exp(cfg["mu_logrho"] + cfg["s_logrho"] * w), lw=0.6, color=cm(k % 20))
        axs[-1].plot(t, ph, lw=0.6, color=cm(k % 20))
    if not has_rho:
        axs[-2].set_ylabel(r"$\rho$"); axs[-2].set_yscale("log"); guides(axs[-2])
    axs[-1].set_ylabel(r"$\Phi$"); axs[-1].set_yscale("log"); guides(axs[-1]); axs[-1].set_xlabel("iteration")
    fig.tight_layout(); fig.savefig(os.path.join(out, "traces_overview.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    for k, r in enumerate(chains):
        ax.plot(data[r][0], data[r][3], lw=0.6, color=cm(k % 20))
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel(r"$\Phi$"); guides(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out, "traces_phi.png"), dpi=150); plt.close(fig)
    print(f"[glam-traces] {len(chains)} chains, stride {stride}, wrote {out}")


if __name__ == "__main__":
    main()
