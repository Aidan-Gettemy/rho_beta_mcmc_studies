#!/usr/bin/env python3
"""dd_end field figure: posterior mean, sd and absolute error of log k for one family of runs (contourf, jet).

    python experiments/dd_end_fields.py [--root DIR] [--truth 2026]

One figure per family, fields_ell<ell>_<p|pf>_b<blocks>.png: columns truth, global pCN, global in the MSM basis, DD primal, DD dual, pooled;
rows posterior mean (colour scale shared with the truth), posterior sd and |mean - truth| (scales shared along the row).
Reads post/fields.npz and post_pooled/fields.npz of the runs; no raw chains."""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "runs/msm_local_kle_2026-11/dd_end"
LABEL = {(0.2, False): "a", (0.2, True): "b", (0.1, False): "c", (0.1, True): "d", (0.05, False): "e", (0.05, True): "f"}
FAM = {(0.2, False): "g", (0.2, True): "h", (0.1, False): "i", (0.1, True): "j", (0.05, False): "k", (0.05, True): "l"}
NB = {2: 4, 4: 16, 8: 64}          # blocks per side -> number in the labels


def load(path, prefix):
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["logk_true"], z[f"{prefix}_logk_mean"], z[f"{prefix}_logk_sd"]


def figure(ell, flux, nb, truth, root, out):
    p = "pf" if flux else "p"
    d = f"{root}/l{ell}/{p}"
    g, f, k = LABEL[(ell, flux)], FAM[(ell, flux)], NB[nb]
    cols = [("global pCN", f"{d}/{g}_t{truth}/post/fields.npz", "run"),
            (f"MSM basis {nb}x{nb}", f"{d}/{f}-{k}_t{truth}/post/fields.npz", "run"),
            (f"DD primal {nb}x{nb}", f"{d}/{f}-{k}-DD_t{truth}/post/fields.npz", "primal"),
            (f"DD dual {nb + 1}x{nb + 1}", f"{d}/{f}-{k}-DD+_t{truth}/post/fields.npz", "dual"),
            ("pooled", f"{d}/{f}-{k}-DD+_t{truth}/post_pooled/fields.npz", "pooled")]
    data = [(t, load(pth, pre)) for t, pth, pre in cols]
    data = [(t, v) for t, v in data if v is not None]
    if len(data) < 2:                     # only the global reference exists: nothing to compare yet
        return None
    truth_f = data[0][1][0]
    n = int(round(np.sqrt(truth_f.size)))
    fl = lambda a: np.asarray(a).reshape(n, n)
    vmin, vmax = float(truth_f.min()), float(truth_f.max())
    sdmax = max(float(v[2].max()) for _, v in data)
    ermax = max(float(np.abs(v[1] - v[0]).max()) for _, v in data)
    ncol = len(data) + 1
    fig, ax = plt.subplots(3, ncol, figsize=(2.6 * ncol, 8), squeeze=False)
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x)
    lev = lambda lo, hi: np.linspace(lo, hi, 21)
    c0 = ax[0, 0].contourf(X, Y, fl(truth_f), levels=lev(vmin, vmax), cmap="jet")
    ax[0, 0].set_title("truth")
    ax[1, 0].axis("off"); ax[2, 0].axis("off")
    for j, (t, (tr, m, s)) in enumerate(data, start=1):
        ax[0, j].contourf(X, Y, fl(m), levels=lev(vmin, vmax), cmap="jet")
        ax[0, j].set_title(t, fontsize=9)
        c1 = ax[1, j].contourf(X, Y, fl(s), levels=lev(0, sdmax), cmap="jet")
        e = np.abs(m - tr)
        c2 = ax[2, j].contourf(X, Y, fl(e), levels=lev(0, ermax), cmap="jet")
        ax[1, j].set_xlabel(f"mean sd {s.mean():.2f}", fontsize=8)
        ax[2, j].set_xlabel(f"rms err {np.sqrt(np.mean((m - tr) ** 2)):.2f}", fontsize=8)
    for a in ax.ravel():
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(c0, ax=ax[0, :], shrink=0.8, label="log k")
    fig.colorbar(c1, ax=ax[1, :], shrink=0.8, label="posterior sd")
    fig.colorbar(c2, ax=ax[2, :], shrink=0.8, label="|mean - truth|")
    fig.suptitle(f"ell {ell}, {'pressure + flux' if flux else 'pressure only'}, {nb}x{nb} blocks, truth {truth}")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--truth", type=int, default=2026)
    ap.add_argument("--only", default=None, help="comma list of ell:flux:nb, e.g. 0.1:1:4 (default: every family found)")
    a = ap.parse_args()
    fams = ([tuple(s.split(":")) for s in a.only.split(",")] if a.only else
            [(ell, fx, nb) for ell in ("0.2", "0.1", "0.05") for fx in ("0", "1") for nb in ("2", "4", "8")])
    os.makedirs(f"{a.root}/fields", exist_ok=True)
    for ell, fx, nb in fams:
        o = figure(float(ell), bool(int(fx)), int(nb), a.truth, a.root, f"{a.root}/fields/fields_ell{ell}_{'pf' if int(fx) else 'p'}_b{nb}.png")
        if o:
            print(o)


if __name__ == "__main__":
    main()
