"""Relative-error and acceptance-rate diagnostics of a GLAM run (no MPI, no solver).

    python experiments/plot_glam_diagnostics.py --exp-tag run1
    python experiments/plot_glam_diagnostics.py --exp-tag run1 --ar-window 200 --chains 0 1 2 3

Relative error (RE) = ||ref - sim|| / ||ref|| of the concentration field and of the pressure drop, the two terms of Phi
(Phi = RE_conc^2 / (2 sigma_conc^2) + RE_dp^2 / (2 sigma_dp^2)).
  * runs made after the RE trace was added store the RE of the current state at EVERY iteration (chain files);
  * older runs fall back to the strided samples of glam_postprocess.py --fields (post/field_stats.npz + samples.npz):
    RE of the samples from the tail only.
Acceptance: the pCN accept/reject of every iteration, split into global and local moves.  Windows of --ar-window
iterations; a window counts for a kind only if it holds at least 10 moves of that kind.

Writes figures/<study>/diagnostics/
  re_traces.png          RE of concentration and pressure drop versus iteration, one line per chain, mean dashed
  re_hist.png            histogram of RE over the tail (all chains pooled) for both components
  acceptance_traces.png  windowed acceptance versus iteration, global and local, with the adaptation target
  acceptance_hist.png    histogram of the windowed acceptance rates in the tail (pooled over chains and windows),
                         per kind, with a normal density of the same mean and sd
and tables/<study>/diagnostics.json (tail means, quantiles, per-chain acceptance).
"""
import _common  # noqa: F401
import argparse
import os
import warnings

import numpy as np

from brho import io
from brho.glam import post


def windowed_acceptance(acc, kind, which, window, min_moves=10):
    """Acceptance rate of `which` moves in consecutive windows -> (window centres, rates, nan where too few moves)."""
    n = len(acc) // window
    centres, rates = np.empty(n), np.full(n, np.nan)
    for j in range(n):
        sl = slice(j * window, (j + 1) * window)
        sel = kind[sl] == which
        centres[j] = j * window + window / 2
        if sel.sum() >= min_moves:
            rates[j] = acc[sl][sel].mean()
    return centres, rates


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--chains", type=int, nargs="+", default=None, help="chain (rank) numbers; default all")
    ap.add_argument("--ar-window", type=int, default=100)
    ap.add_argument("--tail-start", type=int, default=None, help="default: window1 from the manifest")
    ap.add_argument("--stride", type=int, default=0, help="plot every stride-th iteration of the RE traces")
    ap.add_argument("--linear", action="store_true", help="linear RE axis (default log)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")
    res = io.results_dir(exp)
    man = io.load_json(os.path.join(res, "manifest.json"))
    cfg = man["configs"]["GlamConfig"]
    P, n_iter = man["ranks"], cfg["n_iter"]
    tail = cfg["window1"] if args.tail_start is None else args.tail_start
    chains = list(range(P)) if args.chains is None else args.chains
    odir = os.path.join(io.figures_dir(exp), "diagnostics")
    os.makedirs(odir, exist_ok=True)
    cm = plt.get_cmap("tab20")
    summary = dict(chains=chains, tail_start=tail, ar_window=args.ar_window)

    def guides(ax):
        ax.axvline(cfg["window1"], color="k", ls="--", lw=0.8)
        ax.axvline(cfg["window1"] + cfg["window2"], color="k", ls="--", lw=0.8)
        ax.set_xlim(0, n_iter)

    data = {r: post.load_chain(res, r) for r in chains}
    n_it = min(min(post.n_iterations(c) for c in data.values()), n_iter)

    # ------------------------------------------------------------------------------ relative error
    have_re = all(c["re_conc"] is not None for c in data.values())
    if have_re:
        stride = args.stride or max(1, n_it // 2000)
        t = np.arange(0, n_it, stride)
        RE = {name: np.stack([data[r][key][:n_it] for r in chains]).astype(float)
              for name, key in (("conc", "re_conc"), ("dp", "re_dp"))}
        series = {name: (t, RE[name][:, t]) for name in RE}
        tailvals = {name: RE[name][:, tail:].ravel() for name in RE}
        src = "every iteration"
    else:
        fs = os.path.join(res, "post", "field_stats.npz")
        sm = os.path.join(res, "post", "samples.npz")
        if not (os.path.exists(fs) and os.path.exists(sm)):
            print("[diagnostics] no RE trace in the chain files and no post/field_stats.npz: RE figures skipped "
                  "(run glam_postprocess.py --fields, or rerun with a newer version)")
            series = None
        else:
            z, s = np.load(fs), np.load(sm)
            idx = z["sample_index"]
            rk, it = s["rank"][idx], s["iteration"][idx]
            series, tailvals = {}, {}
            for name, key in (("conc", "sample_re_conc"), ("dp", "sample_re_dp")):
                its = np.unique(it)
                M = np.full((len(chains), len(its)), np.nan)
                for ci, r in enumerate(chains):
                    m = rk == r
                    M[ci, np.searchsorted(its, it[m])] = z[key][m]
                series[name] = (its, M)
                tailvals[name] = z[key][np.isin(rk, chains)]
            src = "strided samples of the tail"
    names = {"conc": "Concentration (saturation)", "dp": "Pressure drop"}
    if series is not None:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5.2))
        for ax, name in zip(axs, ("conc", "dp")):
            tt, Y = series[name]
            for k in range(Y.shape[0]):
                ax.plot(tt, Y[k], lw=0.7, color=cm(k % 20), marker="." if not have_re else None, ms=3)
            ax.plot(tt, np.nanmean(Y, axis=0), "k--", lw=2, label=f"mean over chains (tail mean {np.nanmean(tailvals[name]):.3g})")
            ax.set_yscale("linear" if args.linear else "log")
            ax.set_xlabel("iteration"); ax.set_ylabel("relative error")
            ax.set_title(f"{names[name]}: RE versus iteration ({src})", fontsize=10.5)
            ax.grid(alpha=0.3, which="both"); ax.legend(frameon=False, fontsize=9); guides(ax)
        fig.tight_layout(); fig.savefig(os.path.join(odir, "re_traces.png"), dpi=150); plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for ax, name in zip(axs, ("conc", "dp")):
            v = tailvals[name][np.isfinite(tailvals[name])]
            ax.hist(np.log10(v), bins=40, density=True, color="#2a78d6", alpha=0.8)
            ax.set_xlabel("log10(RE)"); ax.set_ylabel("density")
            ax.set_title(f"{names[name]}: RE over the tail ({v.size} values)", fontsize=10.5); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(odir, "re_hist.png"), dpi=150); plt.close(fig)
        summary["re"] = {name: dict(source=src, mean=float(np.nanmean(tailvals[name])),
                                    quantiles={q: float(np.nanpercentile(tailvals[name], q)) for q in (5, 25, 50, 75, 95)})
                         for name in tailvals}

    # ------------------------------------------------------------------------------ acceptance
    kinds = {0: "global", 1: "local"}
    target = cfg.get("target", 0.25)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
    pooled = {0: [], 1: []}
    per_chain = {kinds[k]: {} for k in kinds}
    for ax, k in zip(axs, (0, 1)):
        allr = []
        for ci, r in enumerate(chains):
            c = data[r]
            acc, kind = c["acc"][:n_it].astype(float), c["kind"][:n_it]
            x, y = windowed_acceptance(acc, kind, k, args.ar_window)
            ax.plot(x, y, lw=0.8, color=cm(ci % 20), marker="o", ms=2.5)
            allr.append(y)
            pooled[k].append(y[x >= tail])
            sel = (kind == k) & (np.arange(n_it) >= tail)
            per_chain[kinds[k]][int(r)] = float(acc[sel].mean()) if sel.any() else None
        ax.plot(x, np.nanmean(np.stack(allr), axis=0), "k--", lw=2, label="mean over chains")
        ax.axhline(target, color="r", ls=":", lw=1.5, label=f"adaptation target {target:g}")
        ax.set_title(f"{kinds[k].capitalize()} moves: acceptance in windows of {args.ar_window} iterations", fontsize=10.5)
        ax.set_xlabel("iteration"); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9); guides(ax)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("acceptance rate")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "acceptance_traces.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    summary["acceptance"] = dict(target=target, per_chain_tail=per_chain)
    for ax, k in zip(axs, (0, 1)):
        v = np.concatenate(pooled[k]); v = v[np.isfinite(v)]
        if v.size < 2:
            ax.set_title(f"{kinds[k].capitalize()}: not enough windows in the tail"); continue
        mu, sd = float(v.mean()), float(v.std(ddof=1))
        ax.hist(v, bins=np.linspace(0, 1, 41), density=True, color="#2a78d6", alpha=0.8)
        if sd > 0:
            x = np.linspace(0, 1, 400)
            ax.plot(x, np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), "k", lw=2,
                    label=f"normal: $\\mu$={mu:.3f}, $\\sigma$={sd:.3f}")
        ax.axvline(target, color="r", ls=":", lw=1.5)
        ax.set_xlim(0, 1); ax.set_xlabel("acceptance rate"); ax.set_ylabel("density")
        ax.set_title(f"{kinds[k].capitalize()} moves, tail ({v.size} chain-windows)", fontsize=10.5)
        ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)
        summary["acceptance"][kinds[k]] = dict(mean=mu, sd=sd, windows=int(v.size))
    fig.tight_layout(); fig.savefig(os.path.join(odir, "acceptance_hist.png"), dpi=150); plt.close(fig)

    jd = os.path.join(io.tables_dir(exp), "diagnostics.json")
    os.makedirs(os.path.dirname(jd), exist_ok=True)
    with open(jd, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[diagnostics] wrote {odir} and {jd}")


if __name__ == "__main__":
    main()
