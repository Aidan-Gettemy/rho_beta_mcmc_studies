"""
Quantile-regression recovery of the beta(rho) curve: 5%, 50% and 95% quantiles of log(keff/K0) given rho, fitted
on ALL scatter points (n_rho x n_realizations) for the even polynomials (degrees 2, 4, 6) and for PCHIP
(node-wise quantiles interpolated, anchored at 0).  No solver is run; it reads curve_external.npz.

    python experiments/plot_quantile_curves.py --Kexp 1.036016e-13        # study e3_external
    python experiments/plot_quantile_curves.py --exp-tag pilot --taus 0.05 0.5 0.95

The band is the spread of SINGLE realizations (predictive), not the uncertainty of the mean curve.
beta = Kexp exp(-h): the 95% quantile of h is the 5% quantile of beta.

Output: figures/<study>/fit_quantile.png, tables/<study>/fit_quantile.csv (h and beta quantiles on a
400-point grid per model) and tables/<study>/fit_quantile.json (coefficients, pinball loss, coverage).
"""
import argparse
import json
import os

import _common  # noqa: F401  (first: pins threads before NumPy loads)
import numpy as np

from brho import io
from brho.quantfit import coverage, eval_quantile, quantile_fits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Kexp", type=float, default=1.036016e-13, help="measured effective permeability [m^2]")
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--degrees", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--taus", type=float, nargs=3, default=[0.05, 0.5, 0.95])
    ap.add_argument("--curve", help="curve_external.npz (default: the study's results folder)")
    ap.add_argument("--out", help="figure path (default: figures/<study>/fit_quantile.png)")
    args = ap.parse_args()

    exp = io.EXP_CURVE_EXTERNAL + (f"_{args.exp_tag}" if args.exp_tag else "")
    z = np.load(args.curve or os.path.join(io.results_dir(exp), "curve_external.npz"))
    rhos, keff, K0 = z["rhos"], z["keff"], float(z["K0"])
    if not np.all(np.isfinite(keff)):
        raise SystemExit("keff contains NaN: rerun the missing realizations first")
    taus = tuple(args.taus)
    fit = quantile_fits(rhos, keff, K0, args.degrees, taus)
    names = list(fit["models"])
    cov = {nm: coverage(fit, keff, K0, nm) for nm in names}
    for nm in names:
        print(nm, "coverage", {t: round(c, 4) for t, c in cov[nm].items()},
              "pinball", {t: round(v, 1) for t, v in fit["loss"][nm].items()})

    grid = np.linspace(rhos.min(), rhos.max(), 400)
    Q = {nm: {t: eval_quantile(fit, nm, t, grid) for t in taus} for nm in names}

    fig = args.out or os.path.join(io.figures_dir(exp), "fit_quantile.png")
    os.makedirs(os.path.dirname(fig), exist_ok=True)
    make_figure(rhos, keff, K0, args.Kexp, grid, Q, taus, names, cov, fit, fig)
    print("saved", fig)
    fdir = os.path.dirname(fig)
    print("saved", make_overlay(rhos, keff, K0, args.Kexp, grid, Q, taus, names, os.path.join(fdir, "quantile_beta_overlay.png")))
    for nm in names:
        print("saved", make_beta_panel(rhos, keff, K0, args.Kexp, grid, Q, taus, nm,
                                       os.path.join(fdir, f"quantile_beta_{nm}.png")))

    base = os.path.splitext(fig)[0] if args.out else os.path.join(io.tables_dir(exp), "fit_quantile")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    lo, mid, hi = taus
    cols, data = ["rho"], [grid]
    for nm in names:
        for t, key in zip(taus, ("lo", "median", "hi")):
            cols.append(f"h_{nm}_{key}"); data.append(Q[nm][t])
        cols += [f"beta_{nm}_median", f"beta_{nm}_lo", f"beta_{nm}_hi"]
        data += [args.Kexp * np.exp(-Q[nm][mid]), args.Kexp * np.exp(-Q[nm][hi]), args.Kexp * np.exp(-Q[nm][lo])]
    with open(base + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(grid)):
            f.write(",".join(f"{d[i]:.8e}" for d in data) + "\n")
    summary = dict(taus=list(taus), Kexp=args.Kexp, n_rho=len(rhos), n_realizations=int(keff.shape[1]),
                   coverage={nm: {str(t): c for t, c in cov[nm].items()} for nm in names},
                   pinball_loss={nm: {str(t): v for t, v in fit["loss"][nm].items()} for nm in names},
                   poly_coefficients={nm: {str(t): fit["models"][nm][t]["coef"] for t in taus}
                                      for nm in names if nm != "pchip"},
                   node_quantiles={str(t): fit["node_quantiles"][t].tolist() for t in taus})
    with open(base + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("saved", base + ".csv", base + ".json")


_STYLE = dict(ink="#0b0b0b", ink2="#52514e", surface="#fcfcfb", grid="#e7e6e2", dots="#9a9992")
_COLORS = {"poly2": "#2a78d6", "poly4": "#e07b1f", "poly6": "#2f9e5b", "pchip": "#8a4fc7"}
_UNIT = 1.0e-13


def _style_axes(a):
    st = _STYLE
    a.set_facecolor(st["surface"])
    a.grid(color=st["grid"], lw=0.6, zorder=0, which="both")
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(st["ink2"])
    a.tick_params(colors=st["ink2"], labelsize=9)


def make_overlay(rhos, keff, K0, Kexp, grid, Q, taus, names, path):
    """50% quantile beta(rho) of every model on one axis; shaded 90% region of the data scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = _STYLE
    lo, mid, hi = taus
    logk = np.log(keff / K0)
    q_lo, q_hi = np.quantile(logk, [lo, hi], axis=1)          # data quantiles at the rho nodes
    fig, a = plt.subplots(figsize=(7.6, 5.4), facecolor=st["surface"])
    _style_axes(a)
    # beta = Kexp exp(-h): reversed interval
    a.fill_between(rhos, Kexp * np.exp(-q_hi) / _UNIT, Kexp * np.exp(-q_lo) / _UNIT, color=st["dots"], alpha=0.35, lw=0,
                   zorder=1, label=f"{round(100 * (hi - lo))}% of the data scatter ({round(lo * 100)}-{round(hi * 100)}%)")
    labels = {"pchip": "PCHIP"}
    for nm in names:
        a.plot(grid, Kexp * np.exp(-Q[nm][mid]) / _UNIT, color=_COLORS.get(nm, "k"), lw=2.0, zorder=3,
               label=labels.get(nm, f"degree {nm[4:]}") + f" ({round(mid * 100)}%)")
    a.set_yscale("log")
    a.set_xlabel(r"$\rho$", color=st["ink2"])
    a.set_ylabel(r"$\beta(\rho)$  [$10^{-13}$ m$^2$]", color=st["ink2"])
    a.set_title(f"Median beta(rho) of each model against the {round(100 * (hi - lo))}% data spread",
                fontsize=11, color=st["ink"], loc="left")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=st["surface"])
    plt.close(fig)
    return path


def make_beta_panel(rhos, keff, K0, Kexp, grid, Q, taus, nm, path, seed=0):
    """beta(rho) of one model: scatter of all points and the model's 5/50/95% quantile curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = _STYLE
    lo, mid, hi = taus
    n = keff.shape[1]
    rng = np.random.default_rng(seed)
    dr = float(np.median(np.diff(rhos)))
    beta_pts = Kexp * np.exp(-np.log(keff / K0)) / _UNIT
    fig, a = plt.subplots(figsize=(7.6, 5.4), facecolor=st["surface"])
    _style_axes(a)
    xs = np.repeat(rhos, n) + rng.uniform(-0.38 * dr, 0.38 * dr, size=beta_pts.size)
    a.scatter(xs, beta_pts.ravel(), s=1.8, color=st["dots"], alpha=0.18, linewidths=0, rasterized=True, zorder=1,
              label="realizations")
    c = _COLORS.get(nm, "k")
    b = lambda t: Kexp * np.exp(-Q[nm][t]) / _UNIT  # noqa: E731
    a.fill_between(grid, b(hi), b(lo), color=c, alpha=0.18, lw=0, zorder=2)
    a.plot(grid, b(lo), color=c, lw=1.2, ls=(0, (4, 3)), zorder=3, label=f"{round(lo * 100)}% / {round(hi * 100)}% quantile")
    a.plot(grid, b(hi), color=c, lw=1.2, ls=(0, (4, 3)), zorder=3)
    a.plot(grid, b(mid), color=c, lw=2.4, zorder=4, label=f"{round(mid * 100)}% quantile")
    a.set_yscale("log")
    a.set_xlabel(r"$\rho$", color=st["ink2"])
    a.set_ylabel(r"$\beta(\rho)$  [$10^{-13}$ m$^2$]", color=st["ink2"])
    title = "PCHIP" if nm == "pchip" else f"Degree {nm[4:]} polynomial"
    a.set_title(f"{title}: quantile curves of beta(rho) over all {beta_pts.size:,} points", fontsize=11,
                color=st["ink"], loc="left")
    leg = a.legend(frameon=False, fontsize=9, loc="upper left", markerscale=4)
    for h in leg.legend_handles[:1]:
        h.set_alpha(1)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=st["surface"])
    plt.close(fig)
    return path


def make_figure(rhos, keff, K0, Kexp, grid, Q, taus, names, cov, fit, path, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, surface, grid_c = "#0b0b0b", "#52514e", "#fcfcfb", "#e7e6e2"
    accent, dots = "#2a78d6", "#9a9992"
    logk = np.log(keff / K0)
    n = logk.shape[1]
    rng = np.random.default_rng(seed)
    dr = float(np.median(np.diff(rhos)))
    lo, mid, hi = taus
    titles = {"pchip": "PCHIP (through the node quantiles)"}
    for nm in names:
        if nm.startswith("poly"):
            titles[nm] = f"Degree {nm[4:]} polynomial"
    ncol = len(names)
    fig, ax = plt.subplots(2, ncol, figsize=(4.1 * ncol, 7.6), sharex=True, facecolor=surface, squeeze=False)
    ymin, ymax = float(logk.min()), float(logk.max())
    pad = 0.04 * (ymax - ymin)
    for c, nm in enumerate(names):
        for row in (0, 1):
            a = ax[row, c]
            a.set_facecolor(surface)
            sgn = 1.0 if row == 0 else -1.0
            xs = np.repeat(rhos, n) + rng.uniform(-0.38 * dr, 0.38 * dr, size=logk.size)
            yv = logk.ravel() if row == 0 else np.exp(-logk.ravel())
            a.scatter(xs, yv, s=1.6, color=dots, alpha=0.16, linewidths=0, rasterized=True, zorder=1)
            if row == 0:
                a.fill_between(grid, Q[nm][lo], Q[nm][hi], color=accent, alpha=0.28, lw=0, zorder=3)
                a.plot(grid, Q[nm][mid], color=accent, lw=2, zorder=4)
                a.plot(grid, Q[nm][lo], color=accent, lw=1.0, zorder=4)
                a.plot(grid, Q[nm][hi], color=accent, lw=1.0, zorder=4)
                a.set_ylim(ymin - pad, ymax + pad)
            else:
                a.fill_between(grid, np.exp(-Q[nm][hi]), np.exp(-Q[nm][lo]), color=accent, alpha=0.28, lw=0, zorder=3)
                a.plot(grid, np.exp(-Q[nm][mid]), color=accent, lw=2, zorder=4)
                a.plot(grid, np.exp(-Q[nm][lo]), color=accent, lw=1.0, zorder=4)
                a.plot(grid, np.exp(-Q[nm][hi]), color=accent, lw=1.0, zorder=4)
                a.set_yscale("log")
        sub = "pinball loss " + " / ".join(f"{fit['loss'][nm][t]:.0f}" for t in taus)
        sub += "    below: " + " / ".join(f"{100 * cov[nm][t]:.1f}%" for t in taus)
        ax[0, c].set_title(titles[nm], fontsize=10.5, color=ink, loc="left", pad=22)
        ax[0, c].text(0, 1.015, sub, transform=ax[0, c].transAxes, fontsize=8, color=ink2, va="bottom")
        ax[1, c].set_xlabel(r"$\rho$", color=ink2)
        if c == 0:
            ax[0, c].set_ylabel(r"$\log(K_{eff}/K_0)$", color=ink2)
            ax[1, c].set_ylabel(r"$\beta/K_{exp} = e^{-h}$", color=ink2, fontsize=8.5)
        else:
            ax[0, c].tick_params(labelleft=False)
    for a in ax.ravel():
        a.grid(color=grid_c, lw=0.6, zorder=0)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(ink2)
        a.tick_params(colors=ink2, labelsize=8.5)
    fig.suptitle(f"Quantile regression on all {logk.size:,} points: {int(lo*100)}% / {int(mid*100)}% / {int(hi*100)}% "
                 "(band = spread of single realizations)", fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
