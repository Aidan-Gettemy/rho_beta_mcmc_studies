"""beta(rho) figures for the paper (no statistics text).  Reads curve_external.npz; runs no solver.

  beta_overlay.png      the fitted beta(rho) of every model, overlaid (poly2/4/6, PCHIP)
  beta_<model>.png      one per model: scatter of the single realizations at each rho, their central 90%
                        interval, and the model's median fit with its 90% bootstrap interval

beta(rho) = Kexp * exp(-h(rho)),  h = log(Keff/K0)  (so a realization with effective permeability keff
has beta_i = Kexp * K0 / keff_i).  Tables: tables/<study>/beta_curves.csv.

    python experiments/plot_beta_curves.py --curve runs/juno_external_2026-10/results/e3_external/curve_external.npz \
        --outdir runs/juno_external_2026-10/figures/e3_external
"""
import argparse
import os

import _common  # noqa: F401
import numpy as np

from brho import io
from brho.curvefit import bootstrap_fits

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e7e6e2"
SLOTS = {"poly2": "#2a78d6", "poly4": "#eb6834", "poly6": "#1baf7a", "pchip": "#eda100"}
DASH = {"poly2": "-", "poly4": (0, (6, 2.5)), "poly6": (0, (1.5, 2)), "pchip": (0, (6, 2, 1.5, 2))}
LABEL = {"poly2": "degree 2", "poly4": "degree 4", "poly6": "degree 6", "pchip": "PCHIP"}


def _style(ax, ylabel=True):
    import matplotlib.pyplot as plt  # noqa: F401
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(r"$\rho$", color=INK2, fontsize=11)
    if ylabel:
        ax.set_ylabel(r"$\beta(\rho)$  [$10^{-13}$ m$^2$]", color=INK2, fontsize=11)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curve", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--Kexp", type=float, default=1.036016e-13)
    ap.add_argument("--degrees", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--level", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unweighted", action="store_true")
    ap.add_argument("--linear", action="store_true", help="linear beta axis (default: log)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(a.curve)
    rhos, keff, K0 = z["rhos"], z["keff"], float(z["K0"])
    if not np.all(np.isfinite(keff)):
        raise SystemExit("failed solves (NaN) in the curve file; rerun the missing realizations first")
    boot = bootstrap_fits(rhos, keff, K0, a.degrees, a.n_boot, a.seed, not a.unweighted, a.level)
    S = 1e-13
    # h -> beta: high h = low beta, so the interval ends swap
    beta_i = a.Kexp * K0 / keff / S                                  # (n_rho, N) single realizations
    q = 100 * (1 - a.level) / 2
    d_lo, d_hi = np.percentile(beta_i, [q, 100 - q], axis=1)
    grid = boot["grid"]
    fit = {}
    for nm in boot["names"]:
        b = boot[nm]
        fit[nm] = tuple(a.Kexp * np.exp(-v) / S for v in (b["median"], b["hi"], b["lo"]))   # median, lo, hi
    os.makedirs(a.outdir, exist_ok=True)
    level = int(round(100 * a.level))
    scale = "linear" if a.linear else "log"

    # --- overlay of the fitted curves
    fig, ax = plt.subplots(figsize=(6.2, 4.4), facecolor=SURFACE)
    for nm in boot["names"]:
        ax.plot(grid, fit[nm][0], color=SLOTS[nm], ls=DASH[nm], lw=2, label=LABEL[nm], zorder=3)
    ax.set_yscale(scale)
    _style(ax)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "beta_overlay.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # --- one figure per model
    rng = np.random.default_rng(a.seed)
    dr = float(np.median(np.diff(rhos)))
    n = beta_i.shape[1]
    for nm in boot["names"]:
        fig, ax = plt.subplots(figsize=(6.2, 4.4), facecolor=SURFACE)
        xs = np.repeat(rhos, n) + rng.uniform(-0.38 * dr, 0.38 * dr, size=beta_i.size)
        ax.scatter(xs, beta_i.ravel(), s=1.6, color="#9a9992", alpha=0.16, linewidths=0, rasterized=True,
                   zorder=1, label="simulations")
        ax.plot(rhos, d_lo, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2, label=f"{level}% interval of simulations")
        ax.plot(rhos, d_hi, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
        med, lo, hi = fit[nm]
        ax.fill_between(grid, lo, hi, color=SLOTS[nm], alpha=0.35, lw=0, zorder=3, label=f"{level}% interval of fit")
        ax.plot(grid, med, color=SLOTS[nm], lw=2, zorder=4, label="median of fit")
        ax.set_yscale(scale)
        _style(ax)
        h, l = ax.get_legend_handles_labels()
        order = [l.index(k) for k in ("simulations", f"{level}% interval of simulations", "median of fit",
                                       f"{level}% interval of fit")]
        leg = ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=9, labelcolor=INK2,
                        loc="upper left", markerscale=6)
        for lh in leg.legend_handles[:1]:
            lh.set_alpha(0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, f"beta_{nm}.png"), dpi=200, facecolor=SURFACE)
        plt.close(fig)

    cols, data = ["rho"], [grid]
    for nm in boot["names"]:
        for k, v in zip(("median", "lo", "hi"), fit[nm]):
            cols.append(f"beta_{nm}_{k}_1e-13"); data.append(v)
    with open(os.path.join(a.outdir, "beta_curves.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(grid)):
            f.write(",".join(f"{d[i]:.8e}" for d in data) + "\n")
    print("saved to", a.outdir, "| beta range of simulations [1e-13 m2]:", float(beta_i.min()), float(beta_i.max()))


if __name__ == "__main__":
    main()
