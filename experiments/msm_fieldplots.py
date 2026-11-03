"""
Field plots of the MSM study in the group's contourf style (filled contours, jet).
The colormaps are module-level defaults, changed in one place or per call:
    CMAP_MAIN  fields, means, pressure       (default "jet")
    CMAP_STD   standard-deviation maps       (default "jet"; undecided, change here)
    CMAP_DIFF  signed differences            (default "RdBu_r")
    LEVELS     number of contour levels      (default 20)
Fields are cell-centred (index m = ix * n + iy) and drawn on the unit square.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CMAP_MAIN, CMAP_STD, CMAP_DIFF, LEVELS = "jet", "jet", "RdBu_r", 20


def grid_xy(n):
    c = (np.arange(n) + 0.5) / n
    return np.meshgrid(c, c)                      # X[iy, ix], Y[iy, ix]


def contour(ax, f, n, cmap=None, vmin=None, vmax=None, levels=None):
    """Filled contours of a cell field f (n*n,)."""
    X, Y = grid_xy(n)
    Z = np.asarray(f).reshape(n, n).T             # Z[iy, ix]
    lo = np.min(Z) if vmin is None else vmin
    hi = np.max(Z) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-12
    lv = np.linspace(lo, hi, (levels or LEVELS) + 1)
    cs = ax.contourf(X, Y, np.clip(Z, lo, hi), levels=lv, cmap=cmap or CMAP_MAIN)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    return cs


def plot_fields(F, label, n, path, cmap=None, cmap_std=None):
    """Truth, posterior mean and sd of log kappa; mean and sd of the pressure."""
    panels = [("log permeability (truth)", F["logk_true"], cmap or CMAP_MAIN),
              ("log permeability (mean)", F[f"{label}_logk_mean"], cmap or CMAP_MAIN),
              ("log permeability (std)", F[f"{label}_logk_sd"], cmap_std or CMAP_STD),
              ("pressure (mean)", F[f"{label}_p_mean"], cmap or CMAP_MAIN),
              ("pressure (std)", F[f"{label}_p_sd"], cmap_std or CMAP_STD)]
    fig, ax = plt.subplots(1, 5, figsize=(20, 3.6), constrained_layout=True)
    # truth and mean share one scale (they are the same quantity)
    lo = min(panels[0][1].min(), panels[1][1].min()); hi = max(panels[0][1].max(), panels[1][1].max())
    for k, (a, (t, f, cm)) in enumerate(zip(ax, panels)):
        cs = contour(a, f, n, cm, lo if k < 2 else None, hi if k < 2 else None)
        a.set_title(f"{t} [{label}]" if k else t, fontsize=10)
        fig.colorbar(cs, ax=a, fraction=.046)
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_pairs(F, n, path, cmap=None):
    """Truth and selected chains: (log kappa, pressure, pressure - truth pressure)."""
    labs, lk, pr, er = F["pair_labels"], F["pair_logk"], F["pair_p"], F["pair_relerr"]
    lo, hi = lk.min(), lk.max(); plo, phi_ = pr.min(), pr.max()
    dmax = np.abs(pr[1:] - pr[0]).max()
    fig, ax = plt.subplots(len(labs), 3, figsize=(10, 2.8 * len(labs)), constrained_layout=True)
    for i in range(len(labs)):
        c0 = contour(ax[i, 0], lk[i], n, cmap, lo, hi)
        c1 = contour(ax[i, 1], pr[i], n, cmap, plo, phi_)
        c2 = contour(ax[i, 2], pr[i] - pr[0], n, CMAP_DIFF, -dmax, dmax)
        ax[i, 0].set_ylabel(str(labs[i]))
        ax[i, 0].set_title("log permeability" if i == 0 else "", fontsize=9)
        ax[i, 1].set_title("pressure" if i == 0 else f"rel. err. {er[i]:.4f}", fontsize=9)
        ax[i, 2].set_title("pressure - truth" if i == 0 else "", fontsize=9)
    fig.colorbar(c0, ax=ax[:, 0], fraction=.04); fig.colorbar(c1, ax=ax[:, 1], fraction=.04)
    fig.colorbar(c2, ax=ax[:, 2], fraction=.04)
    fig.savefig(path, dpi=120); plt.close(fig)
