"""Shared figure helpers so all reports look the same."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import io

TRUTH = dict(marker="*", ms=20, mfc="yellow", mec="k", ls="none", zorder=6)
C_CURVE = "#2b5d8a"
C_BAND = "#7fa8d0"
C_POST = "#c0504d"
C_PRIOR = "0.35"
EXT = [0, 1, 0, 1]

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150,
                     "axes.grid": False, "font.size": 10,
                     "axes.titlesize": 10.5, "legend.fontsize": 8.5})


def field(ax, v, n, title="", cmap="viridis", vmin=None, vmax=None,
          obs_mask=None, cbar=True):
    im = ax.imshow(np.asarray(v).reshape(n, n).T, origin="lower", extent=EXT,
                   cmap=cmap, vmin=vmin, vmax=vmax)
    if obs_mask is not None:
        xs, ys = np.where(np.asarray(obs_mask).reshape(n, n))
        ax.scatter((xs + .5) / n, (ys + .5) / n, s=6, c="cyan", marker="s",
                   linewidths=0)
    ax.set_title(title)
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    if cbar:
        plt.colorbar(im, ax=ax, shrink=0.85)
    return im


def truth_star(ax, x, y, label="truth"):
    ax.plot(x, y, label=label, **TRUTH)


def slice_row(n, y=0.53):
    """Row index jr of the horizontal slice closest to y (0.53 -> 8 for n=16,
    16 for n=32; figure labels print the actual cell-center y)."""
    return int(np.clip(round(y * n - 0.5), 0, n - 1))


def save(fig, exp, name):
    path = os.path.join(io.figures_dir(exp), name + ".png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, io.root()))
    return path


def lognormal_pdf(x, mu, s):
    x = np.asarray(x, float)
    return np.exp(-0.5 * ((np.log(x) - mu) / s) ** 2) / (x * s * np.sqrt(2 * np.pi))


def prior_fields(rows, target, n, blocks=None, title="", vs=3.0):
    """Pointwise check of prior sampling (likelihood off), one row per
    strategy.  rows: list of (label, samples (k, N), pmean (N,), pvar (N,)).
    Columns: the k sample fields (common scale +-vs), the pointwise mean, the
    pointwise variance and its ratio to `target` (the exact prior pointwise
    variance, diag(T T^T)); with `blocks`, the subdomain interfaces are drawn
    on the ratio.  Returns the figure; the ratios are in fig.ratios."""
    ns = max(len(r[1]) for r in rows)
    fig, axs = plt.subplots(len(rows), ns + 3, figsize=(3.1 * (ns + 3), 2.9 * len(rows)),
                            squeeze=False)
    vm = max(float(np.abs(r[2]).max()) for r in rows)
    vv = max(float(r[3].max()) for r in rows)
    edges = [i / blocks for i in range(1, blocks)] if blocks else []
    fig.ratios = {}
    for i, (lab, smp, pm, pv) in enumerate(rows):
        for j in range(ns):
            if j < len(smp):
                field(axs[i, j], smp[j], n, f"{lab}: sample {j + 1}", cmap="viridis",
                      vmin=-vs, vmax=vs, cbar=(j == ns - 1))
            else:
                axs[i, j].axis("off")
        field(axs[i, ns], pm, n, f"pointwise mean (max |.| {np.abs(pm).max():.3f})",
              cmap="RdBu_r", vmin=-vm, vmax=vm)
        field(axs[i, ns + 1], pv, n, f"pointwise variance ({pv.min():.2f}-{pv.max():.2f})",
              cmap="magma", vmin=0, vmax=vv)
        ratio = pv / target
        fig.ratios[lab] = ratio
        field(axs[i, ns + 2], ratio, n, f"variance / target ({ratio.min():.2f}-{ratio.max():.2f})",
              cmap="RdBu_r", vmin=0.5, vmax=1.5)
        for e in edges:
            axs[i, ns + 2].axvline(e, color="k", lw=0.8, ls="--")
            axs[i, ns + 2].axhline(e, color="k", lw=0.8, ls="--")
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        ax.title.set_fontsize(9)
    fig.suptitle(title + (f"   target variance = diag(T T^T), mean {np.mean(target):.3f}"))
    fig.tight_layout()
    return fig
