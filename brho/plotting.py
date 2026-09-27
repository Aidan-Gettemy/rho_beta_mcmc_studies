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
