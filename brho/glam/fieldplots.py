"""Field figures for a GLAM run: log-permeability slices, tracer maps, concentration versus time.

Cell vectors are in solver order (x fastest): `to_grid(vec, nx, ny, nz)` gives an (nx, ny, nz) array.  Slices are shown
with x horizontal and y vertical, on a fine grid clipped to the plug disc (the same look as the group's MATLAB plots:
plotPermSlices, plotTracerConcUPGRADED, plotLogpermSliceSTAT).  No MPI, no solver.

  plot_logperm_slice_stat   reference | best | posterior mean | posterior std, one z-slice
  plot_slices               one field, several z-layers (log permeability or std)
  plot_tracer_maps          one concentration field, one panel per time at a z-layer
  plot_conc_vs_time         plug-averaged concentration versus time (reference, best, posterior mean +- 2 sd band)
  plot_conc_z_vs_time       per-z-layer disc-averaged concentration versus time: posterior mean +- 2 sd, reference, best
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

TRIM = 0.985            # shrink the disc radius a little to hide interpolation artefacts at the rim


def to_grid(vec, nx, ny, nz):
    return np.asarray(vec).reshape((nx, ny, nz), order="F")


def custom_redblue(n=256):
    half = n // 2
    r = np.r_[np.linspace(0.15, 0.95, half), np.ones(n - half)]
    g = np.r_[np.linspace(0.35, 0.95, half), np.linspace(0.95, 0.15, n - half)]
    b = np.r_[np.ones(half), np.linspace(0.95, 0.15, n - half)]
    return LinearSegmentedColormap.from_list("customRedBlue", np.c_[r, g, b], N=n)


def get_cmap(name):
    return custom_redblue() if name == "redblue" else plt.get_cmap(name)


def disc_slice(slice_xy, Lx, Ly, fill=np.nan, n_fine=300):
    """slice_xy (nx, ny) -> (ny_fine, n_fine) image clipped to the disc; x runs along the columns."""
    nx, ny = slice_xy.shape
    f = RegularGridInterpolator((np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)), slice_xy, bounds_error=False,
                                fill_value=fill)
    xf, yf = np.linspace(0, Lx, n_fine), np.linspace(0, Ly, n_fine)
    X, Y = np.meshgrid(xf, yf)
    img = f(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    r = TRIM * min(Lx, Ly) / 2
    img[(X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2 > r * r] = np.nan
    return img


def _rim(ax, Lx, Ly):
    t = np.linspace(0, 2 * np.pi, 500)
    r = TRIM * min(Lx, Ly) / 2
    ax.plot(Lx / 2 + r * np.cos(t), Ly / 2 + r * np.sin(t), "w-", lw=1.0)


def _panel(ax, img, Lx, Ly, cmap, clim):
    im = ax.imshow(img, origin="lower", extent=[0, Lx, 0, Ly], cmap=get_cmap(cmap), vmin=clim[0], vmax=clim[1],
                   interpolation="nearest")
    _rim(ax, Lx, Ly)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_aspect("equal")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return im


def _save(fig, path):
    fig.savefig(path, dpi=300 if fig.get_figwidth() >= 12 else 200, facecolor="w", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_logperm_slice_stat(ref, best, mean, std, grid, k, path, dom=None, label="log K"):
    """ref/best/mean/std: cell vectors (None to skip).  k: 0-based z-layer.  dom = (Lx, Ly) in mm."""
    nx, ny, nz = grid
    Lx, Ly = dom or (nx, ny)
    panels = [("Reference", ref, "jet", "val"), ("Best state", best, "jet", "val"),
              ("Posterior mean", mean, "jet", "val"), ("Posterior std", std, "jet", "std")]
    panels = [p for p in panels if p[1] is not None]
    imgs = {t: disc_slice(to_grid(v, nx, ny, nz)[:, :, k], Lx, Ly) for t, v, _, _ in panels}
    val = np.concatenate([imgs[t][np.isfinite(imgs[t])] for t, _, _, kind in panels if kind == "val"])
    fig, axs = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), facecolor="w", squeeze=False,
                            layout="constrained")
    for ax, (t, v, cm, kind) in zip(axs[0], panels):
        clim = (val.min(), val.max()) if kind == "val" else (np.nanmin(imgs[t]), np.nanmax(imgs[t]))
        im = _panel(ax, imgs[t], Lx, Ly, cm, clim)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("x [mm]")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label if kind == "val" else "std")
    axs[0, 0].set_ylabel("y [mm]")
    fig.suptitle(f"{label} slice, z-layer {k + 1} of {nz}", fontsize=13, fontweight="bold")
    return _save(fig, path)


def plot_slices(vec, grid, z_levels, path, dom=None, cmap="jet", label="log K", title=""):
    """One field at several z-layers (1-based `z_levels`), shared colour scale, 5 columns."""
    nx, ny, nz = grid
    Lx, Ly = dom or (nx, ny)
    V = to_grid(vec, nx, ny, nz)
    imgs = [disc_slice(V[:, :, k - 1], Lx, Ly) for k in z_levels]
    lo = min(np.nanmin(i) for i in imgs); hi = max(np.nanmax(i) for i in imgs)
    ncol = min(5, len(z_levels)); nrow = int(np.ceil(len(z_levels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol + 1, 3.3 * nrow), facecolor="w", squeeze=False)
    for s, (k, img) in enumerate(zip(z_levels, imgs)):
        ax = axs[s // ncol, s % ncol]
        im = _panel(ax, img, Lx, Ly, cmap, (lo, hi))
        ax.set_title(f"layer {k}", fontsize=10, fontweight="bold")
        if s // ncol == nrow - 1:
            ax.set_xlabel("x [mm]")
        else:
            ax.set_xticklabels([])
        if s % ncol == 0:
            ax.set_ylabel("y [mm]")
        else:
            ax.set_yticklabels([])
    for s in range(len(z_levels), nrow * ncol):
        axs[s // ncol, s % ncol].axis("off")
    fig.colorbar(im, ax=axs, shrink=0.8, label=label)
    fig.suptitle(title or label, fontsize=13, fontweight="bold")
    return _save(fig, path)


def plot_tracer_maps(S, grid, times, zlevel, path, dom=None, cmap="redblue", label="Tracer concentration",
                     title="", clim=None):
    """S (cells x nt); one panel per time at 1-based z-layer `zlevel`; shared colour scale; 2 rows x 5."""
    nx, ny, nz = grid
    Lx, Ly = dom or (nx, ny)
    nt = S.shape[1]
    imgs = [disc_slice(to_grid(S[:, t], nx, ny, nz)[:, :, zlevel - 1], Lx, Ly, fill=0.0) for t in range(nt)]
    lo, hi = clim if clim else (float(np.min(S)), float(np.max(S)))
    ncol = min(5, nt); nrow = int(np.ceil(nt / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol + 1, 3.3 * nrow), facecolor="w", squeeze=False)
    for t in range(nt):
        ax = axs[t // ncol, t % ncol]
        im = _panel(ax, imgs[t], Lx, Ly, cmap, (lo, hi))
        ax.set_title(f"t = {times[t]:g} s", fontsize=10, fontweight="bold")
        if t // ncol == nrow - 1:
            ax.set_xlabel("x [mm]")
        else:
            ax.set_xticklabels([])
        if t % ncol == 0:
            ax.set_ylabel("y [mm]")
        else:
            ax.set_yticklabels([])
    for t in range(nt, nrow * ncol):
        axs[t // ncol, t % ncol].axis("off")
    fig.colorbar(im, ax=axs, shrink=0.8, label=label)
    fig.suptitle(title or f"{label}, z-layer {zlevel}", fontsize=13, fontweight="bold")
    return _save(fig, path)


def plug_average(S, mask, grid):
    """Per-layer disc average of a cell x time array -> (nz, nt)."""
    nx, ny, nz = grid
    disc = np.asarray(mask).reshape((nx, ny, nz), order="F")[:, :, 0]
    return np.asarray(S, float).reshape((nx, ny, nz, -1), order="F")[disc].mean(axis=0)


def plot_conc_vs_time(times, ref=None, best=None, mean=None, sd=None, path="conc_vs_time.png", nsd=2):
    """Plug-averaged concentration versus time (all arguments (nt,) arrays; sd = std over samples of the average)."""
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor="w")
    if mean is not None and sd is not None:
        ax.fill_between(times, mean - nsd * sd, mean + nsd * sd, color="#2a78d6", alpha=0.25, lw=0,
                        label=f"posterior mean $\\pm${nsd} sd")
    if mean is not None:
        ax.plot(times, mean, color="#2a78d6", lw=2, label="posterior mean")
    if best is not None:
        ax.plot(times, best, color="#e07b1f", lw=1.8, ls="--", label="best state")
    if ref is not None:
        ax.plot(times, ref, "k.", ms=9, label="reference")
    ax.set_xlabel("time [s]"); ax.set_ylabel("plug-averaged concentration")
    ax.grid(alpha=0.3); ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return _save(fig, path)


def plot_conc_z_vs_time(times, z_mm, mean_z, sd_z, layers, path, ref_z=None, best_z=None, nsd=2):
    """Disc-averaged concentration of selected z-layers versus time.  mean_z / sd_z / ref_z / best_z: (nz, nt);
    layers: 1-based layer numbers.  Band = posterior mean +- nsd sd over the samples."""
    cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(layers)))
    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor="w")
    for c, k in zip(cols, layers):
        i = k - 1
        ax.fill_between(times, mean_z[i] - nsd * sd_z[i], mean_z[i] + nsd * sd_z[i], color=c, alpha=0.22, lw=0)
        ax.plot(times, mean_z[i], color=c, lw=2, label=f"z = {z_mm[i]:.1f} mm (layer {k})")
        if ref_z is not None:
            ax.plot(times, ref_z[i], ls="none", marker="o", ms=5, mfc="none", mec=c)
        if best_z is not None:
            ax.plot(times, best_z[i], color=c, lw=1.2, ls="--")
    ax.plot([], [], color="0.3", lw=2, label=f"posterior mean (band: $\\pm${nsd} sd)")
    if ref_z is not None:
        ax.plot([], [], ls="none", marker="o", mfc="none", mec="0.3", label="reference")
    if best_z is not None:
        ax.plot([], [], color="0.3", ls="--", lw=1.2, label="best state")
    ax.set_xlabel("time [s]"); ax.set_ylabel("disc-averaged concentration")
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Concentration averaged over each z-disc, versus time", loc="left", fontsize=11)
    return _save(fig, path)
