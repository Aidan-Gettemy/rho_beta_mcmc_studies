"""
MPSRF and PSRF convergence curves for all KLE coefficients theta, in the
group's MATLAB convention (brho/mpsrf.py has the formulas).

    python reports/report_mpsrf.py [--exp e1_blocked] [--chains 16]
                                   [--skip 2 --step 15 --iter-burn 10]
                                   [--at 4900] [--level 1.2]

Uses the thinned samples saved in every chain file (kept_theta, every `thin`
iterations from iteration 0).  As in computeMPSRFcurve, the curve point at
thinned count k uses samples iter_burn, iter_burn + skip, ..., k (a growing
window from sample iter_burn); the x axis is k * thinning.  Two parameter sets:
    theta   all r KLE coefficients
    all     (theta, log beta, log rho)  = the whole sampled state up to an
            affine map (MPSRF is invariant to those)
Writes, for each set, into figures/<exp>/ and tables/<exp>/:
    fig_mpsrf_<set>.png   MPSRF vs iteration: y = 1, y = 1.2, crossing marked
    fig_psrf_<set>.png    max-over-components PSRF vs iteration, and the final
                          PSRF of every component
    mpsrf_<set>.csv       iteration, MPSRF, max PSRF, PSRF of each component
    mpsrf_summary.json    convergence iterations and values at --at
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from _common import io
from brho import mpsrf as M


def load_kept(exp, n_chains=None):
    paths = sorted(glob.glob(os.path.join(io.results_dir(exp), "chain*.npz")),
                   key=lambda p: int(os.path.basename(p)[5:-4]))
    if not paths:
        raise FileNotFoundError(f"no chain files in {io.results_dir(exp)}")
    if n_chains:
        paths = paths[:n_chains]
    zs = [np.load(p) for p in paths]
    L = min(len(z["kept_it"]) for z in zs)
    theta = np.stack([z["kept_theta"][:L] for z in zs])            # (m, L, r)
    lb = np.stack([z["kept_logbeta"][:L] for z in zs])
    rho = np.stack([z["kept_rho"][:L] for z in zs])
    its = np.asarray(zs[0]["kept_it"][:L])
    return dict(theta=theta, all=np.concatenate(
        [theta, lb[..., None], np.log(rho)[..., None]], axis=2)), its, len(paths)


def style(ax, fs=20):
    ax.tick_params(labelsize=fs)
    ax.xaxis.label.set_size(fs)
    ax.yaxis.label.set_size(fs)
    ax.title.set_size(fs + 2)


def fig_mpsrf(x, curve, conv, level, title, path):
    f, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, np.abs(curve), lw=2, label="MPSRF")
    ax.axhline(1.0, color="r", ls="-", lw=3, label="y=1.0")
    ax.axhline(level, color="r", ls=":", lw=3, label=f"y={level:g}")
    if conv is not None:
        ax.plot([conv, conv], [0, level], "k--", lw=3,
                label=f"MPSRF = {int(conv)}")
        ax.scatter([conv], [level], color="b", zorder=5)
    else:
        ax.plot([], [], "k--", lw=3, label=f"no crossing of {level:g}")
    ax.set_ylim(0, 5)
    ax.set_xlim(0, x.max())
    if np.nanmin(np.abs(curve)) > 5:
        ax.text(0.5, 0.5, f"above 5 throughout (final {abs(curve[-1]):.3g})",
                transform=ax.transAxes, ha="center", fontsize=18)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\hat{R}$")
    ax.set_title(title)
    ax.legend(fontsize=18)
    style(ax)
    f.tight_layout()
    f.savefig(path, dpi=150)
    plt.close(f)


def fig_psrf(x, psrf_max, psrf_final, conv, level, names, title, path):
    f, ax = plt.subplots(1, 2, figsize=(18, 7),
                         gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax[0].plot(x, psrf_max, lw=2, label="max PSRF")
    ax[0].axhline(1.0, color="r", ls="-", lw=3, label="y=1.0")
    ax[0].axhline(level, color="r", ls=":", lw=3, label=f"y={level:g}")
    if conv is not None:
        ax[0].plot([conv, conv], [0, level], "k--", lw=3, label=f"PSRF = {int(conv)}")
        ax[0].scatter([conv], [level], color="b", zorder=5)
    ax[0].set_ylim(0, 5)
    ax[0].set_xlim(0, x.max())
    if np.nanmin(psrf_max) > 5:
        ax[0].text(0.5, 0.5, f"above 5 throughout (final {psrf_max[-1]:.3g})",
                   transform=ax[0].transAxes, ha="center", fontsize=16)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel(r"$\max_i \hat{R}_i$")
    ax[0].set_title("PSRF (worst component) vs iteration")
    ax[0].legend(fontsize=16)
    idx = np.arange(len(psrf_final))
    ax[1].bar(idx, psrf_final, color=np.where(psrf_final > level, "#c44", "#48a"))
    ax[1].axhline(1.0, color="r", lw=2)
    ax[1].axhline(level, color="r", ls=":", lw=2)
    n = len(idx) - (2 if names is not None else 0)          # theta components
    ticks = list(range(0, n, 5)) + ([n, n + 1] if names is not None else [])
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels([f"θ{t + 1}" for t in ticks[:len(range(0, n, 5))]]
                          + (["logβ", "logρ"] if names is not None else []),
                          rotation=60, fontsize=12)
    top = max(5.0, 1.05 * float(np.max(psrf_final)))
    ax[1].set_ylim(0, top)
    ax[1].set_xlabel("component")
    ax[1].set_ylabel(r"final $\hat{R}_i$")
    ax[1].set_title(f"final PSRF per component ({(psrf_final > level).sum()} of "
                    f"{len(psrf_final)} above {level:g})")
    for a in ax:
        style(a, 16)
    f.suptitle(title, fontsize=20)
    f.tight_layout()
    f.savefig(path, dpi=150)
    plt.close(f)


def run(exp, n_chains=None, skip=2, step=15, iter_burn=10, level=1.2, at=4900,
        dispiter=10):
    sets, its, m = load_kept(exp, n_chains)
    thin = int(its[1] - its[0]) if len(its) > 1 else 1
    fdir, tdir = io.figures_dir(exp), io.tables_dir(exp)
    print(f"{exp}: {m} chains, {len(its)} thinned samples each (thinning {thin})")
    print(f"Thinning {thin}: skip {skip}, step {step}, iter burn {iter_burn}")
    summary = dict(exp=exp, n_chains=m, n_samples=len(its), thinning=thin,
                   skip=skip, step=step, iter_burn=iter_burn, level=level)
    for name, arr in sets.items():
        r = arr.shape[2]
        cur = M.mpsrf_curve(arr, skip=skip, step=step, iter_burn=iter_burn,
                            dispiter=dispiter)
        if len(cur["k"]) == 0 or np.all(np.isnan(cur["mpsrf"])):
            print(f"[{name}] too few samples for a curve "
                  f"({len(its)} per chain, step {step}, iter burn {iter_burn})")
            continue
        x = its[cur["k"] - 1].astype(float) + thin      # iterations run so far
        conv = M.convergence_iteration(x, cur["mpsrf"], level)
        conv_p = M.convergence_iteration(x, cur["psrf"], level)
        j = int(np.searchsorted(x, at)) if x.max() >= at else len(x) - 1
        labels = ([f"θ{i + 1}" for i in range(r - 2)] + ["logβ", "logρ"]
                  if name == "all" else None)
        what = (f"all {r} KLE coefficients θ" if name == "theta"
                else f"(θ, log β, log ρ), {r} dims")
        fig_mpsrf(x, cur["mpsrf"], conv, level, f"MPSRF vs Iteration: {what}, {m} chains",
                  os.path.join(fdir, f"fig_mpsrf_{name}.png"))
        fig_psrf(x, cur["psrf"], cur["psrf_i"][-1], conv_p, level, labels,
                 f"PSRF: {what}, {m} chains", os.path.join(fdir, f"fig_psrf_{name}.png"))
        head = "iteration,mpsrf,psrf_max," + ",".join(
            labels or [f"theta{i + 1}" for i in range(r)])
        np.savetxt(os.path.join(tdir, f"mpsrf_{name}.csv"),
                   np.column_stack([x, cur["mpsrf"], cur["psrf"], cur["psrf_i"]]),
                   delimiter=",", header=head, comments="", fmt="%.6g")
        summary[name] = dict(
            dim=r, mpsrf_convergence_iteration=conv, psrf_convergence_iteration=conv_p,
            at_iteration=float(x[j]), mpsrf_at=float(cur["mpsrf"][j]),
            psrf_max_at=float(cur["psrf"][j]), mpsrf_final=float(cur["mpsrf"][-1]),
            psrf_max_final=float(cur["psrf"][-1]),
            n_components_above_level_final=int((cur["psrf_i"][-1] > level).sum()),
            worst_components_final=[
                (labels[i] if labels else f"theta{i + 1}", round(float(cur["psrf_i"][-1][i]), 3))
                for i in np.argsort(cur["psrf_i"][-1])[::-1][:5]])
        c = "never (below the last iteration)" if conv is None else f"{int(conv)}"
        print(f"[{name}] Convergence using {m} Chains is {c}\n"
              f"[{name}] While at iteration {int(x[j])} MPSRF = {cur['mpsrf'][j]:.4f}, "
              f"max PSRF = {cur['psrf'][j]:.4f}; final MPSRF = {cur['mpsrf'][-1]:.4f}")
    io.save_json(os.path.join(tdir, "mpsrf_summary.json"), summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=None,
                    help="run folder, e.g. e1_blocked, e1_global (default: every E1 run)")
    ap.add_argument("--chains", type=int, default=None, help="use the first N chains")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--iter-burn", type=int, default=10)
    ap.add_argument("--level", type=float, default=1.2)
    ap.add_argument("--at", type=float, default=4900,
                    help="also report the values at this iteration")
    a = ap.parse_args(argv)
    exps = [a.exp] if a.exp else [e for e in (io.exp_e1("blocked"), io.exp_e1("global"))
                                  if glob.glob(os.path.join(io.results_dir(e), "chain*.npz"))]
    for e in exps:
        run(e, a.chains, a.skip, a.step, a.iter_burn, a.level, a.at)


if __name__ == "__main__":
    main()
