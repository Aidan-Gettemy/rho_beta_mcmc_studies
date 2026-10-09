"""
Joint posterior of (rho, beta): naive block-pCN (E1) against the curve sampler
(E4, moves A+B) after the level split, as scatter + KDE.

The two samplers give essentially the same posterior log K field (the data pin
log K); where they differ is how that field is split into beta and rho.  This
figure shows the split:

  E1           (rho, log beta) sampled jointly under independent priors
               log beta ~ N(mu, s^2), log rho ~ N(mu, s^2): the pair moves
               along the direction that leaves log K's level fixed.
  E4 + split   rho from the curve sampler; log beta from the level-split draw
               log beta | rho, theta ~ N(mu, 1/P) around the curve beta_bar(rho),
               importance-weighted (brho.beta_upgrade.level_swap).

Panels: (a) E1 scatter + KDE, (b) E4 + level split scatter + KDE, (c) both KDE
contours overlaid.  Each panel also shows the curve beta_bar(rho) with its 90%
scatter band and the truth.  Contours enclose 50% and 90% of the posterior
mass (highest-density regions of the KDE in (rho, log beta)).

    python reports/plot_rho_beta_joint.py \
        [--e1-root runs/cluster_kle] [--e1-exp e1_blocked] \
        [--e4-root runs/cluster_kle_sig2e3] [--e4-exp e4_sampler_ab] [--out-root <e4-root>]

Needs the chain files of both runs (run on the cluster); no PDE solves.
Writes <out-root>/figures/e1_vs_e4_ab/fig_rho_beta_joint.png and
<out-root>/tables/e1_vs_e4_ab/rho_beta_joint.{json,md}.
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from _common import io, C, load_problem
from brho.curve import BetaCurve
from brho.beta_upgrade import constant_coords, level_swap, summarize

E1C, E4C, GR = "#c08a2e", "#2a6f97", "#9aa5ae"


def with_root(root, fn):
    old = os.environ.get("BRHO_ROOT")
    os.environ["BRHO_ROOT"] = os.path.abspath(root)
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("BRHO_ROOT", None)
        else:
            os.environ["BRHO_ROOT"] = old


def load_e1(exp, stride):
    """Pooled post-burn-in (rho, log beta) from E1's per-iteration traces."""
    d = io.results_dir(exp)
    paths = io.chain_files(d)
    if not paths:
        raise SystemExit(f"{exp}: no chain files in {d} (run on the cluster)")
    man = io.read_manifest(exp)
    cfg = C.from_dict(C.NaiveConfig, man["configs"]["NaiveConfig"])
    RH, LB, L = [], [], None
    for p in paths:
        with np.load(p) as z:
            n = int(z["it"])
            L = n if L is None else min(L, n)
            RH.append(np.asarray(z["rho"][:n], float)); LB.append(np.asarray(z["logbeta"][:n], float))
    nb = min(int(cfg.n_burn), L - 1)
    rho = np.concatenate([x[nb:L:stride] for x in RH])
    lb = np.concatenate([x[nb:L:stride] for x in LB])
    return dict(rho=rho, lb=lb, n_chains=len(paths), iterations=L, burn=nb,
                prior=f"log beta ~ N({cfg.mu_logbeta}, {cfg.s_logbeta}^2), "
                      f"log rho ~ N({cfg.mu_logrho}, {cfg.s_logrho}^2)",
                truth=man["truth"])


def load_e4(exp, seed):
    """Curve-sampler samples expanded by the level split (weighted)."""
    prob, man = load_problem(exp)
    d = io.results_dir(exp)
    paths = io.chain_files(d)
    if not paths:
        raise SystemExit(f"{exp}: no chain files in {d} (run on the cluster)")
    syn, kle, r = prob.syn, prob.kle, prob.r
    curve = BetaCurve.load(os.path.join(d, "curve.npz"))
    TH, RH, BH = [], [], []
    for p in paths:
        with np.load(p) as z:
            rho = z["kept_rho"].ravel()
            RH.append(rho); TH.append(z["kept_theta"].reshape(len(rho), r))
            BH.append(z["kept_beta"].ravel())
    theta, rho, beta_h = np.concatenate(TH), np.concatenate(RH), np.concatenate(BH)
    m = np.log(beta_h)                                   # log beta_bar(rho)
    s = np.sqrt(np.array([curve.log_scatter(x) for x in rho]) ** 2 + syn.sigma_e ** 2)
    c, _ = constant_coords(kle.T)
    up = level_swap(theta, rho, c, m, s, np.random.default_rng(seed), return_theta=False)
    grid = np.linspace(max(rho.min() * 0.8, curve.rhos.min()), min(rho.max() * 1.2, curve.rho_max), 200)
    band_m = np.log(curve.beta_bar(grid, syn.keff_obs))
    band_s = np.sqrt(np.array([curve.log_scatter(x) for x in grid]) ** 2 + syn.sigma_e ** 2)
    return dict(rho=rho, lb=up["lb"], w=up["w"], ess=up["ess"], lb_hard=m,
                n_chains=len(paths), curve=(grid, band_m, band_s),
                truth=dict(beta_true=syn.beta_true, rho_true=syn.rho_true))


def resample(x, y, w, n, rng):
    if w is None:
        idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    else:
        idx = rng.choice(len(x), size=n, replace=True, p=w)
    return x[idx], y[idx]


def kde_levels(kde, pts, masses=(0.5, 0.9)):
    """Density thresholds enclosing the given posterior mass (HPD of the KDE)."""
    dens = np.sort(kde(pts))[::-1]
    cum = np.arange(1, len(dens) + 1) / len(dens)
    return [float(dens[np.searchsorted(cum, q)]) for q in sorted(masses, reverse=True)]


def stats(rho, lb, w=None):
    ww = np.full(len(rho), 1 / len(rho)) if w is None else w
    mr, ml = ww @ np.log(rho), ww @ lb
    cov = ww @ ((np.log(rho) - mr) * (lb - ml))
    corr = cov / np.sqrt((ww @ (np.log(rho) - mr) ** 2) * (ww @ (lb - ml) ** 2))
    return dict(corr_logrho_logbeta=float(corr), sd_log_beta=float(np.sqrt(ww @ (lb - ml) ** 2)),
                sd_log_rho=float(np.sqrt(ww @ (np.log(rho) - mr) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1-root", default="runs/cluster_kle")
    ap.add_argument("--e1-exp", default="e1_blocked")
    ap.add_argument("--e4-root", default="runs/cluster_kle_sig2e3")
    ap.add_argument("--e4-exp", default="e4_sampler_ab")
    ap.add_argument("--out-root", default=None, help="default: --e4-root")
    ap.add_argument("--stride", type=int, default=10, help="pool every n-th E1 iteration")
    ap.add_argument("--n-scatter", type=int, default=4000)
    ap.add_argument("--n-kde", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_root = a.out_root or a.e4_root
    rng = np.random.default_rng(a.seed)

    E1 = with_root(a.e1_root, lambda: load_e1(a.e1_exp, a.stride))
    E4 = with_root(a.e4_root, lambda: load_e4(a.e4_exp, a.seed))
    tb, tr = E4["truth"]["beta_true"], E4["truth"]["rho_true"]
    if abs(E1["truth"]["beta_true"] - tb) > 1e-9 or abs(E1["truth"]["rho_true"] - tr) > 1e-9:
        print(f"warning: truths differ: E1 (beta, rho) = ({E1['truth']['beta_true']}, "
              f"{E1['truth']['rho_true']}), E4 ({tb}, {tr})")

    runs = [("naive block-pCN (E1)", E1C, E1["rho"], E1["lb"], None),
            ("curve sampler (A+B) + level split", E4C, E4["rho"], E4["lb"], E4["w"])]
    kdes, scat = [], []
    for _, _, rho, lb, w in runs:
        kx, ky = resample(rho, lb, w, a.n_kde, rng)
        kde = gaussian_kde(np.vstack([kx, ky]))
        kdes.append((kde, kde_levels(kde, np.vstack([kx, ky]))))
        scat.append(resample(rho, lb, w, a.n_scatter, rng))

    allr = np.concatenate([s[0] for s in scat]); alll = np.concatenate([s[1] for s in scat])
    rlo, rhi = np.percentile(allr, [0.2, 99.8]); llo, lhi = np.percentile(alll, [0.2, 99.8])
    pr, pl = 0.08 * (rhi - rlo), 0.08 * (lhi - llo)
    rlo, rhi, llo, lhi = rlo - pr, rhi + pr, llo - pl, lhi + pl
    rlo, rhi = min(rlo, tr - pr), max(rhi, tr + pr)
    llo, lhi = min(llo, np.log(tb) - pl), max(lhi, np.log(tb) + pl)
    gx, gy = np.meshgrid(np.linspace(rlo, rhi, 160), np.linspace(llo, lhi, 160))
    grid = np.vstack([gx.ravel(), gy.ravel()])
    cg, cm, cs = E4["curve"]
    inside = (cg >= rlo) & (cg <= rhi)

    def decorate(ax):
        ax.plot(cg[inside], np.exp(cm[inside]), color="k", lw=1.2, label=r"curve $\bar\beta(\rho)$")
        ax.fill_between(cg[inside], np.exp(cm - 1.645 * cs)[inside], np.exp(cm + 1.645 * cs)[inside],
                        color=GR, alpha=0.25, lw=0, label="curve 90% scatter")
        ax.plot(tr, tb, marker="*", ms=16, mfc="white", mec="k", mew=1.5, zorder=5, ls="none",
                label=f"truth (ρ = {tr:g}, β = {tb:g})")
        ax.set_yscale("log"); ax.set_xlim(rlo, rhi); ax.set_ylim(np.exp(llo), np.exp(lhi))
        ax.set_xlabel("ρ"); ax.set_ylabel("β")
        ticks = [t for t in (0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10) if np.exp(llo) <= t <= np.exp(lhi)]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()

    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), sharex=True, sharey=True)
    S = {}
    for i, ((lab, col, rho, lb, w), (kde, lev), (sx, sy)) in enumerate(zip(runs, kdes, scat)):
        st = stats(rho, lb, w)
        S[lab] = dict(**st, rho=summarize(rho, tr, w=w), beta=summarize(np.exp(lb), tb, w=w))
        dens = kde(grid).reshape(gx.shape)
        for ax, scatter in ((axs[i], True), (axs[2], False)):
            if scatter:
                ax.scatter(sx, np.exp(sy), s=3, alpha=0.25, color=col, lw=0, rasterized=True)
            # lev = [90% threshold, 50% threshold], ascending
            ax.contour(gx, np.exp(gy), dens, levels=lev, colors=[col], linewidths=[2.0, 1.4],
                       linestyles=["-", "--"])
        axs[i].set_title(f"({'ab'[i]}) {lab}\ncorr(log ρ, log β) = {st['corr_logrho_logbeta']:+.2f}, "
                         f"sd(log β) = {st['sd_log_beta']:.3f}", loc="left", fontsize=10)
    for ax in axs:
        decorate(ax)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=E1C, lw=2, label="naive block-pCN (E1)"),
                           Line2D([], [], color=E4C, lw=2, label="curve sampler + level split"),
                           Line2D([], [], color="0.3", lw=2, label="90% region"),
                           Line2D([], [], color="0.3", lw=1.2, ls="--", label="50% region")]
               + axs[2].get_legend_handles_labels()[0][:3], fontsize=9, loc="lower center",
               ncol=7, frameon=False)
    axs[2].set_title("(c) both: KDE contours (50% dashed, 90% solid)", loc="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fd = os.path.join(out_root, "figures", "e1_vs_e4_ab"); os.makedirs(fd, exist_ok=True)
    td = os.path.join(out_root, "tables", "e1_vs_e4_ab"); os.makedirs(td, exist_ok=True)
    fig.savefig(os.path.join(fd, "fig_rho_beta_joint.png"), dpi=150)

    res = dict(e1=dict(exp=a.e1_exp, root=a.e1_root, n_chains=E1["n_chains"], iterations=E1["iterations"],
                       burn=E1["burn"], stride=a.stride, n=int(len(E1["rho"])), prior=E1["prior"]),
               e4=dict(exp=a.e4_exp, root=a.e4_root, n_chains=E4["n_chains"], n=int(len(E4["rho"])),
                       level_split_weight_ess=E4["ess"]),
               truth=dict(beta=tb, rho=tr), summaries=S)
    with open(os.path.join(td, "rho_beta_joint.json"), "w") as f:
        json.dump(res, f, indent=1)
    lines = ["| sampler | ρ median [90%] | β median [90%] | sd(log ρ) | sd(log β) | corr(log ρ, log β) |",
             "|---|---|---|---|---|---|"]
    for lab, v in S.items():
        rq, bq = v["rho"], v["beta"]
        lines.append(f"| {lab} | {rq['median']:.3f} [{rq['lo']:.3f}, {rq['hi']:.3f}] | "
                     f"{bq['median']:.3f} [{bq['lo']:.3f}, {bq['hi']:.3f}] | {v['sd_log_rho']:.3f} | "
                     f"{v['sd_log_beta']:.3f} | {v['corr_logrho_logbeta']:+.3f} |")
    lines += ["", f"Truth: ρ = {tr:g}, β = {tb:g}. E1: {E1['prior']}; {E1['n_chains']} chains, "
              f"burn-in {E1['burn']}, every {a.stride}th iteration. E4: {E4['n_chains']} chains, "
              f"level-split weight ESS {100 * E4['ess'] / len(E4['rho']):.1f}%."]
    with open(os.path.join(td, "rho_beta_joint.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(os.path.join(fd, "fig_rho_beta_joint.png"))


if __name__ == "__main__":
    main()
