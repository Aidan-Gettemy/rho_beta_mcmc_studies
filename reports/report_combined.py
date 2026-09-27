"""
The one-figure argument (E2 + E3 + E4 on the same (beta, rho) plane) and a
single Markdown summary collecting every table.

    figures/combined/fig_argument.png
        (a) E2: every fixed pair fits the data (disks colored by RMS/sigma_p)
        (b) E3 curve band + E4 posterior cloud + truth, same axes
    figures/combined/fig_rho_comparison.png
        rho posterior (E4) vs prior, with the E2 grid rho values marked
    SUMMARY.md   (at BRHO_ROOT) - all tables/*.md + figure list
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from _common import have, load_problem, io
from brho import plotting as P
from brho.curve import BetaCurve

EXP = "combined"


def main():
    parts = [have(io.EXP_GRID), have(io.EXP_CURVE), have(io.EXP_SAMPLER)]
    if any(parts):
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
        src = io.EXP_GRID if parts[0] else (io.EXP_CURVE if parts[1] else io.EXP_SAMPLER)
        prob, _ = load_problem(src)
        syn = prob.syn
        if parts[0]:
            z = np.load(io.results_dir(io.EXP_GRID) + "/grid.npz")
            gs = load_problem(io.EXP_GRID)[0].syn.sigma_p
            sc = ax[0].scatter(z["beta"], z["rho"], c=z["rms"] / gs, s=300,
                               cmap="coolwarm", vmin=0, vmax=3, edgecolors="k", zorder=3)
            plt.colorbar(sc, ax=ax[0], label=r"RMS/$\sigma_p$")
        ax[0].set_title("(a) pressure data only: every pair fits")
        if parts[1]:
            prob_c, _ = load_problem(io.EXP_CURVE)
            cur = BetaCurve.load(io.results_dir(io.EXP_CURVE) + "/curve_kle.npz")
            b = cur.band(prob_c.syn.keff_obs)
            rr = np.linspace(0.0, cur.rho_max, 300)
            ax[1].fill_betweenx(b["rho"], b["q05"], b["q95"], color=P.C_BAND,
                                alpha=0.35, label=r"E3: 5–95% of $\beta_i$")
            ax[1].plot(cur.beta_bar(rr, prob_c.syn.keff_obs), rr, color=P.C_CURVE,
                       lw=2, label=r"E3: $\bar\beta(\rho)$")
        if parts[2]:
            zs = [np.load(f) for f in sorted(glob.glob(
                io.results_dir(io.EXP_SAMPLER) + "/chain*.npz"))]
            rb = np.concatenate([q["kept_beta"] for q in zs])
            rr_ = np.concatenate([q["kept_rho"] for q in zs])
            ax[1].plot(rb, rr_, ".", color=P.C_POST, ms=2, alpha=0.35,
                       label="E4: posterior samples")
        if parts[0]:
            ax[1].plot(z["beta"], z["rho"], "o", mfc="none", mec="0.6", ms=10,
                       label="E2 grid")
        ax[1].set_title(r"(b) + $K_{eff}$ via $\bar\beta(\rho)$: posterior concentrates")
        for a in ax:
            P.truth_star(a, syn.beta_true, syn.rho_true)
            a.set_xscale("log"); a.set_xlabel(r"$\beta$"); a.set_ylabel(r"$\rho$")
            a.set_xlim(0.3, 15); a.set_ylim(0, 3.5)
        ax[1].legend(loc="upper left")
        P.save(fig, EXP, "fig_argument")

    # ---- SUMMARY.md
    root = io.root()
    lines = ["# betarho 2D - results summary", ""]
    for exp in (io.EXP_SETUP, io.EXP_GRID, io.EXP_CURVE, io.EXP_PRIOR,
                io.EXP_SAMPLER, EXP):
        tds = sorted(glob.glob(os.path.join(root, "tables", exp, "*.md")))
        figs = sorted(glob.glob(os.path.join(root, "figures", exp, "*.png")))
        if not (tds or figs):
            continue
        lines += [f"## {exp}", ""]
        for t in tds:
            lines += [f"**{os.path.basename(t)[:-3]}**", "", open(t).read(), ""]
        for j in sorted(glob.glob(os.path.join(root, "tables", exp, "*.json"))):
            lines += [f"`{os.path.relpath(j, root)}`", "", "```json",
                      open(j).read(), "```", ""]
        for f in figs:
            rel = os.path.relpath(f, root)
            lines += [f"![{os.path.basename(f)}]({rel})", ""]
    with open(os.path.join(root, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("  wrote SUMMARY.md")


if __name__ == "__main__":
    main()
