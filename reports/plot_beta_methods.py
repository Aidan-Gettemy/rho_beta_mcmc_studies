"""
Five ways to get beta, side by side: median and 90% interval of beta (log axis)
against the truth, with sd(log beta), coverage and the PDE solves each needs.

    BRHO_ROOT=... python reports/plot_beta_methods.py [--exp e4_sampler]

Reads tables/<exp>/frozen_beta_pcn.json (written by frozen_beta_pcn.py; no
chains, no solves) and writes figures/<exp>/fig_beta_methods.png.
"""
import _common  # noqa: F401
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from _common import io

RED, BLUE, GRAY = "#b0563f", "#2a6f97", "#7a8691"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=io.EXP_SAMPLER)
    args = ap.parse_args()
    path = os.path.join(io.tables_dir(args.exp), "frozen_beta_pcn.json")
    if not os.path.exists(path):
        print(f"{path}: missing (run reports/frozen_beta_pcn.py first)"); return
    R = json.load(open(path))
    fs = R["frozen_state"]
    pcn_solves = R["pde_solves"]

    def solves(n):
        return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n:,}"
    rows = [  # label, summary, sd(log beta), colour, solves
        ("Curve prior at $\\rho^*$", R["prior_at_rho_star"], fs["prior_log_beta_sd"], GRAY, "—"),
        ("Sampler output, β on the curve", R["hard_curve"], R["hard_curve"]["log_beta_sd"], RED, "0"),
        ("Frozen $(\\rho^*,\\theta^*)$, pCN on $\\beta$", R["frozen_pcn"],
         R["frozen_pcn"]["log_beta_sd"], RED, solves(pcn_solves)),
        ("Frozen $(\\rho^*,\\theta^*)$, level swap", R["level_swap_frozen"],
         R["level_swap_frozen"]["log_beta_sd"], BLUE, "0"),
        ("Level swap, all samples", R["level_swap_all"], R["level_swap_all"]["log_beta_sd"], BLUE, "0"),
    ]
    fr_path = os.path.join(io.tables_dir(args.exp), "frozen_rho.json")
    if os.path.exists(fr_path):                       # frozen-rho baseline (frozen_rho_pcn.py)
        F = json.load(open(fr_path))
        if abs(F["rho_star"] - fs["rho"]) > 1e-9:
            print(f"warning: frozen-rho run uses rho* = {F['rho_star']:.4f}, "
                  f"frozen pCN uses {fs['rho']:.4f}")
        rows.insert(3, ("Frozen $\\rho^*$, pCN on $\\theta$ and $\\beta$", F["beta"],
                        F["log_beta_sd"], RED, solves(F["pde_solves_total"])))
    truth = R["truth"]["beta"]

    fig, ax = plt.subplots(figsize=(11, 0.75 * len(rows) + 0.45))
    n = len(rows)
    for i, (lab, q, sd, col, ns) in enumerate(rows):
        y = n - 1 - i
        ax.plot([q["lo"], q["hi"]], [y, y], color=col, lw=3, solid_capstyle="round")
        ax.plot(q["median"], y, "o", ms=8, mfc=col, mec="white", mew=1.5, zorder=3)
        ax.text(1.02, y, f"{sd:.3f}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=10, color="#1f2933")
        ax.text(1.17, y, "covers" if q["covered"] else "misses",
                transform=ax.get_yaxis_transform(), va="center", fontsize=10,
                color="#1f2933", fontweight="normal" if q["covered"] else "bold")
        ax.text(1.31, y, ns, transform=ax.get_yaxis_transform(), va="center",
                fontsize=10, color="#1f2933")
    for x, h in ((1.02, "sd(log β)"), (1.17, "truth"), (1.31, "extra PDE solves (all chains)")):
        ax.text(x, n - 0.35, h, transform=ax.get_yaxis_transform(), fontsize=9,
                color="#6b7785", va="bottom")
    ax.axvline(truth, color="k", lw=1.2, ls="--", zorder=0)
    ax.text(truth, -0.75, f" truth β = {truth:g}", fontsize=9, va="center")
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10)
    ax.set_ylim(-1, n - 0.3)
    ax.set_xscale("log")
    lo = min(r[1]["lo"] for r in rows); hi = max(r[1]["hi"] for r in rows)
    ax.set_xlim(lo / 1.15, hi * 1.15)
    ticks = [t for t in (0.5, 1, 1.5, 2, 3, 4, 5, 7, 10) if lo / 1.15 <= t <= hi * 1.15]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("β  (median and 90% interval, log scale)")
    ax.grid(axis="x", color="0.9", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{len(rows)} ways to get β · frozen state: lowest-Φ sample, ρ* = {fs['rho']:.3f} · "
                 f"σ_rel = {R['sigma_rel']:.3g}", fontsize=11, loc="left", pad=22)
    fig.tight_layout(); fig.subplots_adjust(right=0.66)
    out = os.path.join(io.figures_dir(args.exp), "fig_beta_methods.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
