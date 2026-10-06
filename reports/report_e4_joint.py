"""
The E4 (rho, beta) figure on a fixed rho range, in two versions:

    fig_e4_curve_band.png   the curve beta_bar(rho) and the 5-95% band of the
                            curve realizations beta_i, with the truth; no
                            posterior (needs no chains)
    fig_e4_joint_full.png   the same with the E4 posterior samples (needs the
                            chain checkpoints; reads only kept_rho / kept_beta)

    BRHO_ROOT=... python reports/report_e4_joint.py [--exp e4_sampler] [--rho-max 5]
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from _common import have, io
from brho import plotting as P
from brho.curve import BetaCurve


def draw(curve, Kstar, truth, rho_max, samples=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    b = curve.band(Kstar)
    keep = b["rho"] <= rho_max + 1e-12
    ax.fill_between(b["rho"][keep], b["q05"][keep], b["q95"][keep], color=P.C_BAND,
                    alpha=0.3, label=r"E3: 5–95% of $\beta_i$")
    rr = np.linspace(0, rho_max, 300)
    ax.plot(rr, curve.beta_bar(rr, Kstar), color=P.C_CURVE, lw=1.5, label=r"$\bar\beta(\rho)$")
    if samples is not None:
        ax.plot(samples[0], samples[1], ".", color=P.C_POST, ms=2, alpha=0.4, label="E4 posterior")
    P.truth_star(ax, *truth)
    ax.set_yscale("log"); ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\beta$")
    ax.set_xlim(0, rho_max)
    ax.legend(loc="upper left")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=io.EXP_SAMPLER)
    ap.add_argument("--rho-max", type=float, default=5.0)
    a = ap.parse_args()
    if not have(a.exp):
        print(f"{a.exp}: no results"); return
    d = io.results_dir(a.exp)
    curve = BetaCurve.load(os.path.join(d, "curve.npz"))
    t = np.load(os.path.join(d, "truth.npz"))
    Kstar, truth = float(t["keff_obs"]), (float(t["rho_true"]), float(t["beta_true"]))
    rho_max = min(a.rho_max, curve.rho_max)

    P.save(draw(curve, Kstar, truth, rho_max), a.exp, "fig_e4_curve_band")
    paths = io.chain_files(d)
    if not paths:
        print("no chain files: fig_e4_curve_band only"); return
    rho, beta = [], []
    for p in paths:
        with np.load(p) as z:                 # npz loads keys lazily: only these two
            rho.append(z["kept_rho"]); beta.append(z["kept_beta"])
    rho, beta = np.concatenate(rho), np.concatenate(beta)
    P.save(draw(curve, Kstar, truth, rho_max, (rho, beta)), a.exp, "fig_e4_joint_full")
    print(f"{len(rho)} samples from {len(paths)} chains")


if __name__ == "__main__":
    main()
