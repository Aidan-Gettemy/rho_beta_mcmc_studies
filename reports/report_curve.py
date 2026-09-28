"""
E3 figures and tables (method report Fig. 1 + finite-size diagnostics).

    fig_curve_beta.png      beta_bar(rho) for the chosen statistic, +/-1 std and
                            5-95% bands of the beta_i sample, the two other
                            statistics as thin lines, truth star (log beta)
    fig_curve_f.png         f(rho) = ensemble Keff at beta=1 with standard error
                            and one-realization scatter; Matheron 2D value 1;
                            KLE vs full-rank Cholesky overlay (consistency)
    fig_curve_scatter.png   relative scatter of Keff across realizations vs rho
    fig_curve_hist.png      histogram of beta_i at the grid rho nearest rho_true,
                            with beta_bar and beta_true marked
    fig_curve_level.png     spatial mean / variance of the MC realizations
                            (KLE vs Cholesky) - why a realization's own level
                            makes the constant-Keff set a band
    tables/e3_curve/curve_table.{md,csv}, curve_summary.json
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import os
import numpy as np
import matplotlib.pyplot as plt

from _common import have, load_problem, io
from brho import plotting as P
from brho.curve import BetaCurve, STATISTICS

EXP = io.EXP_CURVE


def main():
    if not have(EXP):
        print("E3: no results"); return
    prob, man = load_problem(EXP)
    syn = prob.syn
    Ks = syn.keff_obs
    d = io.results_dir(EXP)
    cur = BetaCurve.load(os.path.join(d, "curve_kle.npz"))
    zk = np.load(os.path.join(d, "curve_kle.npz"))
    chol_path = os.path.join(d, "curve_cholesky.npz")
    cho = BetaCurve.load(chol_path) if os.path.exists(chol_path) else None
    b = cur.band(Ks)
    rr = np.linspace(0, cur.rho_max, 300)

    # ---- main curve
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.fill_between(b["rho"], b["q05"], b["q95"], color=P.C_BAND, alpha=0.25,
                    label=r"5–95% of $\beta_i$")
    ax.fill_between(b["rho"], np.maximum(b["mean"] - b["std"], 0.5 * b["q05"]),
                    b["mean"] + b["std"], color=P.C_BAND, alpha=0.45,
                    label=r"mean $\pm$ 1 std of $\beta_i$ (lower edge clipped)")
    ax.plot(rr, cur.beta_bar(rr, Ks), "-", color=P.C_CURVE, lw=2,
            label=rf"$\bar\beta(\rho)$ [{cur.statistic}]")
    ax.plot(b["rho"], b["curve"], "o", color=P.C_CURVE, ms=4)
    for st, ls in zip([s for s in STATISTICS if s != cur.statistic], (":", "-.")):
        alt = BetaCurve(cur.rhos, cur.keff1, st)
        ax.plot(rr, alt.beta_bar(rr, Ks), ls, color="k", lw=1, label=st)
    P.truth_star(ax, syn.rho_true, syn.beta_true)
    ax.axhline(Ks, color="0.5", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\beta$")
    ax.set_title(rf"Offline curve under $K_{{eff}}=K^*={Ks:.3f}$  "
                 f"({cur.keff1.shape[1]} realizations / $\\rho$, KLE r={prob.r})")
    ax.legend(loc="upper left")
    P.save(fig, EXP, "fig_curve_beta")

    # ---- f(rho)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    k1 = cur.keff1
    ax.fill_between(cur.rhos, np.percentile(k1, 5, 1), np.percentile(k1, 95, 1),
                    color="0.85", label="one-realization 5–95%")
    ax.errorbar(cur.rhos, b["f_mean"], yerr=b["f_se"], fmt="o-", ms=3,
                color=P.C_CURVE, label="KLE: mean $K_{eff}$ ± s.e.")
    if cho is not None:
        bc = cho.band(Ks)
        ax.errorbar(cho.rhos, bc["f_mean"], yerr=bc["f_se"], fmt="s--", ms=3,
                    color="C1", label="full-rank Cholesky")
    ax.axhline(1.0, color="k", lw=1, ls=":", label="Matheron 2D (geometric mean) = 1")
    ax.set_yscale("log"); ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$f(\rho)=\overline{K}_{eff}(\rho,\beta=1)$")
    ax.set_title("Ensemble effective permeability at $\\beta=1$")
    ax.legend()
    P.save(fig, EXP, "fig_curve_f")

    # ---- relative scatter
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(cur.rhos, b["keff_relstd"], "o-", color=P.C_CURVE, label="std/mean of $K_{eff}$")
    ax.plot(cur.rhos, b["std"] / b["mean"], "s--", color="C3", label=r"std/mean of $\beta_i$")
    ax.axvline(syn.rho_true, color="k", ls=":", lw=1)
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel("relative scatter")
    ax.set_title("Realization-to-realization scatter (the band belongs to the curve)")
    ax.legend()
    P.save(fig, EXP, "fig_curve_scatter")

    # ---- beta_i histogram at rho_true
    a = int(np.argmin(abs(cur.rhos - syn.rho_true)))
    bi = Ks / cur.keff1[a]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(bi, bins=min(40, max(8, len(bi) // 4)), color=P.C_BAND, edgecolor="w")
    ax.axvline(float(cur.beta_bar(cur.rhos[a], Ks)), color=P.C_CURVE, lw=2,
               label=r"$\bar\beta$")
    ax.axvline(syn.beta_true, color="r", lw=2, label=r"$\beta^\dagger$")
    ax.set_xlabel(r"$\beta_i=K^*/K_{eff}(e^{\rho G_i})$")
    ax.set_title(rf"$\beta_i$ at $\rho={cur.rhos[a]:.2f}$")
    ax.legend()
    P.save(fig, EXP, "fig_curve_hist")

    # ---- realization level / variance
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(zk["Gbar"], 25, alpha=0.6, density=True, label="KLE")
    ax[1].hist(zk["Gvar"], 25, alpha=0.6, density=True, label="KLE")
    if cho is not None:
        zc = np.load(chol_path)
        ax[0].hist(zc["Gbar"], 25, alpha=0.6, density=True, label="Cholesky")
        ax[1].hist(zc["Gvar"], 25, alpha=0.6, density=True, label="Cholesky")
    ax[0].axvline(syn.Gbar, color="r", label="truth field")
    ax[0].set_title(r"spatial mean $\bar G_i$"); ax[0].legend()
    ax[1].set_title(r"spatial variance of $G_i$"); ax[1].legend()
    P.save(fig, EXP, "fig_curve_level")

    # ---- tables
    alt = {st: BetaCurve(cur.rhos, cur.keff1, st).beta_bar(cur.rhos, Ks)
           for st in STATISTICS}
    rows = [dict(rho=float(cur.rhos[i]), f_mean=float(b["f_mean"][i]),
                 f_se=float(b["f_se"][i]),
                 **{f"beta_{st}": float(alt[st][i]) for st in STATISTICS},
                 beta_std=float(b["std"][i]), beta_q05=float(b["q05"][i]),
                 beta_q95=float(b["q95"][i]),
                 keff_relstd=float(b["keff_relstd"][i]))
            for i in range(len(cur.rhos))]
    io.write_table(EXP, "curve_table", rows)
    bt = float(cur.beta_bar(syn.rho_true, Ks))
    summ = dict(Kstar=Ks, statistic=cur.statistic, n_real=int(cur.keff1.shape[1]),
                n_rho=len(cur.rhos), rho_range=[float(cur.rhos[0]), cur.rho_max],
                beta_bar_at_rho_true=bt, beta_true=syn.beta_true,
                rel_err_at_truth=bt / syn.beta_true - 1,
                beta_bar_at_0=float(cur.beta_bar(0.0, Ks)),
                band_at_rho_true=[float(np.percentile(bi, 5)), float(np.percentile(bi, 95))],
                minutes=man.get("minutes"))
    if cho is not None:
        summ["cholesky_vs_kle_beta_at_truth"] = float(cho.beta_bar(syn.rho_true, Ks)) / bt - 1
    io.save_json(io.tables_dir(EXP) + "/curve_summary.json", summ)
    print("E3:", summ)


if __name__ == "__main__":
    main()
