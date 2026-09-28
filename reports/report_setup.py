"""
Setup figures and numbers (report Fig. 1 + checks).

    fig_setup.png         G_true | log K_true | p_true with the observation cells
    fig_kle_spectrum.png  eigenvalue decay and cumulative energy, r marked
    fig_truth_hist.png    histogram of G_true and log K_true
    tables/setup/setup.{md,csv,json}
        r, energy, sigma_p, K*_true, K*_obs, level, Gbar, scaling-lemma errors
Uses whichever experiment ran first (E2, E3 or E4) for the configuration.
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import numpy as np
import matplotlib.pyplot as plt

from _common import have, load_problem, io
from brho import plotting as P

EXP = io.EXP_SETUP


def main():
    src = next((e for e in (io.EXP_GRID, io.EXP_CURVE, io.EXP_SAMPLER) if have(e)), None)
    if src is None:
        print("setup: no experiment results yet"); return
    prob, man = load_problem(src)
    syn, phys, kle, n = prob.syn, prob.phys, prob.kle, prob.pc.n

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))
    P.field(ax[0], syn.G_true, n, f"input field $G$ ({syn.mode}), $\\ell={prob.pc.ell}$",
            cmap="RdBu_r")
    P.field(ax[1], syn.logK_true, n,
            rf"$\log K_{{true}}=\log {syn.beta_true}+{syn.rho_true}\,G$")
    P.field(ax[2], syn.p_true, n,
            f"$p_{{true}}$ (influx 1 left, $p$=0 right)\n{syn.n_obs} obs, chessboard",
            cmap="magma", obs_mask=syn.obs_mask)
    P.save(fig, EXP, "fig_setup")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    lam = kle.lam_all
    ax[0].semilogy(np.arange(1, len(lam) + 1), np.maximum(lam, 1e-16), "k.", ms=3)
    ax[0].axvline(kle.r, color="r", ls=":", label=f"r = {kle.r}")
    ax[0].set_xlabel("mode"); ax[0].set_ylabel(r"$\lambda_i$"); ax[0].legend()
    ax[0].set_title(f"KLE eigenvalues ({kle.method})")
    ax[1].plot(np.arange(1, len(lam) + 1), np.cumsum(lam) / kle.total, "k-")
    ax[1].axhline(prob.pc.energy, color="r", ls=":")
    ax[1].axvline(kle.r, color="r", ls=":")
    ax[1].set_xlim(0, 3 * kle.r); ax[1].set_xlabel("modes")
    ax[1].set_ylabel("cumulative energy")
    ax[1].set_title(f"energy captured {kle.energy:.3f}")
    P.save(fig, EXP, "fig_kle_spectrum")

    # the basis itself: leading modes and pointwise variance of the truncated
    # field (the Nystrom basis is smooth up to the boundary; the discrete one
    # lives on the cell centers only)
    fig, ax = plt.subplots(1, 5, figsize=(19, 3.6))
    for j in range(4):
        P.field(ax[j], kle.T[:, j] / np.sqrt(kle.lam[j]), n,
                rf"mode {j + 1}: $\lambda$={kle.lam[j]:.3g}", cmap="RdBu_r")
    var = kle.pointwise_variance()
    P.field(ax[4], var, n, f"pointwise var of $G$: mean {var.mean():.3f}",
            cmap="magma")
    P.save(fig, EXP, "fig_kle_basis")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(syn.G_true, 30, color=P.C_BAND, edgecolor="w", density=True)
    x = np.linspace(-4, 4, 200)
    ax[0].plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1)
    ax[0].set_title(f"$G$: mean {syn.Gbar:.3f}, std {syn.G_std:.3f}")
    ax[1].hist(syn.logK_true, 30, color=P.C_BAND, edgecolor="w", density=True)
    ax[1].axvline(syn.level_true, color="purple", ls=":", label="level")
    ax[1].set_title(r"$\log K_{true}$"); ax[1].legend()
    P.save(fig, EXP, "fig_truth_hist")

    sc = phys.scaling_check(syn.K_true)
    nqc = kle.meta.get("nq_check", {})
    row = dict(truth_mode=syn.mode, n=n, ell=prob.pc.ell, kle_method=kle.method,
               kle_nq=kle.meta.get("nq", ""), kle_quadrature=kle.meta.get("quadrature", ""),
               kle_nq_rel_change=nqc.get("max_rel_eig_change", ""), r=kle.r,
               energy=kle.energy, mean_pointwise_var=float(var.mean()), n_obs=syn.n_obs, sigma_p=syn.sigma_p,
               sigma_e=syn.sigma_e, keff_true=syn.keff_true,
               keff_obs=syn.keff_obs, level_true=syn.level_true,
               Gbar=syn.Gbar, G_std=syn.G_std,
               scaling_p_err=sc["p_rel_err"], scaling_keff_err=sc["keff_rel_err"])
    io.write_table(EXP, "setup", [row])
    io.save_json(io.tables_dir(EXP) + "/setup.json", row)
    print("setup:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
