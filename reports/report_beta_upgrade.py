"""
Level-swap post-processing of a hard-curve (E4, 3-move) run: honest beta
uncertainty from the saved (rho, theta) samples, no new PDE solves
(brho/beta_upgrade.py has the derivation).

    BRHO_ROOT=... python reports/report_beta_upgrade.py [--exp e4_sampler]
        [--seed 0] [--solve-check 200]

--solve-check N: also re-solve N upgraded samples to measure the misfit change
caused by the 3.4% of the constant field outside the KLE span (optional exact
correction; the default 0 does no PDE solve at all).

Writes tables/<exp>/beta_upgrade.{json,md} and figures/<exp>/fig_beta_upgrade.png:
    beta: hard curve | upgraded (weighted) | upgraded (unweighted)
    rho:  hard curve | reweighted
    level before and after (must agree), weight ESS, and the "frozen" variant
    (one draw set at the highest-posterior sample).
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from _common import have, load_problem, io, C
from brho.curve import BetaCurve
from brho.samplers import load_chains
from brho.beta_upgrade import constant_coords, level_swap, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=io.EXP_SAMPLER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solve-check", type=int, default=0)
    args = ap.parse_args()
    exp = args.exp
    if not have(exp):
        print(f"{exp}: no results"); return
    prob, man = load_problem(exp)
    d = io.results_dir(exp)
    paths = io.chain_files(d)
    if not paths:
        print(f"{exp}: no chains"); return
    ch = load_chains(paths)
    curve = BetaCurve.load(os.path.join(d, "curve.npz"))
    scfg = C.from_dict(C.SamplerConfig, man["configs"]["SamplerConfig"])
    syn, kle, r = prob.syn, prob.kle, prob.r
    Kstar = syn.keff_obs
    sig_e2 = syn.sigma_e ** 2

    theta = ch["kept_theta"].reshape(-1, r)
    rho = ch["kept_rho"].ravel()
    beta_h = ch["kept_beta"].ravel()
    pobs = ch["kept_pobs"].reshape(len(rho), -1)
    m = np.log(beta_h)                                   # = log beta_bar(rho)
    s = np.sqrt(np.array([curve.log_scatter(x) for x in rho]) ** 2 + sig_e2)
    c, c_resid = constant_coords(kle.T)
    rng = np.random.default_rng(args.seed)
    up = level_swap(theta, rho, c, m, s, rng)

    lb_true = np.log(syn.beta_true)
    level_h = m + rho * kle.spatial_mean(theta)
    level_u = up["lb"] + rho * kle.spatial_mean(up["theta"])
    res = dict(
        exp=exp, n_samples=int(len(rho)), n_chains=int(ch["kept_rho"].shape[0]),
        c_norm=float(np.linalg.norm(c)), constant_field_residual=c_resid,
        truth=dict(beta=syn.beta_true, rho=syn.rho_true, level=float(syn.level_true)),
        beta_hard=summarize(beta_h, syn.beta_true),
        beta_upgraded=summarize(np.exp(up["lb"]), syn.beta_true, w=up["w"]),
        beta_upgraded_unweighted=summarize(np.exp(up["lb"]), syn.beta_true),
        log_beta_sd=dict(hard=float(np.std(m)),
                         upgraded=summarize(up["lb"], lb_true, w=up["w"])["sd"],
                         per_sample_sd_median=float(np.median(up["sd"]))),
        rho_hard=summarize(rho, syn.rho_true),
        rho_reweighted=summarize(rho, syn.rho_true, w=up["w"]),
        level_hard=summarize(level_h, syn.level_true),
        level_upgraded=summarize(level_u, syn.level_true, w=up["w"]),
        level_shift_max=float(np.max(np.abs(level_u - level_h))),
        weight_ess=up["ess"], weight_ess_fraction=up["ess"] / len(rho),
        logw_sd=float(np.std(up["logw"])),
    )

    # "frozen" variant: one (rho, theta), the highest-posterior kept sample
    res_pt = pobs - syn.d_p[None, :]
    phi = 0.5 * prob._w * np.einsum("ij,ij->i", res_pt, res_pt)
    w_lat = (np.log(rho) - scfg.mu_rho) / scfg.s_rho
    logpost = -phi - 0.5 * np.einsum("ij,ij->i", theta, theta) - 0.5 * w_lat ** 2
    k = int(np.nanargmax(logpost))
    nf = 20000
    fz = level_swap(np.repeat(theta[k:k + 1], nf, 0), np.full(nf, rho[k]), c,
                    np.full(nf, m[k]), np.full(nf, s[k]), rng, return_theta=False)
    res["frozen"] = dict(rho=float(rho[k]), beta_on_curve=float(beta_h[k]),
                         beta=summarize(np.exp(fz["lb"]), syn.beta_true))

    if args.solve_check > 0:
        idx = rng.choice(len(rho), size=min(args.solve_check, len(rho)), replace=False)
        dphi = []
        for i in idx:
            K = np.exp(up["lb"][i] + rho[i] * kle.field(up["theta"][i]))
            dphi.append(prob.potential(K, scfg.use_keff_term)[0] - phi[i])
        dphi = np.array(dphi)
        res["solve_check"] = dict(n=int(len(idx)), dphi_median=float(np.median(dphi)),
                                  dphi_abs_max=float(np.max(np.abs(dphi))),
                                  dphi_sd=float(np.std(dphi)))

    td, fd = io.tables_dir(exp), io.figures_dir(exp)
    with open(os.path.join(td, "beta_upgrade.json"), "w") as f:
        json.dump(res, f, indent=1)
    rows = [("beta, hard curve", res["beta_hard"]),
            ("beta, upgraded (weighted)", res["beta_upgraded"]),
            ("beta, upgraded (unweighted)", res["beta_upgraded_unweighted"]),
            ("beta, frozen at best sample", res["frozen"]["beta"]),
            ("rho, hard curve", res["rho_hard"]),
            ("rho, reweighted", res["rho_reweighted"]),
            ("level, hard curve", res["level_hard"]),
            ("level, upgraded", res["level_upgraded"])]
    with open(os.path.join(td, "beta_upgrade.md"), "w") as f:
        f.write("| quantity | median | 90% interval | sd | truth | covered |\n|---|---|---|---|---|---|\n")
        for name, q in rows:
            f.write(f"| {name} | {q['median']:.3f} | [{q['lo']:.3f}, {q['hi']:.3f}] | "
                    f"{q['sd']:.3f} | {q['truth']:.3f} | {q['covered']} |\n")
        f.write(f"\nweight ESS {res['weight_ess']:.0f} of {res['n_samples']} "
                f"({100 * res['weight_ess_fraction']:.1f}%); sd(log w) {res['logw_sd']:.3f}; "
                f"max level shift {res['level_shift_max']:.2e}; |c| {res['c_norm']:.3f}, "
                f"constant-field residual {res['constant_field_residual']:.3f}\n")
        if "solve_check" in res:
            sc = res["solve_check"]
            f.write(f"\nsolve check ({sc['n']} samples): Phi_new - Phi_old median "
                    f"{sc['dphi_median']:.3g}, sd {sc['dphi_sd']:.3g}, max |.| {sc['dphi_abs_max']:.3g}\n")

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    lbins = np.linspace(min(up["lb"].min(), m.min()) - 0.1, max(up["lb"].max(), m.max()) + 0.1, 70)
    ax[0].hist(m, lbins, density=True, alpha=0.6, color="#b0563f", label="hard curve")
    ax[0].hist(up["lb"], lbins, weights=up["w"] * len(rho), density=True, alpha=0.6,
               color="#2a6f97", label="upgraded")
    ax[0].axvline(lb_true, color="k", lw=2, label="truth")
    for q, col in ((res["beta_hard"], "#b0563f"), (res["beta_upgraded"], "#2a6f97")):
        ax[0].axvspan(np.log(q["lo"]), np.log(q["hi"]), color=col, alpha=0.08)
    ax[0].set_xlabel(r"$\log\beta$")
    ax[0].set_title(r"$\beta$: hard %.2f [%.2f, %.2f] $\to$ %.2f [%.2f, %.2f]" % (
        res["beta_hard"]["median"], res["beta_hard"]["lo"], res["beta_hard"]["hi"],
        res["beta_upgraded"]["median"], res["beta_upgraded"]["lo"], res["beta_upgraded"]["hi"]),
        fontsize=10)
    ax[0].legend(fontsize=8)
    sub = rng.choice(len(rho), size=min(6000, len(rho)), replace=False)
    xx = np.linspace(max(0.05, rho.min() * 0.8), rho.max() * 1.2, 200)
    mm = np.log([curve.beta_bar(x, Kstar) for x in xx])
    ss = np.sqrt(np.array([curve.log_scatter(x) for x in xx]) ** 2 + sig_e2)
    ax[1].fill_between(xx, np.exp(mm - 2 * ss), np.exp(mm + 2 * ss), color="0.88", zorder=0,
                       label=r"curve $\pm 2s(\rho)$")
    ax[1].plot(xx, np.exp(mm), color="0.4", lw=1, zorder=1)
    ax[1].scatter(rho[sub], np.exp(up["lb"][sub]), s=4, alpha=0.3, color="#2a6f97", zorder=2, label="upgraded")
    ax[1].scatter(rho[sub], beta_h[sub], s=4, alpha=0.6, color="#b0563f", zorder=3, label="hard curve")
    ax[1].plot(syn.rho_true, syn.beta_true, "*", ms=14, mfc="yellow", mec="k", zorder=4, label="truth")
    ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel(r"$\beta$")
    ax[1].legend(fontsize=8)
    rbins = np.linspace(rho.min(), rho.max(), 50)
    ax[2].hist(rho, rbins, density=True, alpha=0.6, color="#b0563f", label="hard curve")
    ax[2].hist(rho, rbins, weights=up["w"] * len(rho), density=True, alpha=0.6,
               color="#2a6f97", label="reweighted")
    ax[2].axvline(syn.rho_true, color="k", lw=2, label="truth")
    ax[2].set_xlabel(r"$\rho$")
    ax[2].set_title(f"weight ESS {100 * res['weight_ess_fraction']:.0f}%", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig_beta_upgrade.png"), dpi=150)
    print(open(os.path.join(td, "beta_upgrade.md")).read())


if __name__ == "__main__":
    main()
