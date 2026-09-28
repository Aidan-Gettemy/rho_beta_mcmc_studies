"""
E5 prior test: the 4-move sampler with Phi = 0 must reproduce the prior
    theta ~ N(0, I_r),  w ~ N(0, 1) truncated at the curve's rho_max,
    xi_b = (log beta - m(rho)) / s(rho) ~ N(0, 1),  all independent.
Moves B-D carry the curve-prior ratio, C the Jacobian, D the swap: with the
likelihood off they must reproduce the prior exactly.

    fig_e5_prior_test.png  (a) sorted diag cov(theta) vs direct draws
                           (b) QQ of w vs truncated N(0,1)
                           (c) QQ of xi_b vs N(0,1)
                           (d) rho histogram vs truncated lognormal prior
    tables/e5_prior_test/prior_test.{md,json}
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from _common import have, io
from report_soft import load
from brho import plotting as P
from brho.samplers import trace_burn_index

EXP = io.EXP_SOFT_PRIOR


def main():
    if not have(EXP):
        print("E5 prior test: no results"); return
    prob, man, ch, curve, scfg = load(EXP)
    if ch is None:
        print("E5 prior test: set up, no chains yet"); return
    r = prob.r
    ib = trace_burn_index(scfg.n_burn, scfg.trace_every)
    th = ch["kept_theta"].reshape(-1, r)
    w = ch["tr_w"][:, ib:].ravel()
    xi = ch["tr_xi_b"][:, ib:].ravel()
    rho = ch["tr_rho"][:, ib:].ravel()
    w_max = (np.log(curve.rho_max) - scfg.mu_rho) / scfg.s_rho
    tn = stats.truncnorm(-np.inf, w_max)
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(th.shape)
    cov_c, cov_r = np.cov(th.T), np.cov(ref.T)
    frob_c = np.linalg.norm(cov_c - np.eye(r)) / np.sqrt(r)
    frob_r = np.linalg.norm(cov_r - np.eye(r)) / np.sqrt(r)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.0))
    ax[0].plot(np.sort(np.diag(cov_r))[::-1], "k-", lw=2, label="direct draws")
    ax[0].plot(np.sort(np.diag(cov_c))[::-1], lw=1.5, label="chain")
    ax[0].axhline(1, color="0.5", ls=":")
    ax[0].set_title(r"sorted diag cov($\theta$) (target 1)"); ax[0].legend()
    q = np.linspace(0.005, 0.995, 100)
    ax[1].plot(tn.ppf(q), np.quantile(w, q), ".", ms=4)
    ax[1].plot(tn.ppf(q), tn.ppf(q), "k-", lw=1)
    ax[1].set_title(r"QQ: $w$ vs truncated N(0,1)")
    ax[2].plot(stats.norm.ppf(q), np.quantile(xi, q), ".", ms=4, color="C2")
    ax[2].plot(stats.norm.ppf(q), stats.norm.ppf(q), "k-", lw=1)
    ax[2].set_title(r"QQ: $\xi_\beta=(\log\beta-m(\rho))/s(\rho)$ vs N(0,1)")
    ax[3].hist(rho, 60, density=True, color=P.C_BAND, edgecolor="w", label="chain")
    xx = np.linspace(0.01, curve.rho_max, 400)
    ax[3].plot(xx, P.lognormal_pdf(xx, scfg.mu_rho, scfg.s_rho) / tn.cdf(w_max) * 1.0,
               "k--", lw=1.2, label="prior (truncated)")
    ax[3].set_title(r"$\rho$"); ax[3].legend()
    P.save(fig, EXP, "fig_e5_prior_test")

    acc = ch["acc_tot"].sum(0) / (scfg.n_total * ch["acc_tot"].shape[0])
    out = dict(n_samples=int(th.shape[0]), frob_chain=float(frob_c),
               frob_reference=float(frob_r), w_mean=float(w.mean()), w_sd=float(w.std()),
               w_mean_target=float(tn.mean()), w_sd_target=float(tn.std()),
               xi_mean=float(xi.mean()), xi_sd=float(xi.std()),
               corr_w_xi=float(np.corrcoef(w, xi)[0, 1]),
               theta_norm2_over_r=float((th ** 2).sum(1).mean() / r),
               ks_w_p=float(stats.kstest(w, tn.cdf).pvalue),
               ks_xi_p=float(stats.kstest(xi, "norm").pvalue),
               acc=dict(zip("ABCD", map(float, acc))))
    io.save_json(os.path.join(io.tables_dir(EXP), "prior_test.json"), out)
    with open(os.path.join(io.tables_dir(EXP), "prior_test.md"), "w") as f:
        f.write("| check | chain | target |\n|---|---|---|\n")
        f.write(f"| w mean / sd | {out['w_mean']:.3f} / {out['w_sd']:.3f} | "
                f"{out['w_mean_target']:.3f} / {out['w_sd_target']:.3f} |\n")
        f.write(f"| xi_b mean / sd | {out['xi_mean']:.3f} / {out['xi_sd']:.3f} | 0 / 1 |\n")
        f.write(f"| corr(w, xi_b) | {out['corr_w_xi']:.3f} | 0 |\n")
        f.write(f"| theta_norm2 / r | {out['theta_norm2_over_r']:.3f} | 1 |\n")
        f.write(f"| cov(theta) error (Frobenius / sqrt r) | {frob_c:.3f} | "
                f"{frob_r:.3f} (direct draws) |\n")
        f.write(f"| acceptance A-D | " + "/".join(f"{v:.2f}" for v in acc) + " | "
                + "/".join(f"{t:.2f}" for t in scfg.targets) + " (targets) |\n")
    print("E5 prior test:", {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in out.items()})


if __name__ == "__main__":
    main()
