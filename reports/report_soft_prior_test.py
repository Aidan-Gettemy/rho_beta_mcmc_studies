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
                           (e) law of the spatial mean of G = T theta
                           (f) law of the empirical variance of G
                           [(e)-(f): the multiscale prior test of Pereira,
                           "Variance control in distribution", Sec. 3]
    fig_e5_prior_fields.png  pointwise check of G = T theta, rows direct
                           draws / chain: three thinned sample fields (last
                           kept sample of different chains), pointwise mean,
                           pointwise variance, and variance / target with
                           target = diag(T T^T); subdomain interfaces dashed
                           for a multiscale innovation
    tables/<exp>/prior_test.{md,json}

    python reports/report_soft_prior_test.py [--exp e5_prior_test_ms_white]

KS p-values assume independent samples; chain samples are autocorrelated,
so they are indicative only (small p is expected even for a correct chain
of modest length).  The decisive checks are the cov(theta) error against
direct draws of the same size and the field-law summaries.
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from _common import have, io
from report_soft import load
from brho import plotting as P
from brho import diagnostics as D
from brho.samplers import trace_burn_index

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=io.EXP_SOFT_PRIOR,
                    help="e5_prior_test (global), e5_prior_test_ms_raw, e5_prior_test_ms_white")
    EXP = ap.parse_args(argv).exp
    if not have(EXP):
        print(f"{EXP}: no results"); return
    prob, man, ch, curve, scfg = load(EXP)
    if ch is None:
        print(f"{EXP}: set up, no chains yet"); return
    innov = getattr(scfg, "theta_innovation", "global")
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

    # field laws (Pereira's prior test): G = T theta, chain vs direct draws
    G_c, G_r = prob.kle.field(th), prob.kle.field(ref)
    mean_c, mean_r = G_c.mean(1), G_r.mean(1)
    var_c, var_r = G_c.var(1), G_r.var(1)

    fig, axs = plt.subplots(2, 3, figsize=(16, 8.4))
    ax = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]
    fig.suptitle(f"{EXP}: prior test (likelihood off), theta innovation = {innov}")
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
    lo, hi = np.quantile(np.r_[mean_r, mean_c], [0.001, 0.999])
    bins = np.linspace(lo, hi, 50)
    ax[4].hist(mean_r, bins, density=True, histtype="step", color="k", lw=2, label="direct draws")
    ax[4].hist(mean_c, bins, density=True, histtype="step", lw=1.8, label="chain")
    ax[4].set_title(f"spatial mean of G: sd {mean_c.std():.3f} (direct {mean_r.std():.3f})")
    ax[4].legend()
    lo, hi = np.quantile(np.r_[var_r, var_c], [0.001, 0.999])
    bins = np.linspace(lo, hi, 50)
    ax[5].hist(var_r, bins, density=True, histtype="step", color="k", lw=2, label="direct draws")
    ax[5].hist(var_c, bins, density=True, histtype="step", lw=1.8, label="chain")
    ax[5].set_title(f"empirical variance of G: {var_c.mean():.3f}±{var_c.std():.3f} "
                    f"(direct {var_r.mean():.3f}±{var_r.std():.3f})", fontsize=10)
    ax[5].legend()
    fig.tight_layout()
    P.save(fig, EXP, "fig_e5_prior_test")

    # pointwise fields: direct draws vs chain (exact moments from theta)
    target = (prob.kle.T ** 2).sum(1)
    kth = np.asarray(ch["kept_theta"])               # (chains, kept, r)
    show_c = kth[:3, -1] if kth.shape[0] >= 3 else kth[0, -3:]
    pm_r, pv_r = D.pointwise_moments(ref, prob.kle.T)
    pm_c, pv_c = D.pointwise_moments(th, prob.kle.T)
    blocks = (scfg.block_n if innov.startswith("block") else
              getattr(scfg, "ms_blocks", None) if innov != "global" else None)
    figf = P.prior_fields([("direct draws", prob.kle.field(ref[:3]), pm_r, pv_r),
                           (f"chain ({innov})", prob.kle.field(show_c), pm_c, pv_c)],
                          target, prob.pc.n, blocks,
                          f"{EXP}: pointwise fields of G after prior sampling "
                          "(likelihood off)" + (f", {blocks}x{blocks} subdomains (dashed)."
                                                if blocks else "."))
    ratio_c = figf.ratios[f"chain ({innov})"]
    P.save(figf, EXP, "fig_e5_prior_fields")

    acc = ch["acc_tot"].sum(0) / (scfg.n_total * ch["acc_tot"].shape[0])
    if "acc_tot_prop" in ch:                         # per proposal (E8 random schedule)
        acc = ch["acc_tot"].sum(0) / np.maximum(ch["acc_tot_prop"].sum(0), 1)
    out = dict(n_samples=int(th.shape[0]), frob_chain=float(frob_c),
               frob_reference=float(frob_r), w_mean=float(w.mean()), w_sd=float(w.std()),
               w_mean_target=float(tn.mean()), w_sd_target=float(tn.std()),
               xi_mean=float(xi.mean()), xi_sd=float(xi.std()),
               corr_w_xi=float(np.corrcoef(w, xi)[0, 1]),
               theta_norm2_over_r=float((th ** 2).sum(1).mean() / r),
               ks_w_p=float(stats.kstest(w, tn.cdf).pvalue),
               ks_xi_p=float(stats.kstest(xi, "norm").pvalue),
               acc=dict(zip("ABCD", map(float, acc))),
               theta_innovation=innov,
               theta_var_sorted_max_min=[float(np.diag(cov_c).max()),
                                         float(np.diag(cov_c).min())],
               spatial_mean_sd=float(mean_c.std()),
               spatial_mean_sd_reference=float(mean_r.std()),
               field_var_mean_sd=[float(var_c.mean()), float(var_c.std())],
               field_var_mean_sd_reference=[float(var_r.mean()), float(var_r.std())],
               ks_spatial_mean_p=float(stats.ks_2samp(mean_c, mean_r).pvalue),
               ks_field_var_p=float(stats.ks_2samp(var_c, var_r).pvalue),
               pointwise_var_ratio_min=float(ratio_c.min()),
               pointwise_var_ratio_max=float(ratio_c.max()),
               pointwise_mean_max_abs=float(np.abs(pm_c).max()),
               pointwise_mean_max_abs_reference=float(np.abs(pm_r).max()))
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
        f.write(f"| spatial mean of G: sd | {out['spatial_mean_sd']:.3f} | "
                f"{out['spatial_mean_sd_reference']:.3f} (direct draws) |\n")
        f.write(f"| empirical variance of G: mean / sd | {var_c.mean():.3f} / "
                f"{var_c.std():.3f} | {var_r.mean():.3f} / {var_r.std():.3f} (direct draws) |\n")
        f.write(f"| pointwise variance / target (min–max) | {ratio_c.min():.2f}–"
                f"{ratio_c.max():.2f} | 1 |\n")
        f.write(f"| max abs pointwise mean of G | {np.abs(pm_c).max():.3f} | "
                f"{np.abs(pm_r).max():.3f} (direct draws) |\n")
        f.write(f"| acceptance A-D | " + "/".join(f"{v:.2f}" for v in acc) + " | "
                + "/".join(f"{t:.2f}" for t in scfg.targets) + " (targets) |\n")
    print(f"{EXP}:", {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in out.items()})


if __name__ == "__main__":
    main()
