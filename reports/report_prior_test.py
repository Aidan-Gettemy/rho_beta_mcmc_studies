"""
E4 prior test: the 3-step sampler with Phi = 0 must reproduce the prior
(theta ~ N(0, I_r), w ~ N(0,1) truncated at the curve's rho_max).  This checks
the ridge-move Jacobian (-r eta) and the pCN blocks without any PDE solve.

    fig_prior_test.png   (a) sorted diag of cov(theta) vs direct draws
                         (b) law of the spatial mean of G(theta)
                         (c) rho histogram vs (truncated) lognormal prior
                         (d) QQ plot of w vs truncated N(0,1)
    tables/e4_prior_test/prior_test.{md,csv,json}
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from _common import have, io
from report_sampler import load
from brho import plotting as P
from brho.samplers import trace_burn_index

EXP = io.EXP_PRIOR


def main():
    if not have(EXP):
        print("prior test: no results"); return
    prob, man, ch, curve, scfg = load(EXP)
    if ch is None:
        print("prior test: set up, no chains yet"); return
    kle, r = prob.kle, prob.r
    nb = scfg.n_burn
    th = ch["kept_theta"].reshape(-1, r)
    w = ch["tr_w"][:, trace_burn_index(nb, scfg.trace_every):].ravel()
    rho = ch["kept_rho"].ravel()
    ref = np.random.default_rng(7).standard_normal(th.shape)
    wmax = (np.log(curve.rho_max) - scfg.mu_rho) / scfg.s_rho
    trunc = stats.truncnorm(-np.inf, wmax)

    def frob(x):
        return float(np.linalg.norm(np.cov(x.T) - np.eye(r)) / np.sqrt(r))

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    ax[0].plot(np.sort(np.diag(np.cov(ref.T)))[::-1], "k-", lw=2, label="direct draws")
    ax[0].plot(np.sort(np.diag(np.cov(th.T)))[::-1], "-", color="C0", label="chain")
    ax[0].axhline(1, color="0.5", ls=":")
    ax[0].set_title(r"sorted diag cov$(\theta)$ (target 1)"); ax[0].legend()
    gm_c, gm_r = kle.spatial_mean(th), kle.spatial_mean(ref)
    bins = np.linspace(min(gm_c.min(), gm_r.min()), max(gm_c.max(), gm_r.max()), 40)
    ax[1].hist(gm_r, bins, density=True, alpha=0.5, color="k", label="direct")
    ax[1].hist(gm_c, bins, density=True, alpha=0.5, color="C0", label="chain")
    ax[1].set_title(rf"spatial mean of $G$: sd {gm_c.std():.3f} vs {gm_r.std():.3f}")
    ax[1].legend()
    ax[2].hist(rho, 50, density=True, color=P.C_BAND, edgecolor="w", label="chain")
    xx = np.linspace(0.02, curve.rho_max, 300)
    ax[2].plot(xx, P.lognormal_pdf(xx, scfg.mu_rho, scfg.s_rho) / stats.norm.cdf(wmax),
               "k--", label="prior (truncated)")
    ax[2].set_title(r"$\rho$"); ax[2].legend()
    q = np.linspace(0.005, 0.995, 100)
    ax[3].plot(trunc.ppf(q), np.quantile(w, q), "o", ms=3)
    ax[3].plot([-3, wmax], [-3, wmax], "k-", lw=1)
    ax[3].set_title("QQ: $w$ vs truncated N(0,1)")
    P.save(fig, EXP, "fig_prior_test")

    res = dict(n_samples=len(th), frob_chain=frob(th), frob_reference=frob(ref),
               gbar_sd_chain=float(gm_c.std()), gbar_sd_reference=float(gm_r.std()),
               w_mean=float(w.mean()), w_sd=float(w.std()),
               w_mean_target=float(trunc.mean()), w_sd_target=float(trunc.std()),
               ks_w_p=float(stats.kstest(w[::max(1, scfg.thin)], trunc.cdf).pvalue),
               ks_gbar_p=float(stats.ks_2samp(gm_c, gm_r).pvalue),
               prior_mass_above_rho_max=float(1 - stats.norm.cdf(wmax)),
               acc=(ch["acc_tot"] / ch["it"][:, None]).mean(0).tolist())
    io.write_table(EXP, "prior_test", [{k: v for k, v in res.items() if k != "acc"}])
    io.save_json(io.tables_dir(EXP) + "/prior_test.json", res)
    print("prior test:", {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in res.items()})


if __name__ == "__main__":
    main()
