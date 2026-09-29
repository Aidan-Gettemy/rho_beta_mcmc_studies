"""
E4 figures and tables (3-step sampler with the beta_bar(rho) curve).

    fig_e4_posteriors.png   rho (with prior), beta = beta_bar(rho), level
                            (spatial mean of log K); truth lines; 90% intervals
    fig_e4_joint.png        posterior (rho, beta) samples on the E3 curve band
    fig_e4_field.png        true log K | posterior mean | posterior sd |
                            slice y=0.53 with 5-95% band
    fig_e4_traces.png       rho, beta, level, Phi traces for all chains
    fig_e4_convergence.png  growing-window MPSRF(rho, level) and PSRF of rho,
                            beta, level, Phi, ||theta||^2, max over leading theta
    fig_e4_ppc.png          posterior predictive: pressures (mean + 90%), scaled
                            residuals vs N(0,1), predicted Keff vs K*
    fig_e4_amplitude.png    ||theta||^2 / r vs its chi^2_r prior band, and
                            rho * sqrt(||theta||^2/r) (the identified amplitude)
    fig_e4_adaptation.png   step sizes during burn-in, acceptance per move
    tables/e4_sampler/e4_stats.{md,csv}   one row per quantity: median, 90% CI,
        mean, sd, truth, z, covered, PSRF, split-R-hat, ESS
    tables/e4_sampler/e4_chains.{md,csv}  per-chain acceptance, steps, means
    tables/e4_sampler/e4_summary.json     coverage, mean Phi vs floor, MPSRF, ...
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from _common import have, load_problem, io, C
from brho import plotting as P
from brho import diagnostics as D
from brho.curve import BetaCurve
from brho.samplers import load_chains, trace_burn_index

EXP = io.EXP_SAMPLER


def load(exp):
    prob, man = load_problem(exp)
    d = io.results_dir(exp)
    paths = io.chain_files(d)
    if not paths:
        return prob, man, None, None, None
    ch = load_chains(paths)
    curve = BetaCurve.load(os.path.join(d, "curve.npz"))
    scfg = C.from_dict(C.SamplerConfig, man["configs"]["SamplerConfig"])
    return prob, man, ch, curve, scfg


def main(exp=EXP):
    if not have(exp):
        print("E4: no results"); return
    prob, man, ch, curve, scfg = load(exp)
    if ch is None:
        print(f"{exp}: set up, no chains yet"); return
    syn, kle, n, r = prob.syn, prob.kle, prob.pc.n, prob.r
    nb = scfg.n_burn
    te = scfg.trace_every                      # traces recorded every te its
    ib = trace_burn_index(nb, te)
    post = {k: ch["tr_" + k][:, ib:] for k in ("rho", "beta", "level", "phi",
                                               "theta_norm2", "w")}
    lead = ch["tr_theta_lead"][:, ib:, :]
    rho_s, beta_s = ch["kept_rho"].ravel(), ch["kept_beta"].ravel()
    th_s = ch["kept_theta"].reshape(-1, r)
    # level = spatial mean of log K, exactly; the full fields are summarized
    # a block of cells at a time below (the sample x cell matrix is GBs)
    level_s = np.log(beta_s) + rho_s * kle.spatial_mean(th_s)
    floor = prob.noise_floor(scfg.use_keff_term)
    n_ch = ch["tr_rho"].shape[0]

    # ---------------------------------------------------------- posteriors
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    s_rho = D.summarize(rho_s, syn.rho_true)
    ax[0].hist(rho_s, 40, density=True, color=P.C_BAND, edgecolor="w")
    xx = np.linspace(0.05, min(curve.rho_max, 6), 300)
    ax[0].plot(xx, P.lognormal_pdf(xx, scfg.mu_rho, scfg.s_rho), "--",
               color=P.C_PRIOR, lw=1, label="prior")
    ax[0].axvline(syn.rho_true, color="r", lw=2, label=r"$\rho^\dagger$")
    ax[0].axvspan(s_rho["lo"], s_rho["hi"], color="k", alpha=0.07, label="90%")
    ax[0].set_title(f"$\\rho$: {s_rho['median']:.3f} [{s_rho['lo']:.2f}, {s_rho['hi']:.2f}]")
    ax[0].legend()
    s_beta = D.summarize(beta_s, syn.beta_true)
    ax[1].hist(beta_s, 40, density=True, color="#7fd0a8", edgecolor="w")
    ax[1].axvline(syn.beta_true, color="r", lw=2, label=r"$\beta^\dagger$")
    ax[1].axvspan(s_beta["lo"], s_beta["hi"], color="k", alpha=0.07)
    ax[1].set_title(rf"$\beta=\bar\beta(\rho)$: {s_beta['median']:.3f} "
                    f"[{s_beta['lo']:.2f}, {s_beta['hi']:.2f}]")
    ax[1].legend()
    s_lev = D.summarize(level_s, syn.level_true)
    ax[2].hist(level_s, 40, density=True, color="#b07fd0", edgecolor="w")
    ax[2].axvline(syn.level_true, color="purple", lw=2, ls=":", label="true level")
    ax[2].set_title(f"level (mean $\\log K$): {s_lev['median']:.3f} "
                    f"[{s_lev['lo']:.2f}, {s_lev['hi']:.2f}]")
    ax[2].legend()
    P.save(fig, exp, "fig_e4_posteriors")

    # ---------------------------------------------------------- joint
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    b = curve.band(syn.keff_obs)
    ax.fill_between(b["rho"], b["q05"], b["q95"], color=P.C_BAND, alpha=0.3,
                    label=r"E3: 5–95% of $\beta_i$")
    rr = np.linspace(0, curve.rho_max, 300)
    ax.plot(rr, curve.beta_bar(rr, syn.keff_obs), color=P.C_CURVE, lw=1.5,
            label=r"$\bar\beta(\rho)$")
    ax.plot(rho_s, beta_s, ".", color=P.C_POST, ms=2, alpha=0.4, label="E4 posterior")
    P.truth_star(ax, syn.rho_true, syn.beta_true)
    ax.set_yscale("log"); ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\beta$")
    ax.set_xlim(0, min(curve.rho_max, max(3.5, rho_s.max() * 1.2)))
    ax.set_title("Posterior lives on the curve; data select the $\\rho$ segment")
    ax.legend()
    P.save(fig, exp, "fig_e4_joint")

    # ---------------------------------------------------------- field
    fs = D.logK_summaries(np.log(beta_s), rho_s, th_s, kle, syn.logK_true)
    m_lk, sd_lk, lo_lk, hi_lk, cov90 = fs["mean"], fs["sd"], fs["lo"], fs["hi"], fs["coverage"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    v0 = min(syn.logK_true.min(), m_lk.min()); v1 = max(syn.logK_true.max(), m_lk.max())
    P.field(ax[0], syn.logK_true, n, r"true $\log K$", vmin=v0, vmax=v1)
    P.field(ax[1], m_lk, n, r"posterior mean $\log K$", vmin=v0, vmax=v1)
    P.field(ax[2], sd_lk, n, r"posterior sd $\log K$", cmap="magma")
    jr = P.slice_row(n); xs = prob.phys.xc; g = prob.phys.grid
    ax[3].fill_between(xs, g(lo_lk)[:, jr], g(hi_lk)[:, jr], color=P.C_BAND,
                       alpha=0.5, label="5–95%")
    ax[3].plot(xs, g(m_lk)[:, jr], "b-", label="posterior mean")
    ax[3].plot(xs, g(syn.logK_true)[:, jr], "k-", lw=2, label="truth")
    ax[3].set_xlabel(f"x (y = {xs[jr]:.2f})")
    ax[3].set_title(f"slice; pointwise 90% coverage {cov90:.0%}")
    ax[3].legend()
    P.save(fig, exp, "fig_e4_field")

    # ---------------------------------------------------------- traces
    keys = [("rho", r"$\rho$", syn.rho_true), ("beta", r"$\beta$", syn.beta_true),
            ("level", "level", syn.level_true), ("phi", r"$\Phi$", floor)]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    th = max(1, ch["tr_rho"].shape[1] // 2000)
    for j, (k, lab, tv) in enumerate(keys):
        for c in range(n_ch):
            t = ch["tr_" + k][c]
            ax[j].plot(np.arange(0, len(t), th) * te, t[::th], lw=0.5, alpha=0.7)
        ax[j].axhline(tv, color="r", lw=1.2, ls=":")
        ax[j].axvline(nb, color="gray", ls="--", lw=1)
        ax[j].set_title(f"{lab}   PSRF={D.psrf(post[k]):.3f}")
        ax[j].set_xlabel("iteration")
        if k == "phi":
            ax[j].set_yscale("log")
    P.save(fig, exp, "fig_e4_traces")

    # ---------------------------------------------------------- convergence
    ts, mp = D.window_curve(D.mpsrf, np.stack([post["rho"], post["level"]], -1))
    curves = {k: D.window_curve(D.psrf, post[k])[1]
              for k in ("rho", "beta", "level", "phi", "theta_norm2")}
    _, th_max = D.window_curve(lambda a: max(D.psrf(a[:, :, i]) for i in range(a.shape[2])),
                               lead)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax[0].plot(ts * te + nb, mp, "k-", lw=2, label=r"MPSRF $(\rho,$ level$)$")
    ax[1].plot(ts * te + nb, th_max, color="0.5", lw=1.2,
               label=rf"max PSRF $\theta_{{1..{lead.shape[2]}}}$")
    for k, v in curves.items():
        ax[1].plot(ts * te + nb, v, lw=1.5, label=f"PSRF {k}")
    for a in ax:
        a.axhline(1.1, color="r", ls=":", lw=1.2)
        a.set_xlabel("iteration $t$ (window $[t/2,t]$ of post-burn-in)")
        a.legend(); a.set_ylim(0.95, 3.0)     # values above 3 are clipped
    ax[0].set_title(f"Brooks–Gelman MPSRF, {n_ch} chains")
    ax[1].set_title("univariate PSRF")
    P.save(fig, exp, "fig_e4_convergence")

    # ---------------------------------------------------------- PPC
    pobs = ch["kept_pobs"].reshape(-1, syn.n_obs)
    keff_s = ch["kept_keff"].ravel()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    if np.isfinite(pobs).all():
        pm = pobs.mean(0); plo, phi_ = np.percentile(pobs, [5, 95], 0)
        ax[0].errorbar(syn.d_p, pm, yerr=[pm - plo, phi_ - pm], fmt=".", ms=3,
                       elinewidth=0.6, alpha=0.7)
        lo, hi = syn.d_p.min(), syn.d_p.max()
        ax[0].plot([lo, hi], [lo, hi], "k-", lw=1)
        ax[0].set_xlabel("observed"); ax[0].set_ylabel("posterior predicted (90%)")
        res = (syn.d_p - pm) / syn.sigma_p
        ax[1].hist(res, 25, density=True, color=P.C_BAND, edgecolor="w")
        x = np.linspace(-4, 4, 200)
        ax[1].plot(x, stats.norm.pdf(x), "k--", lw=1)
        ax[1].set_title(f"scaled residuals: mean {res.mean():.2f}, sd {res.std():.2f}")
        ax[2].hist(keff_s, 40, density=True, color="#7fd0a8", edgecolor="w")
        ax[2].axvline(syn.keff_obs, color="r", lw=2, label="$K^*$ (measured)")
        ax[2].axvline(syn.keff_true, color="k", ls=":", label="$K_{eff}$ true")
        ax[2].set_title("predicted $K_{eff}$ (not in likelihood)"
                        if not scfg.use_keff_term else "predicted $K_{eff}$")
        ax[2].legend()
    P.save(fig, exp, "fig_e4_ppc")

    # ---------------------------------------------------------- amplitude
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    tn = post["theta_norm2"].ravel() / r
    ax[0].hist(tn, 40, density=True, color=P.C_BAND, edgecolor="w", label="posterior")
    x = np.linspace(max(0.01, tn.min() * 0.8), tn.max() * 1.2, 300)
    ax[0].plot(x, r * stats.chi2.pdf(x * r, r), "k--", lw=1, label=r"prior $\chi^2_r/r$")
    ax[0].set_title(r"$\|\theta\|^2/r$"); ax[0].legend()
    amp = (post["rho"] * np.sqrt(post["theta_norm2"] / r)).ravel()
    ax[1].hist(amp, 40, density=True, color="#e0a060", edgecolor="w")
    ax[1].axvline(syn.rho_true * syn.G_std, color="r", lw=2,
                  label=r"$\rho^\dagger\,$std$(G_{true})$")
    ax[1].set_title(r"$\rho\,\|\theta\|/\sqrt{r}$ (field amplitude, ridge-invariant)")
    ax[1].legend()
    P.save(fig, exp, "fig_e4_amplitude")

    # ---------------------------------------------------------- adaptation
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = ["A: $s_\\theta$", "B: $s_w$", "C: $s_{rg}$"]
    for c in range(n_ch):
        sh = ch["s_hist"][c]
        for j in range(3):
            ax[0].semilogy((np.arange(len(sh)) + 1) * scfg.adapt_every, sh[:, j],
                           color=f"C{j}", lw=0.8, label=names[j] if c == 0 else None)
    ax[0].set_title("step sizes during burn-in (frozen after)")
    ax[0].set_xlabel("iteration"); ax[0].legend()
    it_tot = ch["it"][:, None].astype(float)
    acc = ch["acc_tot"] / it_tot
    for j in range(3):
        ax[1].bar(np.arange(n_ch) + 0.25 * (j - 1), acc[:, j], 0.25,
                  color=f"C{j}", label=names[j][:1])
        ax[1].axhline(scfg.targets[j], color=f"C{j}", ls=":", lw=1)
    ax[1].set_xlabel("chain"); ax[1].set_ylabel("overall acceptance")
    ax[1].set_title("acceptance per move (dotted: target)"); ax[1].legend()
    P.save(fig, exp, "fig_e4_adaptation")

    # ---------------------------------------------------------- tables
    def row(name, samples, trace, truth):
        s = D.summarize(samples, truth)
        return dict(quantity=name, median=s["median"], q05=s["lo"], q95=s["hi"],
                    mean=s["mean"], sd=s["sd"], truth=truth, z=s.get("z"),
                    covered=s.get("covered"), psrf=D.psrf(trace),
                    split_rhat=D.split_rhat(trace), ess=D.ess_multi(trace))
    rows = [row("rho", rho_s, post["rho"], syn.rho_true),
            row("beta", beta_s, post["beta"], syn.beta_true),
            row("level", level_s, post["level"], syn.level_true),
            row("Phi", post["phi"].ravel(), post["phi"], floor),
            row("theta_norm2/r", tn, post["theta_norm2"] / r, 1.0)]
    for i in range(lead.shape[2]):
        rows.append(row(f"theta_{i + 1}", lead[:, :, i].ravel(), lead[:, :, i], None))
    io.write_table(exp, "e4_stats", rows)
    crow = [dict(chain=c, acc_A=float(acc[c, 0]), acc_B=float(acc[c, 1]),
                 acc_C=float(acc[c, 2]), s_theta=float(ch["s_final"][c, 0]),
                 s_w=float(ch["s_final"][c, 1]), s_rg=float(ch["s_final"][c, 2]),
                 rho_mean=float(post["rho"][c].mean()),
                 level_mean=float(post["level"][c].mean()),
                 phi_mean=float(post["phi"][c].mean()))
            for c in range(n_ch)]
    io.write_table(exp, "e4_chains", crow)
    summ = dict(n_chains=n_ch, n_total=scfg.n_total, n_burn=nb, thin=scfg.thin,
                n_kept=len(rho_s), r=r, Kstar=syn.keff_obs,
                curve_statistic=curve.statistic, noise_floor=floor,
                mean_phi=float(post["phi"].mean()), coverage90_logK=cov90,
                mpsrf_final=float(mp[-1]), psrf_theta_lead_max=float(th_max[-1]),
                rho=s_rho, beta=s_beta, level=s_lev,
                info_limit_rel_sd=float(1 / np.sqrt(2 * r)),
                rho_rel_sd=s_rho["sd"] / s_rho["mean"],
                minutes=man.get("minutes"), status=man.get("status"))
    io.save_json(io.tables_dir(exp) + "/e4_summary.json", summ)
    print("E4:", {k: summ[k] for k in ("mean_phi", "noise_floor", "coverage90_logK",
                                      "mpsrf_final")},
          "rho", round(s_rho["median"], 3), "beta", round(s_beta["median"], 3))


if __name__ == "__main__":
    main()
