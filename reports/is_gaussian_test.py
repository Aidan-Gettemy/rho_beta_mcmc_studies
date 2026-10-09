"""
Best-case importance-sampling test: how much ESS per full forward solve can a
Gaussian proposal fitted to the true posterior give?

Any proposal built from cheaper (local, subdomain) samplers only approximates
the posterior, so a Gaussian fitted to the curve-sampler posterior itself is
the best a Gaussian proposal can do.  If even this one gives a small ESS per
solve, a proposal assembled from local samplers will do worse.

Target: the curve-sampler (E4) posterior in x = (theta, w), d = r + 1,

    log pi(x) = -Phi(theta, w) - |theta|^2 / 2 - w^2 / 2 + const,
    log rho = mu_rho + s_rho w,  beta = beta_bar(rho)   (ThreeStepSampler.potential),

Proposal: q = N(m, c * S), (m, S) the mean and covariance of the kept samples
of the run, c the inflation (default 1.2; several values give several tests).
Fresh iid draws from q, one PDE solve each, self-normalised weights
w_i ~ pi(x_i) / q(x_i).

Reported per inflation:
  ESS / N        Kish (sum w)^2 / sum w^2
  Pareto k-hat   tail shape of the weights (Vehtari et al., PSIS; Zhang &
                 Stephens 2009 fit).  < 0.5 good, 0.5-0.7 usable, > 0.7 the
                 estimate is unreliable however large N is.
  IS estimates of rho, beta, level against the MCMC values.
and, for comparison, the run's own ESS per solve (rho, beta, level, worst theta_i).

    BRHO_ROOT=... python reports/is_gaussian_test.py --exp e4_sampler_ab [--n 500] [--inflate 1.2]

Cost: N PDE solves per inflation value (500 x ~2 ms at 32x32).  Needs the
chain files of the run.  Writes tables/<exp>/is_gaussian.{json,md} and
figures/<exp>/fig_is_gaussian.png.
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from _common import have, load_problem, io, C
from brho.curve import BetaCurve
from brho.samplers import ThreeStepSampler
from brho import diagnostics as D


def pareto_khat(logw):
    """PSIS tail-shape diagnostic k-hat of the importance weights (Zhang &
    Stephens 2009 generalized-Pareto fit with the weak prior used in PSIS)."""
    lw = np.sort(np.asarray(logw, float))
    lw = lw[np.isfinite(lw)]
    n = len(lw)
    M = int(min(n // 5, np.ceil(3 * np.sqrt(n))))
    if M < 5:
        return np.nan
    w = np.exp(lw - lw[-1])
    x = w[-M:] - w[-M - 1]                     # exceedances over the cutoff
    x = np.sort(x[x > 0])
    if len(x) < 5:
        return np.nan
    m = len(x)
    m_est = 30 + int(np.sqrt(m))
    b = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    b /= 3 * x[int(m / 4 + 0.5) - 1]
    b += 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    L = m * (np.log(-(b / k)) - k - 1)
    wt = 1 / np.exp(L - L[:, None]).sum(axis=1)
    b_post = float(np.sum(b * wt / wt.sum()))
    k_post = float(np.log1p(-b_post * x).mean())
    return (m * k_post + 10 * 0.5) / (m + 10)


def kish(logw):
    lw = np.asarray(logw, float)
    lw = lw[np.isfinite(lw)]
    if len(lw) == 0:
        return 0.0
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (w @ w))


def wquant(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(q, cw, x[o])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="e4_sampler_ab")
    ap.add_argument("--n", type=int, default=500, help="draws (PDE solves) per inflation")
    ap.add_argument("--inflate", type=float, nargs="+", default=[1.2],
                    help="proposal covariance = inflate * posterior covariance")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    exp = a.exp
    if not have(exp):
        print(f"{exp}: no results"); return
    prob, man = load_problem(exp)
    d = io.results_dir(exp)
    paths = io.chain_files(d)
    if not paths:
        print(f"{exp}: no chain files in {d}"); return
    scfg = C.from_dict(C.SamplerConfig, man["configs"]["SamplerConfig"])
    curve = BetaCurve.load(os.path.join(d, "curve.npz"))
    smp = ThreeStepSampler(prob, curve, scfg)
    r = prob.r
    mu_r, s_r = scfg.mu_rho, scfg.s_rho

    # ---- posterior samples of the run -> Gaussian fit, and the run's own ESS
    TH, RHO, BETA = [], [], []
    for p in paths:
        with np.load(p) as z:
            rho = z["kept_rho"].ravel()
            TH.append(z["kept_theta"].reshape(len(rho), r))
            RHO.append(rho); BETA.append(z["kept_beta"].ravel())
    L = min(len(x) for x in RHO)                      # equal length for ESS
    TH = np.stack([t[:L] for t in TH]); RHO = np.stack([x[:L] for x in RHO])
    BETA = np.stack([x[:L] for x in BETA])
    W = (np.log(RHO) - mu_r) / s_r
    LEV = np.log(BETA) + RHO * prob.kle.spatial_mean(TH)         # level = spatial mean of log K
    X = np.concatenate([TH, W[..., None]], axis=2).reshape(-1, r + 1)
    if len(X) < 20 * (r + 1):
        print(f"{exp}: only {len(X)} kept samples for a covariance in d = {r + 1}; "
              f"need a longer run"); return
    m = X.mean(0)
    S = np.cov(X, rowvar=False)
    n_ch = len(paths)
    solves_run = n_ch * scfg.n_total * len(getattr(scfg, "moves", "ABC"))
    ess_run = dict(rho=D.ess_multi(RHO), beta=D.ess_multi(BETA), level=D.ess_multi(LEV),
                   theta_min=min(D.ess_multi(TH[:, :, i]) for i in range(r)))
    mc = dict(rho=(float(RHO.mean()), float(RHO.std())),
              beta=(float(BETA.mean()), float(BETA.std())),
              level=(float(LEV.mean()), float(LEV.std())))
    print(f"{exp}: {n_ch} chains x {L} kept samples; fit N(m, S) in d = {r + 1}; "
          f"run used {solves_run:,} solves")

    rng = np.random.default_rng(a.seed)
    prob.phys.warmup()
    out, draws = [], {}
    for c in a.inflate:
        Lc = np.linalg.cholesky(c * S)
        Z = rng.standard_normal((a.n, r + 1))
        Xq = m + Z @ Lc.T
        logq = -0.5 * (Z * Z).sum(1) - np.log(np.diag(Lc)).sum()     # up to a constant
        logp = np.empty(a.n); rho_q = np.exp(mu_r + s_r * Xq[:, -1])
        beta_q = np.full(a.n, np.nan); lev_q = np.full(a.n, np.nan)
        t0 = time.time()
        for i in range(a.n):
            th, w = Xq[i, :r], float(Xq[i, -1])
            phi = smp.potential(th, w)[0]
            logp[i] = -phi - 0.5 * th @ th - 0.5 * w * w if np.isfinite(phi) else -np.inf
            b = smp.beta(rho_q[i])
            if np.isfinite(b):
                beta_q[i] = b; lev_q[i] = smp.level(th, w)
        logw = logp - logq
        ok = np.isfinite(logw)
        wn = np.zeros(a.n); wn[ok] = np.exp(logw[ok] - logw[ok].max()); wn /= wn.sum()
        ess = kish(logw); kh = pareto_khat(logw)
        est = {}
        for name, v in (("rho", rho_q), ("beta", beta_q), ("level", lev_q)):
            g = np.isfinite(v) & ok
            mu = float(np.sum(wn[g] * v[g]) / wn[g].sum())
            sd = float(np.sqrt(np.sum(wn[g] * (v[g] - mu) ** 2) / wn[g].sum()))
            est[name] = dict(mean=mu, sd=sd, mcmc_mean=mc[name][0], mcmc_sd=mc[name][1],
                             z_vs_mcmc=(mu - mc[name][0]) / (mc[name][1] / np.sqrt(max(ess, 1))),
                             q05=float(wquant(v[g], wn[g], 0.05)),
                             q95=float(wquant(v[g], wn[g], 0.95)))
        res = dict(inflate=c, n=a.n, ess=ess, ess_frac=ess / a.n, khat=kh,
                   max_weight=float(wn.max()), outside_curve=int((~ok).sum()),
                   ess_per_solve=ess / a.n, seconds=time.time() - t0, estimates=est)
        out.append(res); draws[c] = (logw, rho_q, wn)
        print(f"inflate {c:g}: ESS {ess:.1f} / {a.n} ({100 * ess / a.n:.2f}%), k-hat {kh:.2f}, "
              f"max weight {wn.max():.3f}, {res['seconds']:.0f} s")

    base = {k: v / solves_run for k, v in ess_run.items()}
    R = dict(exp=exp, dim=r + 1, n_chains=n_ch, kept_per_chain=L, run_solves=solves_run,
             run_ess=ess_run, run_ess_per_solve=base, mcmc=mc, sigma_rel=prob.sigma_rel,
             tests=out)
    td, fd = io.tables_dir(exp), io.figures_dir(exp)
    with open(os.path.join(td, "is_gaussian.json"), "w") as f:
        json.dump(io._jsonable(R), f, indent=1)
    lines = [f"Best-case IS test, {exp}: proposal N(m, c S) fitted to {n_ch} x {L} kept samples, "
             f"d = {r + 1}\n",
             "| inflation c | N solves | ESS | ESS / solve | Pareto k-hat | max weight | "
             "rho IS mean (MCMC) | beta IS mean (MCMC) | level IS mean (MCMC) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in out:
        e = t["estimates"]
        lines.append(f"| {t['inflate']:g} | {t['n']} | {t['ess']:.1f} | {t['ess_per_solve']:.2e} | "
                     f"{t['khat']:.2f} | {t['max_weight']:.3f} | "
                     + " | ".join(f"{e[k]['mean']:.3f} ({e[k]['mcmc_mean']:.3f})"
                                  for k in ("rho", "beta", "level")) + " |")
    lines += ["", f"Curve sampler run: {solves_run:,} solves; ESS per solve "
              + ", ".join(f"{k} {v:.2e}" for k, v in base.items()),
              "", "k-hat < 0.5 good, 0.5-0.7 usable, > 0.7 unreliable (Vehtari et al., PSIS)."]
    with open(os.path.join(td, "is_gaussian.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figure: log-weight spread, weight concentration, rho marginal
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = ["#2a6f97", "#b0563f", "#5b8c5a", "#7a5195"]
    for j, c in enumerate(a.inflate):
        logw, rho_q, wn = draws[c]
        lw = logw[np.isfinite(logw)]
        col = cols[j % len(cols)]
        ax[0].hist(lw - lw.max(), bins=40, alpha=0.5, color=col, label=f"c = {c:g}")
        sw = np.sort(wn)[::-1]
        ax[1].plot(np.arange(1, len(sw) + 1), np.cumsum(sw), color=col, lw=2,
                   label=f"c = {c:g}: ESS {kish(logw):.0f}/{a.n}, k̂ {pareto_khat(logw):.2f}")
        ax[2].hist(rho_q, bins=40, weights=wn, density=True, histtype="step", lw=2,
                   color=col, label=f"IS, c = {c:g}")
    ax[2].hist(RHO.ravel(), bins=60, density=True, color="0.75", label="curve sampler")
    ax[2].axvline(prob.syn.rho_true, color="k", ls="--", lw=1)
    ax[0].set_xlabel("log weight − max"); ax[0].set_title("log importance weights", loc="left",
                                                           fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("number of largest weights")
    ax[1].set_ylabel("share of total weight"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title("weight concentration", loc="left", fontsize=10); ax[1].legend(fontsize=8)
    ax[2].set_xlabel("ρ"); ax[2].set_title("ρ: weighted proposal vs MCMC", loc="left",
                                           fontsize=10)
    ax[2].legend(fontsize=8)
    fig.suptitle(f"Best-case Gaussian importance sampling, d = {r + 1}, σ_rel = "
                 f"{prob.sigma_rel:.3g}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_is_gaussian.png"), dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
