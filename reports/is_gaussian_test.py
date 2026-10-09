"""
Best-case importance-sampling test: how much ESS per full forward solve can a
Gaussian proposal fitted to the true posterior give, and how well does that
Gaussian describe the posterior samples in the first place?

Any proposal built from cheaper (local, subdomain) samplers only approximates
the posterior, so a Gaussian fitted to the curve-sampler posterior itself is
the best a Gaussian proposal can do.  If even this one gives a small ESS per
solve, a proposal assembled from local samplers will do worse.

Target: the curve-sampler (E4) posterior (ThreeStepSampler.potential),

    log pi(theta, w) = -Phi(theta, w) - |theta|^2 / 2 - w^2 / 2 + const,
    log rho = mu_rho + s_rho w,  beta = beta_bar(rho),  log K = log beta + rho T theta.

Two coordinate systems (--coords):
    theta   x = (theta, w)                       the sampler's coordinates
    u       x = (u, w),  u = rho * theta         field coordinates: log K = log beta_bar(rho) + T u.
            The data pin the field, so the ridge theta ~ u / rho (curved in theta)
            is close to straight in u.  Density: pi_u(u, w) = pi(u / rho, w) rho^-r.

Proposal: q = N(m, c S), (m, S) the mean and covariance of the kept samples in
the chosen coordinates, c the inflation.  Fresh iid draws from q, one PDE solve
each, self-normalised weights pi(x_i) / q(x_i).

Fit diagnostics, no PDE solves (the kept samples carry their own Phi):
  log pi vs log q over the posterior samples   a perfect fit is a line of slope 1;
                                               sd of the log ratio measures the misfit
  predicted ESS / N = 1 / (E_pi[r] E_pi[1/r]),  r = pi / q   (exact identity:
                                               E_q[r^2] / E_q[r]^2 = E_pi[r] E_pi[1/r])
  Mahalanobis D^2 of the samples vs chi^2_d    the Gaussian's prediction
  most curved direction                        the eigen-direction of S whose
                                               posterior mean bends most with w,
                                               against the Gaussian's straight line

IS results per (coords, inflation): Kish ESS / N, Pareto k-hat (Vehtari et al.,
PSIS; < 0.5 good, 0.5-0.7 usable, > 0.7 unreliable however large N), Phi of the
draws, IS estimates of rho, beta, level against the MCMC values; and the run's
own ESS per solve.

    BRHO_ROOT=... python reports/is_gaussian_test.py --exp e4_sampler_ab \
        [--coords theta u] [--n 500] [--inflate 1.2]

Cost: N PDE solves per (coords, inflation) pair (~2 ms each at 32x32).  Needs
the chain files of the run.  Writes tables/<exp>/is_gaussian.{json,md},
figures/<exp>/fig_is_gaussian.png (weights) and fig_is_gaussian_fit.png (fit).
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

COORDS = ("theta", "u")
COL = {"theta": "#b0563f", "u": "#2a6f97"}
LABEL = {"theta": "θ coordinates", "u": "field coordinates u = ρθ"}


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


def logmeanexp(x):
    x = np.asarray(x, float)
    mx = x.max()
    return float(mx + np.log(np.mean(np.exp(x - mx))))


def wquant(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(q, cw, x[o])


def to_coords(kind, theta, rho):
    """(theta, rho) -> field part of x."""
    return theta if kind == "theta" else rho[..., None] * theta


def jac(kind, rho, r):
    """log |d theta / d x| added to log pi in these coordinates."""
    return 0.0 if kind == "theta" else -r * np.log(rho)


class Gauss:
    def __init__(self, m, S):
        self.m, self.S = m, S
        self.L = np.linalg.cholesky(S)
        self.logdet = 2 * np.log(np.diag(self.L)).sum()

    def maha2(self, X):
        Y = np.linalg.solve(self.L, (X - self.m).T)
        return (Y * Y).sum(0)

    def logpdf(self, X, c=1.0):                  # N(m, c S), without the 2 pi constant
        d = len(self.m)
        return -0.5 * self.maha2(X) / c - 0.5 * (self.logdet + d * np.log(c))


def curvature(X):
    """For the leading eigen-directions of the field block of cov(X), how much
    of the posterior variance along the direction a quadratic in w explains
    beyond the straight line a Gaussian implies.  Returns the worst one."""
    Xf, w = X[:, :-1], X[:, -1]
    ev, V = np.linalg.eigh(np.cov(Xf, rowvar=False))
    wc = (w - w.mean()) / w.std()
    A1 = np.column_stack([np.ones_like(wc), wc])
    A2 = np.column_stack([A1, wc ** 2 - 1])
    best = None
    for j in range(Xf.shape[1]):
        y = (Xf - Xf.mean(0)) @ V[:, j]
        tot = y.var()
        r1 = y - A1 @ np.linalg.lstsq(A1, y, rcond=None)[0]
        r2 = y - A2 @ np.linalg.lstsq(A2, y, rcond=None)[0]
        gain = (r1.var() - r2.var()) / tot             # share explained by curvature only
        if best is None or gain > best[0]:
            best = (float(gain), j, y, float(ev[j]))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="e4_sampler_ab")
    ap.add_argument("--coords", nargs="+", default=list(COORDS), choices=COORDS)
    ap.add_argument("--n", type=int, default=500, help="draws (PDE solves) per test")
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
    use_keff = bool(scfg.use_keff_term)
    curve = BetaCurve.load(os.path.join(d, "curve.npz"))
    smp = ThreeStepSampler(prob, curve, scfg)
    syn, r = prob.syn, prob.r
    mu_r, s_r = scfg.mu_rho, scfg.s_rho

    # ---- posterior samples of the run, with their own Phi (no solves)
    TH, RHO, BETA, PHI = [], [], [], []
    for p in paths:
        with np.load(p) as z:
            rho = z["kept_rho"].ravel()
            pobs = z["kept_pobs"].reshape(len(rho), -1)
            res = syn.d_p[None, :] - pobs
            phi = 0.5 * prob._w * np.einsum("ij,ij->i", res, res)
            if use_keff:
                phi += 0.5 * ((np.log(syn.keff_obs) - np.log(z["kept_keff"].ravel()))
                              / syn.sigma_e) ** 2
            TH.append(z["kept_theta"].reshape(len(rho), r))
            RHO.append(rho); BETA.append(z["kept_beta"].ravel()); PHI.append(phi)
    L = min(len(x) for x in RHO)                      # equal length for ESS
    TH = np.stack([t[:L] for t in TH]); RHO = np.stack([x[:L] for x in RHO])
    BETA = np.stack([x[:L] for x in BETA]); PHI = np.stack([x[:L] for x in PHI])
    n_ch = len(paths)
    if n_ch * L < 20 * (r + 1):
        print(f"{exp}: only {n_ch * L} kept samples for a covariance in d = {r + 1}; "
              f"need a longer run"); return
    W = (np.log(RHO) - mu_r) / s_r
    LEV = np.log(BETA) + RHO * prob.kle.spatial_mean(TH)         # level = spatial mean of log K
    th_all, rho_all, w_all = TH.reshape(-1, r), RHO.ravel(), W.ravel()
    lp_theta = -PHI.ravel() - 0.5 * (th_all * th_all).sum(1) - 0.5 * w_all ** 2
    solves_run = n_ch * scfg.n_total * len(getattr(scfg, "moves", "ABC"))
    ess_run = dict(rho=D.ess_multi(RHO), beta=D.ess_multi(BETA), level=D.ess_multi(LEV),
                   theta_min=min(D.ess_multi(TH[:, :, i]) for i in range(r)))
    mc = dict(rho=(float(RHO.mean()), float(RHO.std())),
              beta=(float(BETA.mean()), float(BETA.std())),
              level=(float(LEV.mean()), float(LEV.std())))
    print(f"{exp}: {n_ch} chains x {L} kept samples, d = {r + 1}; run used {solves_run:,} solves")

    # ---- fit and fit diagnostics per coordinate system (no solves)
    rng = np.random.default_rng(a.seed)
    sub = rng.choice(len(w_all), size=min(50_000, len(w_all)), replace=False)
    fits, fitdiag = {}, {}
    for k in a.coords:
        X = np.column_stack([to_coords(k, th_all, rho_all), w_all])
        G = Gauss(X.mean(0), np.cov(X, rowvar=False))
        lp = lp_theta + jac(k, rho_all, r)
        D2 = G.maha2(X)
        pred = {}
        for c in a.inflate:
            lr = lp - G.logpdf(X, c)                   # log r = log pi~ - log q
            pred[c] = float(np.exp(-(logmeanexp(lr) + logmeanexp(-lr))))
        lr1 = lp - G.logpdf(X)
        gain, j, ydir, ev = curvature(X[sub])
        fits[k] = (G, lp)
        fitdiag[k] = dict(sd_log_ratio=float(lr1.std()), predicted_ess_frac=pred,
                          maha2_mean=float(D2.mean()), maha2_sd=float(D2.std()),
                          chi2_mean=r + 1, chi2_sd=float(np.sqrt(2 * (r + 1))),
                          curvature_share=gain, curved_direction=int(j),
                          _plot=(lp[sub], G.logpdf(X[sub]), D2, X[sub, -1], ydir))
        print(f"fit [{k}]: sd(log pi - log q) = {lr1.std():.2f}; Mahalanobis D^2 "
              f"{D2.mean():.1f} +- {D2.std():.1f} (Gaussian: {r + 1} +- {np.sqrt(2 * (r + 1)):.1f}); "
              f"worst curvature share {gain:.3f}; predicted ESS/N "
              + ", ".join(f"c={c:g}: {v:.2e}" for c, v in pred.items()))

    # ---- importance sampling with fresh draws
    prob.phys.warmup()
    out, draws = [], {}
    for k in a.coords:
        G, _ = fits[k]
        for c in a.inflate:
            Lc = np.sqrt(c) * G.L
            Z = rng.standard_normal((a.n, r + 1))
            Xq = G.m + Z @ Lc.T
            logq = -0.5 * (Z * Z).sum(1) - 0.5 * (G.logdet + (r + 1) * np.log(c))
            wq = Xq[:, -1]
            rho_q = np.exp(mu_r + s_r * wq)
            th_q = Xq[:, :r] if k == "theta" else Xq[:, :r] / rho_q[:, None]
            logp = np.empty(a.n); phi_q = np.full(a.n, np.inf)
            beta_q = np.full(a.n, np.nan); lev_q = np.full(a.n, np.nan)
            t0 = time.time()
            for i in range(a.n):
                phi = smp.potential(th_q[i], float(wq[i]))[0]
                phi_q[i] = phi
                logp[i] = (-phi - 0.5 * th_q[i] @ th_q[i] - 0.5 * wq[i] ** 2
                           + jac(k, rho_q[i], r)) if np.isfinite(phi) else -np.inf
                b = smp.beta(rho_q[i])
                if np.isfinite(b):
                    beta_q[i] = b; lev_q[i] = smp.level(th_q[i], float(wq[i]))
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
                                 q05=float(wquant(v[g], wn[g], 0.05)),
                                 q95=float(wquant(v[g], wn[g], 0.95)))
            fin = phi_q[np.isfinite(phi_q)]
            res = dict(coords=k, inflate=c, n=a.n, ess=ess, ess_frac=ess / a.n, khat=kh,
                       reliable=bool(np.isfinite(kh) and kh <= 0.7),
                       predicted_ess_frac=fitdiag[k]["predicted_ess_frac"][c],
                       max_weight=float(wn.max()), outside_curve=int((~ok).sum()),
                       phi_draws=dict(median=float(np.median(fin)),
                                      q05=float(np.percentile(fin, 5)),
                                      q95=float(np.percentile(fin, 95))),
                       seconds=time.time() - t0, estimates=est)
            out.append(res); draws[(k, c)] = (logw, rho_q, wn, fin)
            print(f"IS [{k}, c = {c:g}]: ESS {ess:.1f} / {a.n} ({100 * ess / a.n:.2f}%), "
                  f"k-hat {kh:.2f}, max weight {wn.max():.3f}, median Phi of draws "
                  f"{res['phi_draws']['median']:.1f}, {res['seconds']:.0f} s")

    # ---- tables
    base = {k: v / solves_run for k, v in ess_run.items()}
    phi_post = dict(median=float(np.median(PHI)), q05=float(np.percentile(PHI, 5)),
                    q95=float(np.percentile(PHI, 95)))
    R = dict(exp=exp, dim=r + 1, n_chains=n_ch, kept_per_chain=L, run_solves=solves_run,
             run_ess=ess_run, run_ess_per_solve=base, mcmc=mc, sigma_rel=prob.sigma_rel,
             phi_posterior=phi_post,
             fit={k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                  for k, v in fitdiag.items()},
             tests=out)
    td, fd = io.tables_dir(exp), io.figures_dir(exp)
    with open(os.path.join(td, "is_gaussian.json"), "w") as f:
        json.dump(io._jsonable(R), f, indent=1)
    lines = [f"Best-case IS test, {exp}: proposal N(m, c S) fitted to {n_ch} x {L} kept samples, "
             f"d = {r + 1}\n",
             "How well the Gaussian describes the posterior samples (no solves):\n",
             "| coordinates | sd(log π − log q) | Mahalanobis D² mean ± sd (Gaussian: "
             f"{r + 1} ± {np.sqrt(2 * (r + 1)):.1f}) | worst curvature share | predicted ESS / N |",
             "|---|---|---|---|---|"]
    for k, v in fitdiag.items():
        lines.append(f"| {k} | {v['sd_log_ratio']:.2f} | {v['maha2_mean']:.1f} ± {v['maha2_sd']:.1f} | "
                     f"{v['curvature_share']:.3f} | "
                     + ", ".join(f"c={c:g}: {p:.2e}" for c, p in v["predicted_ess_frac"].items())
                     + " |")
    lines += ["", "Importance sampling with fresh draws (one PDE solve each):\n",
              "| coordinates | c | N solves | ESS | ESS / solve | Pareto k-hat | max weight | "
              "median Φ of draws [5%, 95%] | ρ IS mean (MCMC) | β IS mean (MCMC) | level IS mean (MCMC) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in out:
        e, ph = t["estimates"], t["phi_draws"]
        rate = f"{t['ess_frac']:.2e}" if t["reliable"] else "unreliable (k̂ > 0.7)"
        lines.append(f"| {t['coords']} | {t['inflate']:g} | {t['n']} | {t['ess']:.1f} | {rate} | "
                     f"{t['khat']:.2f} | {t['max_weight']:.3f} | "
                     f"{ph['median']:.0f} [{ph['q05']:.0f}, {ph['q95']:.0f}] | "
                     + " | ".join(f"{e[q]['mean']:.3f} ({e[q]['mcmc_mean']:.3f})"
                                  for q in ("rho", "beta", "level")) + " |")
    lines += ["", f"Posterior Φ (kept samples): median {phi_post['median']:.1f} "
              f"[{phi_post['q05']:.1f}, {phi_post['q95']:.1f}]",
              f"Curve sampler run: {solves_run:,} solves; ESS per solve "
              + ", ".join(f"{k} {v:.2e}" for k, v in base.items()),
              "", "k-hat < 0.5 good, 0.5-0.7 usable, > 0.7 unreliable (Vehtari et al., PSIS). "
              "Predicted ESS / N = 1 / (E_π[r] E_π[1/r]), r = π/q, from the posterior samples."]
    with open(os.path.join(td, "is_gaussian.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figure 1: the importance weights
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
    ls = ["-", "--", ":", "-."]
    for (k, c), (logw, rho_q, wn, fin) in draws.items():
        st = ls[a.inflate.index(c) % len(ls)]
        col = COL[k]
        lw_ = logw[np.isfinite(logw)]
        lab = f"{k}, c = {c:g}"
        ax[0].hist(lw_ - lw_.max(), bins=40, histtype="step", lw=2, ls=st, color=col, label=lab)
        sw = np.sort(wn)[::-1]
        ax[1].plot(np.arange(1, len(sw) + 1), np.cumsum(sw), color=col, lw=2, ls=st,
                   label=f"{lab}: ESS {kish(logw):.0f}/{a.n}, k̂ {pareto_khat(logw):.2f}")
        ax[2].hist(np.log10(fin), bins=40, histtype="step", lw=2, ls=st, color=col, density=True,
                   label=f"draws from q, {lab}")
        ax[3].hist(rho_q, bins=40, weights=wn, density=True, histtype="step", lw=2, ls=st,
                   color=col, label=f"IS, {lab}")
    ax[2].hist(np.log10(PHI.ravel()), bins=40, density=True, color="0.75", label="posterior samples")
    ax[3].hist(RHO.ravel(), bins=60, density=True, color="0.75", label="curve sampler")
    ax[3].axvline(syn.rho_true, color="k", ls="--", lw=1)
    ax[0].set_xlabel("log weight − max")
    ax[1].set_xscale("log"); ax[1].set_xlabel("number of largest weights")
    ax[1].set_ylabel("share of total weight"); ax[1].set_ylim(0, 1.02)
    ax[2].set_xlabel("log₁₀ Φ")
    ax[3].set_xlabel("ρ")
    for a_, t in zip(ax, ("log importance weights", "weight concentration",
                          "misfit Φ: proposal draws vs posterior", "ρ: weighted proposal vs MCMC")):
        a_.set_title(t, loc="left", fontsize=10); a_.legend(fontsize=7)
    fig.suptitle(f"Best-case Gaussian importance sampling, d = {r + 1}, σ_rel = "
                 f"{prob.sigma_rel:.3g}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_is_gaussian.png"), dpi=150)

    # ---- figure 2: how well the fitted Gaussian describes the posterior samples
    from scipy.stats import chi2
    nk = len(a.coords)
    fig, ax = plt.subplots(nk, 3, figsize=(16, 4.3 * nk), squeeze=False)
    for i, k in enumerate(a.coords):
        v = fitdiag[k]
        lp_s, lq_s, D2, w_s, ydir = v["_plot"]
        col = COL[k]
        a_ = ax[i, 0]
        x0, y0 = lq_s - lq_s.mean(), lp_s - lp_s.mean()
        a_.scatter(x0, y0, s=2, alpha=0.15, color=col, rasterized=True)
        lim = [min(x0.min(), y0.min()), max(x0.max(), y0.max())]
        a_.plot(lim, lim, "k--", lw=1, label="perfect fit (slope 1)")
        a_.set_xlabel("log q (fitted Gaussian), centred")
        a_.set_ylabel("log π (posterior), centred")
        a_.set_title(f"{LABEL[k]}: density at the posterior samples\n"
                     f"sd(log π − log q) = {v['sd_log_ratio']:.2f}, predicted ESS/N "
                     + ", ".join(f"{p:.1e}" for p in v["predicted_ess_frac"].values()),
                     loc="left", fontsize=10)
        a_.legend(fontsize=8)
        a_ = ax[i, 1]
        a_.hist(D2, bins=80, density=True, color=col, alpha=0.6, label="posterior samples")
        xs = np.linspace(0, max(np.percentile(D2, 99.5), 2.5 * (r + 1)), 400)
        a_.plot(xs, chi2.pdf(xs, r + 1), "k", lw=1.5, label=f"χ²({r + 1}) if Gaussian")
        a_.set_xlabel("Mahalanobis D² under the fitted Gaussian")
        a_.set_title(f"{LABEL[k]}: Mahalanobis distance\nD² {v['maha2_mean']:.1f} ± {v['maha2_sd']:.1f} "
                     f"(Gaussian {r + 1} ± {np.sqrt(2 * (r + 1)):.1f})", loc="left", fontsize=10)
        a_.legend(fontsize=8)
        a_ = ax[i, 2]
        a_.scatter(w_s, ydir, s=2, alpha=0.1, color=col, rasterized=True)
        bins = np.quantile(w_s, np.linspace(0.01, 0.99, 25))
        mid = 0.5 * (bins[1:] + bins[:-1])
        idx = np.digitize(w_s, bins) - 1
        bm = np.array([ydir[idx == b].mean() if np.any(idx == b) else np.nan
                       for b in range(len(mid))])
        a_.plot(mid, bm, "ko-", ms=3, lw=1.5, label="posterior mean, binned")
        p1 = np.polyfit(w_s, ydir, 1)
        a_.plot(mid, np.polyval(p1, mid), color="0.3", ls="--", lw=1.5,
                label="Gaussian (straight line)")
        a_.set_xlabel("w = log ρ / s_ρ")
        a_.set_ylabel(f"eigen-direction {v['curved_direction']} of S")
        a_.set_title(f"{LABEL[k]}: most curved direction\n"
                     f"bend explains {100 * v['curvature_share']:.1f}% of its variance",
                     loc="left", fontsize=10)
        a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_is_gaussian_fit.png"), dpi=130)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
