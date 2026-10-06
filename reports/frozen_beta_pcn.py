"""
The "obvious" beta fix, run as a sampler: freeze (rho*, theta*) at the stored
state of a hard-curve (E4) run with the lowest misfit Phi, then sample log beta
alone with pCN and full PDE solves, under the curve prior at rho*,

    log beta ~ N(m*, s*^2),   m* = log beta_bar(rho*),
                              s*^2 = log_scatter(rho*)^2 + sigma_e^2,

    z = (log beta - m*) / s*,  z' = sqrt(1 - b^2) z + b xi,
    accept with min(1, exp(Phi(log beta) - Phi(log beta'))).

Shown side by side with the level swap (brho/beta_upgrade.py), which moves
log beta and theta together along the direction that leaves log K unchanged:
    (1) at the same frozen (rho*, theta*), 20 000 draws;
    (2) over every kept sample of the run (the method in the paper).

With theta frozen, a change of log beta rescales the whole field and the
pressures by e^-delta (flux scaling), so Phi grows like delta^2 / (2 sigma_rel^2)
and the posterior of log beta is pinned at width ~ sigma_rel, whatever the
prior.  The script also evaluates that 1D posterior exactly on a grid (no MCMC)
so that the narrow result cannot be blamed on the sampler.

    BRHO_ROOT=... python reports/frozen_beta_pcn.py [--exp e4_sampler]
        [--chains 8] [--iters 10000] [--burn 2000] [--seed 0]

Cost: chains * iters + ~300 PDE solves (about 2 ms each at 32x32).
Writes tables/<exp>/frozen_beta_pcn.{json,md} and figures/<exp>/fig_frozen_beta_pcn.png.
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
from brho.beta_upgrade import constant_coords, level_swap, summarize

TARGET_ACC = 0.44          # 1D random-walk-type optimum
trap = getattr(np, "trapezoid", None) or np.trapz


def ess_1d(x):
    """Effective sample size of one chain (initial positive sequence)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or np.var(x) == 0:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / (np.var(x) * n)
    tau = 1.0
    for k in range(1, n - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
    return float(n / tau)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=io.EXP_SAMPLER)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--iters", type=int, default=10000, help="per chain, incl. burn-in")
    ap.add_argument("--burn", type=int, default=2000, help="adaptive pCN step during burn-in")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    exp = args.exp
    if not have(exp):
        print(f"{exp}: no results"); return
    prob, man = load_problem(exp)
    d = io.results_dir(exp)
    paths = io.chain_files(d)
    if not paths:
        print(f"{exp}: no chains"); return
    scfg = C.from_dict(C.SamplerConfig, man["configs"]["SamplerConfig"])
    use_keff = bool(scfg.use_keff_term)
    curve = BetaCurve.load(os.path.join(d, "curve.npz"))
    syn, kle, r = prob.syn, prob.kle, prob.r
    sig_e2 = syn.sigma_e ** 2
    lb_true = float(np.log(syn.beta_true))
    rng = np.random.default_rng(args.seed)

    # ---- kept samples, read lazily (only what is needed)
    th, rh, bh, po, ke = [], [], [], [], []
    for p in paths:
        with np.load(p) as z:
            th.append(z["kept_theta"].reshape(-1, r)); rh.append(z["kept_rho"].ravel())
            bh.append(z["kept_beta"].ravel()); po.append(z["kept_pobs"].reshape(len(rh[-1]), -1))
            ke.append(z["kept_keff"].ravel())
    theta, rho, beta_h = np.concatenate(th), np.concatenate(rh), np.concatenate(bh)
    pobs, keff = np.concatenate(po), np.concatenate(ke)
    del th, po
    res_pt = syn.d_p[None, :] - pobs
    phi_kept = 0.5 * prob._w * np.einsum("ij,ij->i", res_pt, res_pt)
    if use_keff:
        phi_kept += 0.5 * ((np.log(syn.keff_obs) - np.log(keff)) / syn.sigma_e) ** 2
    phi_kept[~np.isfinite(phi_kept)] = np.inf
    k = int(np.argmin(phi_kept))
    rho_s, theta_s = float(rho[k]), theta[k].copy()
    m_s = float(np.log(beta_h[k]))
    s_s = float(np.sqrt(curve.log_scatter(rho_s) ** 2 + sig_e2))
    G = rho_s * kle.field(theta_s)               # rho* T theta*, frozen

    n_solves = 0

    def phi(lb):
        nonlocal n_solves
        n_solves += 1
        return prob.potential(np.exp(lb + G), use_keff)[0]

    phi_star = phi(m_s)
    print(f"{exp}: {len(rho)} kept samples; lowest Phi {phi_kept[k]:.3f} "
          f"(re-solved {phi_star:.3f}) at rho* = {rho_s:.4f}, beta_bar(rho*) = {beta_h[k]:.4f}; "
          f"noise floor {prob.noise_floor(use_keff):.2f}")
    print(f"prior at rho*: log beta ~ N({m_s:.4f}, {s_s:.4f}^2); truth log beta {lb_true:.4f} "
          f"({(lb_true - m_s) / s_s:+.2f} prior sd)")

    # ---- the obvious sampler: pCN on z = (log beta - m*) / s*, full PDE solves
    t0 = time.time()
    chains, accs, steps, starts = [], [], [], []
    for ci in range(args.chains):
        z = float(rng.standard_normal())            # start from the curve prior
        starts.append(m_s + s_s * z)
        f = phi(m_s + s_s * z)
        b, nacc, win = 0.5, 0, 0
        out = np.empty(args.iters)
        for it in range(args.iters):
            zp = np.sqrt(1 - b * b) * z + b * rng.standard_normal()
            fp = phi(m_s + s_s * zp)
            ok = np.log(rng.random()) < f - fp
            if ok:
                z, f = zp, fp
            out[it] = m_s + s_s * z
            if it < args.burn:                      # adapt the pCN step
                win += ok
                if (it + 1) % 20 == 0:
                    b = float(np.clip(b * (1.2 if win / 20 > TARGET_ACC else 1 / 1.2), 1e-6, 1.0))
                    win = 0
            else:
                nacc += ok
        chains.append(out); accs.append(nacc / max(1, args.iters - args.burn)); steps.append(b)
    chains = np.array(chains)
    post = chains[:, args.burn:]
    print(f"pCN: {args.chains} x {args.iters} its in {time.time() - t0:.0f}s; "
          f"step b {np.median(steps):.2e}, acceptance {np.mean(accs):.2f}")

    # ---- exact 1D posterior on a grid (no MCMC): prior x likelihood
    c0, w0 = np.median(post), max(np.std(post), 1e-4)
    grid = np.linspace(c0 - 10 * w0, c0 + 10 * w0, 241)
    phig = np.array([phi(x) for x in grid])
    lpg = -phig - 0.5 * ((grid - m_s) / s_s) ** 2
    pg = np.exp(lpg - lpg.max()); pg /= trap(pg, grid)
    g_mean = float(trap(grid * pg, grid))
    g_sd = float(np.sqrt(trap((grid - g_mean) ** 2 * pg, grid)))
    cdf = np.cumsum(pg) * (grid[1] - grid[0])
    g_lo, g_hi = np.interp([0.05, 0.95], cdf / cdf[-1], grid)
    # curvature of Phi in log beta at the minimum -> sd ~ 1/sqrt(Phi'')
    i0 = int(np.argmin(phig)); h = grid[1] - grid[0]
    i0 = min(max(i0, 1), len(grid) - 2)
    phi2 = float((phig[i0 + 1] - 2 * phig[i0] + phig[i0 - 1]) / h ** 2)

    # ---- the level swap: at the same frozen state, and over the whole run
    c, c_resid = constant_coords(kle.T)
    nf = 20000
    fz = level_swap(np.repeat(theta_s[None, :], nf, 0), np.full(nf, rho_s), c,
                    np.full(nf, m_s), np.full(nf, s_s), rng, return_theta=False)
    m_all = np.log(beta_h)
    s_all = np.sqrt(np.array([curve.log_scatter(x) for x in rho]) ** 2 + sig_e2)
    up = level_swap(theta, rho, c, m_all, s_all, rng, return_theta=False)

    ess = [ess_1d(x) for x in post]
    S = lambda x, w=None: summarize(np.exp(x), syn.beta_true, w=w)   # noqa: E731
    res = dict(
        exp=exp, sigma_rel=prob.sigma_rel, use_keff_term=use_keff,
        frozen_state=dict(index=k, rho=rho_s, beta_bar=float(beta_h[k]), phi=phi_star,
                          phi_from_kept=float(phi_kept[k]),
                          noise_floor=prob.noise_floor(use_keff),
                          prior_log_beta_mean=m_s, prior_log_beta_sd=s_s,
                          theta_dot_c=float(theta_s @ c)),
        truth=dict(beta=syn.beta_true, rho=syn.rho_true, log_beta=lb_true),
        prior_at_rho_star=summarize(np.exp(m_s + s_s * rng.standard_normal(nf)), syn.beta_true),
        frozen_pcn=dict(**S(post.ravel()), chains=args.chains, iters=args.iters, burn=args.burn,
                        acceptance=float(np.mean(accs)), pcn_step=float(np.median(steps)),
                        ess_total=float(np.sum(ess)),
                        log_beta_sd=float(np.std(post)),
                        chain_medians=[float(np.median(x)) for x in post],
                        starts=[float(x) for x in starts]),
        frozen_exact_grid=dict(log_beta_mean=g_mean, log_beta_sd=g_sd,
                               beta_lo=float(np.exp(g_lo)), beta_hi=float(np.exp(g_hi)),
                               covered=bool(g_lo <= lb_true <= g_hi),
                               phi_curvature=phi2,
                               sd_from_curvature=float(1 / np.sqrt(phi2 + 1 / s_s ** 2)),
                               sigma_rel=prob.sigma_rel),
        level_swap_frozen=dict(**S(fz["lb"]), log_beta_sd=float(fz["sd"][0])),
        level_swap_all=dict(**S(up["lb"], up["w"]),
                            log_beta_sd=summarize(up["lb"], w=up["w"])["sd"],
                            weight_ess_fraction=up["ess"] / len(rho)),
        hard_curve=dict(**S(m_all), log_beta_sd=float(np.std(m_all))),
        pde_solves=n_solves, c_norm=float(np.linalg.norm(c)), constant_field_residual=c_resid,
    )

    td, fd = io.tables_dir(exp), io.figures_dir(exp)
    with open(os.path.join(td, "frozen_beta_pcn.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    rows = [("prior at rho* (curve)", res["prior_at_rho_star"], s_s, "0"),
            ("frozen (rho*, theta*), pCN on beta", res["frozen_pcn"],
             res["frozen_pcn"]["log_beta_sd"], f"{n_solves}"),
            ("frozen (rho*, theta*), level swap", res["level_swap_frozen"],
             res["level_swap_frozen"]["log_beta_sd"], "0"),
            ("hard curve, all samples", res["hard_curve"], res["hard_curve"]["log_beta_sd"], "0"),
            ("level swap, all samples", res["level_swap_all"],
             res["level_swap_all"]["log_beta_sd"], "0")]
    with open(os.path.join(td, "frozen_beta_pcn.md"), "w") as fh:
        fh.write(f"Frozen state: lowest-Phi kept sample, rho* = {rho_s:.4f}, Phi = {phi_star:.3f} "
                 f"(noise floor {prob.noise_floor(use_keff):.2f}); sigma_rel = {prob.sigma_rel:.3g}; "
                 f"truth beta = {syn.beta_true}\n\n")
        fh.write("| beta from | median | 90% interval | sd(log beta) | covers truth | new PDE solves |\n"
                 "|---|---|---|---|---|---|\n")
        for name, q, sd, ns in rows:
            fh.write(f"| {name} | {q['median']:.3f} | [{q['lo']:.3f}, {q['hi']:.3f}] | "
                     f"{sd:.4f} | {q['covered']} | {ns} |\n")
        g = res["frozen_exact_grid"]
        fh.write(f"\nExact 1D posterior at the frozen state (grid, no MCMC): sd(log beta) "
                 f"{g['log_beta_sd']:.4f}, 90% beta [{g['beta_lo']:.3f}, {g['beta_hi']:.3f}], "
                 f"covers {g['covered']}; 1/sqrt(Phi'' + 1/s*^2) = {g['sd_from_curvature']:.4f} "
                 f"(sigma_rel {prob.sigma_rel:.3g}).\n"
                 f"pCN: acceptance {res['frozen_pcn']['acceptance']:.2f}, step {res['frozen_pcn']['pcn_step']:.2e}, "
                 f"ESS {res['frozen_pcn']['ess_total']:.0f}; chains started from the prior.\n")

    # ---- figure
    cols = dict(prior="0.6", pcn="#b0563f", lsf="#e0a030", lsa="#2a6f97")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    a = ax[0]
    for x in chains:
        a.plot(np.arange(1, args.iters + 1), x, lw=0.6, alpha=0.8)
    a.axhline(lb_true, color="k", lw=1.5, ls="--", label="truth")
    a.axhspan(m_s - 1.645 * s_s, m_s + 1.645 * s_s, color=cols["prior"], alpha=0.15,
              label="curve prior 90%")
    a.axvline(args.burn, color="0.3", lw=0.8, ls=":")
    a.set_xscale("log")
    a.set_xlabel("iteration"); a.set_ylabel(r"$\log\beta$")
    a.set_title(f"pCN on log beta, (rho*, theta*) frozen: {args.chains} chains from the prior",
                fontsize=10)
    a.legend(fontsize=8, loc="upper right")

    a = ax[1]
    lo = min(m_s - 4 * s_s, up["lb"].min()); hi = max(m_s + 4 * s_s, up["lb"].max())
    xx = np.linspace(lo, hi, 400)
    a.plot(xx, np.exp(-0.5 * ((xx - m_s) / s_s) ** 2) / (s_s * np.sqrt(2 * np.pi)),
           color=cols["prior"], lw=1.5, label=r"curve prior at $\rho^*$")
    bins = np.linspace(lo, hi, 90)
    a.hist(up["lb"], bins, weights=up["w"] * len(rho), density=True, alpha=0.45,
           color=cols["lsa"], label="level swap, all samples")
    a.hist(fz["lb"], bins, density=True, histtype="step", lw=1.8, color=cols["lsf"],
           label=r"level swap, frozen $(\rho^*,\theta^*)$")
    a.plot(grid, pg, color=cols["pcn"], lw=2, label=r"pCN on $\beta$, frozen (exact)")
    a.axvline(lb_true, color="k", lw=1.5, ls="--", label="truth")
    a.set_yscale("log"); a.set_ylim(1e-2, pg.max() * 2)
    a.set_xlabel(r"$\log\beta$"); a.set_ylabel("density (log scale)")
    a.set_title("same frozen state, two ways of moving beta", fontsize=10)
    a.legend(fontsize=7.5, loc="upper left")

    a = ax[2]
    pbins = np.linspace(grid[0], grid[-1], 60)
    a.hist(post.ravel(), pbins, density=True, alpha=0.5, color=cols["pcn"], label="pCN samples")
    a.plot(grid, pg, color="k", lw=1.2, label="exact (grid)")
    if grid[0] <= lb_true <= grid[-1]:
        a.axvline(lb_true, color="k", lw=1.5, ls="--", label="truth")
    else:
        a.annotate(f"truth at {lb_true:.3f}\n({(lb_true - g_mean) / g_sd:+.0f} sd) " +
                   ("→" if lb_true > g_mean else "←"),
                   xy=(0.97 if lb_true > g_mean else 0.03, 0.6), xycoords="axes fraction",
                   ha="right" if lb_true > g_mean else "left", fontsize=9)
    a.set_xlabel(r"$\log\beta$")
    a.set_title(f"zoom: sd(log beta) = {g_sd:.4f} vs sigma_rel = {prob.sigma_rel:.3g}", fontsize=10)
    a.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "fig_frozen_beta_pcn.png"), dpi=150)
    print(open(os.path.join(td, "frozen_beta_pcn.md")).read())


if __name__ == "__main__":
    main()
