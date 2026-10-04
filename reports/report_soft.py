"""
E5 figures and tables: beta free with the soft curve prior, moves A-D.

    fig_e5_posteriors.png   rho (with prior), beta (with the soft prior at the
                            posterior rho), level; truth lines; 90% intervals
    fig_e5_joint.png        posterior (rho, beta) over the soft-prior band
                            beta_bar(rho) e^{+/- s(rho)} and the E3 curve
    fig_e5_traces.png       rho, beta, level, Phi for all chains (+ PSRF)
    fig_e5_field.png        true log K | posterior mean | sd | slice with band
    fig_e5_moves.png        acceptance of moves A-D per chain,
                            and step-size adaptation
    tables/e5_soft/e5_stats.{md,csv}   one row per quantity: median, 90% CI,
        mean, sd, truth, z, covered, PSRF, split-R-hat, ESS
    tables/e5_soft/e5_summary.json     the same + coverage, Phi vs floor,
        acceptance, soft prior at the truth, the E4 comparison if present

    python reports/report_soft.py [--exp e5_soft]
The E5 prior test (likelihood off) is reports/report_soft_prior_test.py.
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from _common import have, load_problem, io, C
from brho import plotting as P
from brho import diagnostics as D
from brho.curve import BetaCurve
from brho.samplers import load_chains, trace_burn_index, SoftCurveSampler

MOVES = ("A: θ", "B: w", "C: ridge", "D: β–θ swap")


def load(exp):
    prob, man = load_problem(exp)
    d = io.results_dir(exp)
    paths = io.chain_files(d)
    if not paths:
        return prob, man, None, None, None
    ch = load_chains(paths)
    curve = BetaCurve.load(os.path.join(d, "curve.npz"))
    scfg = C.from_dict(C.SoftSamplerConfig, man["configs"]["SoftSamplerConfig"])
    return prob, man, ch, curve, scfg


def write_table(rows, path_md, path_csv, cols):
    with open(path_md, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |\n")
    with open(path_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r.get(c)) for c in cols) + "\n")


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=io.EXP_SOFT)
    a = ap.parse_args(argv)
    exp = a.exp
    if not have(exp):
        print("E5: no results"); return
    prob, man, ch, curve, scfg = load(exp)
    if ch is None:
        print(f"{exp}: set up, no chains yet"); return
    smp = SoftCurveSampler(prob, curve, scfg)
    syn, kle, n, r = prob.syn, prob.kle, prob.pc.n, prob.r
    nb, te = scfg.n_burn, scfg.trace_every
    ib = trace_burn_index(nb, te)
    post = {k: ch["tr_" + k][:, ib:] for k in ("rho", "beta", "level", "phi",
                                               "theta_norm2", "w", "xi_b")}
    n_ch = ch["tr_rho"].shape[0]
    rho_s, beta_s = ch["kept_rho"].ravel(), ch["kept_beta"].ravel()
    th_s = ch["kept_theta"].reshape(-1, r)
    # level = spatial mean of log K, exactly; the full fields are summarized
    # a block of cells at a time below (the sample x cell matrix is GBs)
    level_s = np.log(beta_s) + rho_s * kle.spatial_mean(th_s)
    floor = prob.noise_floor()
    Ks = syn.keff_obs

    # ---------------------------------------------------------- posteriors
    s_rho = D.summarize(rho_s, syn.rho_true)
    s_beta = D.summarize(beta_s, syn.beta_true)
    s_lb = D.summarize(np.log(beta_s), np.log(syn.beta_true))
    s_lev = D.summarize(level_s, syn.level_true)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    ax[0].hist(rho_s, 40, density=True, color=P.C_BAND, edgecolor="w")
    xx = np.linspace(0.05, min(curve.rho_max, 6), 300)
    ax[0].plot(xx, P.lognormal_pdf(xx, scfg.mu_rho, scfg.s_rho), "--",
               color=P.C_PRIOR, lw=1, label="prior")
    ax[0].axvline(syn.rho_true, color="r", lw=2, label=r"$\rho^\dagger$")
    ax[0].axvspan(s_rho["lo"], s_rho["hi"], color="k", alpha=0.07, label="90%")
    ax[0].set_title(f"$\\rho$: {s_rho['median']:.3f} [{s_rho['lo']:.2f}, {s_rho['hi']:.2f}]")
    ax[0].legend()
    ax[1].hist(beta_s, 50, density=True, color="#7fd0a8", edgecolor="w", label="posterior")
    rm = float(np.median(rho_s))
    bb = np.linspace(max(1e-3, beta_s.min() * 0.7), beta_s.max() * 1.3, 400)
    m_, s_ = smp.m(rm), smp.s(rm)
    ax[1].plot(bb, stats.lognorm.pdf(bb, s_, scale=np.exp(m_)), "--", color=P.C_PRIOR,
               lw=1, label=f"soft prior at ρ={rm:.2f}")
    ax[1].axvline(syn.beta_true, color="r", lw=2, label=r"$\beta^\dagger$")
    ax[1].axvspan(s_beta["lo"], s_beta["hi"], color="k", alpha=0.07)
    ax[1].set_title(f"$\\beta$ (free): {s_beta['median']:.3f} "
                    f"[{s_beta['lo']:.2f}, {s_beta['hi']:.2f}]")
    ax[1].legend(fontsize=8)
    ax[2].hist(level_s, 40, density=True, color="#b07fd0", edgecolor="w")
    ax[2].axvline(syn.level_true, color="purple", lw=2, ls=":", label="true level")
    ax[2].set_title(f"level (mean $\\log K$): {s_lev['median']:.3f} "
                    f"[{s_lev['lo']:.2f}, {s_lev['hi']:.2f}]")
    ax[2].legend()
    P.save(fig, exp, "fig_e5_posteriors")

    # ---------------------------------------------------------- joint
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    rr = np.linspace(0.02, curve.rho_max, 300)
    mm = np.array([smp.m(x) for x in rr]); ss = np.array([smp.s(x) for x in rr])
    ax.fill_between(rr, np.exp(mm - 2 * ss), np.exp(mm + 2 * ss), color=P.C_BAND,
                    alpha=0.2, label=r"soft prior $\pm 2s(\rho)$")
    ax.fill_between(rr, np.exp(mm - ss), np.exp(mm + ss), color=P.C_BAND,
                    alpha=0.35, label=r"soft prior $\pm s(\rho)$")
    ax.plot(rr, np.exp(mm), color=P.C_CURVE, lw=1.5, label=r"$\bar\beta(\rho)$ (E3)")
    ax.plot(rho_s, beta_s, ".", color=P.C_POST, ms=2, alpha=0.35)
    # proxy handle: the scatter's own marker (ms=2, alpha=0.35) is invisible in the legend
    ax.plot([], [], "o", color=P.C_POST, ms=5, label="E5 posterior")
    P.truth_star(ax, syn.rho_true, syn.beta_true)
    ax.set_yscale("log"); ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\beta$")
    ax.set_xlim(0, min(curve.rho_max, max(3.5, rho_s.max() * 1.2)))
    ax.set_title(r"E5: $\beta$ free inside the soft curve band")
    ax.legend(fontsize=8)
    P.save(fig, exp, "fig_e5_joint")

    # ---------------------------------------------------------- traces
    keys = [("rho", r"$\rho$", syn.rho_true), ("beta", r"$\beta$", syn.beta_true),
            ("level", "level", syn.level_true), ("phi", r"$\Phi$", floor)]
    psrf = {k: D.psrf(post[k]) if n_ch > 1 else np.nan
            for k in ("rho", "beta", "level", "phi", "theta_norm2", "xi_b")}
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    stride = max(1, ch["tr_rho"].shape[1] // 2000)
    for j, (k, lab, tv) in enumerate(keys):
        for c in range(n_ch):
            t = ch["tr_" + k][c]
            ax[j].plot(np.arange(0, len(t), stride) * te, t[::stride], lw=0.5, alpha=0.7)
        ax[j].axhline(tv, color="r", lw=1.2, ls=":")
        ax[j].axvline(nb, color="gray", ls="--", lw=1)
        ax[j].set_title(f"{lab}   PSRF={psrf[k]:.3f}")
        ax[j].set_xlabel("iteration")
        if k in ("phi", "beta"):
            ax[j].set_yscale("log")
    P.save(fig, exp, "fig_e5_traces")

    # ---------------------------------------------------------- field
    fs = D.logK_summaries(np.log(beta_s), rho_s, th_s, kle, syn.logK_true)
    m_lk, sd_lk, lo_lk, hi_lk, cov90 = fs["mean"], fs["sd"], fs["lo"], fs["hi"], fs["coverage"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    v0 = min(syn.logK_true.min(), m_lk.min()); v1 = max(syn.logK_true.max(), m_lk.max())
    P.field(ax[0], syn.logK_true, n, r"true $\log K$", vmin=v0, vmax=v1)
    P.field(ax[1], m_lk, n, r"posterior mean $\log K$", vmin=v0, vmax=v1)
    P.field(ax[2], sd_lk, n, r"posterior sd $\log K$", cmap="magma")
    jr = P.slice_row(n); xs = prob.phys.xc
    ax[3].fill_between(xs, P.line(lo_lk, n, jr), P.line(hi_lk, n, jr), color=P.C_BAND,
                       alpha=0.5, label="5–95%")
    ax[3].plot(xs, P.line(m_lk, n, jr), "b-", label="posterior mean")
    ax[3].plot(xs, P.line(syn.logK_true, n, jr), "k-", lw=2, label="truth")
    ax[3].set_xlabel(f"x (y = {xs[jr]:.2f})")
    ax[3].set_title(f"slice; pointwise 90% coverage {cov90:.0%}")
    ax[3].legend()
    P.save(fig, exp, "fig_e5_field")

    # ---------------------------------------------------------- moves
    n_post = scfg.n_total - scfg.n_burn
    nm = ch["acc_tot"].shape[1]
    # whole-run rates per proposal (E8's random block schedule skips moves)
    props = ch["acc_tot_prop"] if "acc_tot_prop" in ch else np.full(ch["acc_tot"].shape,
                                                                     float(scfg.n_total))
    acc = ch["acc_tot"] / np.maximum(props, 1)
    names = list(man.get("moves") or MOVES)[:nm] if nm > 4 else list(MOVES)
    targets = list(scfg.targets) + [scfg.block_target if scfg.block_target > 0
                                    else scfg.targets[0]] * (nm - len(scfg.targets))
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.0))
    if nm <= 4:
        wdt = 0.8 / nm
        for j in range(nm):
            ax[0].bar(np.arange(n_ch) + (j - (nm - 1) / 2) * wdt, acc[:, j], wdt,
                      label=names[j])
        for j, tg in enumerate(targets):
            ax[0].axhline(tg, color=f"C{j}", ls=":", lw=1)
        ax[0].set_xlabel("chain")
        ax[0].legend(fontsize=8)
    else:                                 # E8: one bar per move, chain spread
        ax[0].bar(np.arange(nm), acc.mean(0), 0.7, color=P.C_BAND,
                  yerr=[acc.mean(0) - acc.min(0), acc.max(0) - acc.mean(0)], capsize=2)
        ax[0].plot(np.arange(nm), targets, "k_", ms=14, label="target")
        ax[0].set_xticks(np.arange(nm), names, rotation=60, fontsize=7)
        ax[0].legend(fontsize=8)
    ax[0].set_ylabel("acceptance (whole run)")
    ax[0].set_title("acceptance per move (targets dotted / black; bars: min-max over chains)")
    for c in range(n_ch):
        sh = ch["s_hist"][c]
        for j in range(sh.shape[1]):
            ax[1].plot(np.arange(1, len(sh) + 1) * scfg.adapt_every, sh[:, j],
                       color=f"C{j % 10}", lw=0.6, alpha=0.6,
                       label=names[j] if c == 0 and j < 10 else None)
    ax[1].set_yscale("log"); ax[1].set_xlabel("iteration (burn-in)")
    ax[1].set_title("step sizes during adaptation (frozen after)")
    ax[1].legend(fontsize=8)
    P.save(fig, exp, "fig_e5_moves")

    # ---------------------------------------------------------- tables
    def row(name, samples, truth, pkey=None, tr=None):
        s = D.summarize(samples, truth)
        out = dict(quantity=name, median=s["median"], q05=s["lo"], q95=s["hi"],
                   mean=s["mean"], sd=s["sd"], truth=truth,
                   z=s.get("z"), covered=s.get("covered"))
        if tr is not None and n_ch > 1:
            out.update(psrf=D.psrf(tr), split_rhat=D.split_rhat(tr), ess=D.ess_multi(tr))
        return out
    rows = [row("rho", rho_s, syn.rho_true, tr=post["rho"]),
            row("beta", beta_s, syn.beta_true, tr=post["beta"]),
            row("log beta", np.log(beta_s), np.log(syn.beta_true), tr=np.log(post["beta"])),
            row("level", level_s, syn.level_true, tr=post["level"]),
            row("Phi", post["phi"].ravel(), floor, tr=post["phi"]),
            row("theta_norm2/r", post["theta_norm2"].ravel() / r, 1.0,
                tr=post["theta_norm2"] / r),
            row("xi_b", post["xi_b"].ravel(), None, tr=post["xi_b"])]
    if "tr_sigma_rel" in ch:
        # learned noise level (--learn-noise 1); 'truth' = the matched value
        # sigma_p / |d|, the actual per-sensor noise in RE units
        sr = ch["tr_sigma_rel"][:, ib:]
        sr_match = float(syn.sigma_p / np.linalg.norm(syn.d_p))
        rows.append(row("sigma_rel (learned)", sr.ravel(), sr_match, tr=sr))
    cols = ["quantity", "median", "q05", "q95", "mean", "sd", "truth", "z", "covered",
            "psrf", "split_rhat", "ess"]
    tdir = io.tables_dir(exp)
    write_table(rows, os.path.join(tdir, "e5_stats.md"), os.path.join(tdir, "e5_stats.csv"), cols)
    summ = dict(n_chains=n_ch, n_total=scfg.n_total, n_burn=scfg.n_burn, thin=scfg.thin,
                r=r, truth_mode=prob.pc.truth_mode, Kstar=Ks,
                noise_floor=floor, mean_phi=float(post["phi"].mean()),
                coverage90_logK=cov90,
                acceptance_mean=dict(zip(names, map(float, acc.mean(0)))),
                steps_median=dict(zip(names, map(float, np.median(ch["s_final"], 0)))),
                solves_per_iteration=man.get("solves_per_iteration", 3),
                block=man.get("block"),
                soft_prior_at_truth=man.get("soft_prior_at_truth"),
                rho=s_rho, beta=s_beta, log_beta=s_lb, level=s_lev,
                psrf={k: float(v) for k, v in psrf.items()})
    e4 = os.path.join(io.tables_dir(io.EXP_SAMPLER), "e4_summary.json")
    if os.path.exists(e4):
        j4 = io.load_json(e4)
        summ["e4_for_comparison"] = dict(rho=j4.get("rho"), beta=j4.get("beta"))
    io.save_json(os.path.join(tdir, "e5_summary.json"), summ)
    print(f"E5: rho {s_rho['median']:.3f} [{s_rho['lo']:.2f}, {s_rho['hi']:.2f}] "
          f"(truth {syn.rho_true}); beta {s_beta['median']:.3f} [{s_beta['lo']:.2f}, "
          f"{s_beta['hi']:.2f}] (truth {syn.beta_true}); PSRF rho/beta/level "
          f"{psrf['rho']:.3f}/{psrf['beta']:.3f}/{psrf['level']:.3f}; mean Phi "
          f"{summ['mean_phi']:.1f} vs floor {floor:.1f}; logK 90% coverage {cov90:.0%}")


if __name__ == "__main__":
    main()
