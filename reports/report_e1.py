"""
E1 post-processing: the naive (theta, beta, rho) model, blocked and/or global pCN.

    python reports/report_e1.py [--move all|blocked|global] [--n-pfield 6]
                                [--n-pstats 200] [--window 1000]

Each run folder (results/e1_blocked, results/e1_global) gets the figures and
tables below in figures/e1_<move>/ and tables/e1_<move>/.  With both runs
present, figures/e1_compare/ adds
    fig_e1_compare_traces.png          log beta, rho, level, relative error of all
                                       chains against PDE solves (equal compute)
    fig_e1_compare_identifiability.png spread of chain means / prior sd and PSRF
                                       per quantity, blocked vs global
    tables/e1_compare/e1_compare.{md,csv}

Figures per run
    fig_e1_acceptance.png   running acceptance per chain (window) and per block
                            (A: theta, B: (xi_beta, xi_rho) for "blocked"),
                            post-adaptation rates, step-size histories
    fig_e1_relerr.png       data-to-sample relative error ||d - p_obs|| / ||d||
                            vs iteration, all chains, with the noise level
    fig_e1_traces.png       beta and rho traces: the drift
    fig_e1_plane.png        post-adaptation (beta, rho) samples of every chain in
                            the (beta, rho) plane (+ E2 grid if present): chains
                            settle in different places along the plateau
    fig_e1_identified.png   what the data DO identify: level (spatial mean of
                            log K) and amplitude (rho * std G) vs beta and rho,
                            with PSRF across chains
    fig_e1_pressure.png     reconstructed posterior pressure fields: truth,
                            samples from chains with very different (beta, rho),
                            posterior mean and sd, residual of the mean
    fig_e1_logK.png         the corresponding log K samples: the same field
                            from different (beta, rho)
    fig_e1_fit.png          predicted vs observed pressure (thinned samples)
Tables per run
    e1_chains.{md,csv}      per chain: acceptance, step, final and mean beta/rho,
                            relative error, Phi
    e1_summary.json         PSRF / ESS of log beta, rho, level, amplitude, Phi;
                            spread of chain means; noise-level relative error
Data (results/e1_<move>/pfield_samples.npz)
    the reconstructed samples: chain, iteration, beta, rho, log K, pressure
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from _common import have, load_problem, io, C
from brho import plotting as P
from brho import diagnostics as D
from brho.naive import load_chains, reconstruct_fields

MOVES = ("blocked", "global")


def psrf_or_nan(x):
    """PSRF needs >= 2 chains; NaN (reported as n/a) otherwise."""
    return D.psrf(x) if x.shape[0] >= 2 else float("nan")


def fmt(v):
    return "n/a" if not np.isfinite(v) else f"{v:.2f}"


def running_mean(x, w):
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def main(argv=None):  # argv=[] when called from report_all
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--move", default="all", choices=("all",) + MOVES,
                    help="which E1 run(s) to report; 'all' also compares them")
    ap.add_argument("--n-pfield", type=int, default=6, help="pressure samples to show")
    ap.add_argument("--n-pstats", type=int, default=200,
                    help="samples reconstructed for the posterior mean / sd")
    ap.add_argument("--window", type=int, default=None, help="acceptance window")
    ap.add_argument("--exp-tag", default="", help="tagged run, e.g. a7 -> e1_blocked_a7")
    a = ap.parse_args(argv)
    moves = MOVES if a.move == "all" else (a.move,)
    done = {}
    for mv in moves:
        exp = io.exp_e1(mv, a.exp_tag)
        if not have(exp):
            print(f"E1 [{mv}]: no results"); continue
        out = report_move(exp, a)
        if out:
            done[mv] = out
    if len(done) == 2:
        compare(done)


def report_move(EXP, a):
    """All figures and tables of one E1 run folder; returns data for compare()."""
    prob, man = load_problem(EXP)
    cfg = C.from_dict(C.NaiveConfig, man["configs"]["NaiveConfig"])
    blocked = cfg.move == "blocked"
    ns = (int(getattr(cfg, "theta_repeat", 1)) + 1) if blocked else 1   # PDE solves / it
    paths = io.chain_files(io.results_dir(EXP))
    if not paths:
        print(f"E1 [{cfg.move}]: set up, no chains yet"); return None
    ch = load_chains(paths)
    syn, n = prob.syn, prob.pc.n
    L, nb = ch["n_common"], min(cfg.n_burn, ch["n_common"] - 1)
    m = len(paths)
    its = np.arange(L)
    th = max(1, L // 3000)                             # plotting stride
    win = a.window or max(50, min(1000, L // 20))
    cols = plt.cm.viridis(np.linspace(0, 0.95, m))
    lw = 0.6 if m <= 16 else 0.35
    beta_true, rho_true = syn.beta_true, syn.rho_true
    noise_rel = syn.sigma_p * np.sqrt(syn.n_obs) / np.linalg.norm(syn.d_p)

    # ---------------------------------------------------------- acceptance
    blocks = [("acc", "A: theta" if blocked else "global move", cfg.target)]
    if blocked:
        blocks.append(("acc_h", r"B: ($\xi_\beta, \xi_\rho$)", cfg.target_h))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for j, (k, lab, tg) in enumerate(blocks):
        for c in range(m):
            ax[0].plot(its[::th], running_mean(ch[k][c], win)[::th],
                       color=cols[c], lw=lw, ls="-" if j == 0 else ":")
        ax[0].axhline(tg, color="r", ls=":", lw=1)
    ax[0].axvline(nb, color="gray", ls="--", lw=1, label="end adaptation")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel(f"acceptance (window {win})")
    ax[0].set_title("running acceptance" + (" (solid: A, dotted: B)" if blocked else ""))
    ax[0].legend()
    acc_all = ch["acc"].mean(1)
    acc_post = ch["acc"][:, nb:].mean(1)
    acc_h_post = ch["acc_h"][:, nb:].mean(1) if blocked else None
    wbar = 0.4 if blocked else 0.6
    ax[1].bar(np.arange(m) - (0.2 if blocked else 0), acc_post, wbar, label=blocks[0][1])
    if blocked:
        ax[1].bar(np.arange(m) + 0.2, acc_h_post, wbar, label=blocks[1][1])
    ax[1].set_xlabel("chain")
    ax[1].set_title("after adaptation: " + ", ".join(
        f"{lab} {100 * v.mean():.1f}%" for (k, lab, _), v in
        zip(blocks, [acc_post] + ([acc_h_post] if blocked else []))))
    ax[1].legend()
    for c in range(m):
        sh = ch["s_hist"][c]
        for j in range(sh.shape[1]):
            ax[2].semilogy((np.arange(len(sh)) + 1) * cfg.adapt_every, sh[:, j],
                           color=cols[c], lw=lw, ls="-" if j == 0 else ":")
    ax[2].set_xlabel("iteration")
    ax[2].set_title("pCN step sizes (frozen after adaptation)"
                    + (" (solid: A, dotted: B)" if blocked else ""))
    P.save(fig, EXP, "fig_e1_acceptance")

    # ---------------------------------------------------------- relative error
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for c in range(m):
        ax.semilogy(its[::th], ch["relerr"][c][::th], color=cols[c], lw=lw)
    med = np.median(ch["relerr"], 0)
    ax.semilogy(its[::th], med[::th], "k-", lw=1.8, label="median over chains")
    ax.axhline(noise_rel, color="r", ls=":", lw=1.5,
               label=rf"noise level $\sigma_p\sqrt{{n}}/\|d\|$ = {noise_rel:.1e}")
    ax.axvline(nb, color="gray", ls="--", lw=1)
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\|d - p_{obs}\| / \|d\|$")
    ax.set_title(f"data-to-sample relative error, {m} chains [{cfg.move}]"); ax.legend()
    P.save(fig, EXP, "fig_e1_relerr")

    # ---------------------------------------------------------- beta, rho traces
    beta = np.exp(ch["logbeta"]); rho = ch["rho"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    for c in range(m):
        ax[0].semilogy(its[::th], beta[c][::th], color=cols[c], lw=lw)
        ax[1].plot(its[::th], rho[c][::th], color=cols[c], lw=lw)
    ax[0].axhline(beta_true, color="r", lw=1.5, ls=":", label=r"$\beta^\dagger$")
    ax[1].axhline(rho_true, color="r", lw=1.5, ls=":", label=r"$\rho^\dagger$")
    post = lambda x: x[:, nb:]
    for a_, x, name in ((ax[0], ch["logbeta"], r"$\log\beta$"), (ax[1], rho, r"$\rho$")):
        a_.axvline(nb, color="gray", ls="--", lw=1)
        a_.set_xlabel("iteration"); a_.legend()
        a_.set_title(f"{name} traces, PSRF = {fmt(psrf_or_nan(post(x)))}")
    P.save(fig, EXP, "fig_e1_traces")

    # ---------------------------------------------------------- (beta, rho) plane
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    grid = os.path.join(io.results_dir(io.EXP_GRID), "grid.npz")
    if os.path.exists(grid):
        z = np.load(grid)
        sc = ax.scatter(z["beta"], z["rho"], c=z["rms"] / syn.sigma_p, s=260,
                        cmap="coolwarm", vmin=0, vmax=3, edgecolors="k", alpha=0.5,
                        zorder=1, label="E2 grid (RMS/$\\sigma_p$)")
        plt.colorbar(sc, ax=ax, label=r"E2 RMS/$\sigma_p$")
    step = max(1, (L - nb) // 400)
    for c in range(m):
        ax.plot(beta[c, nb::step], rho[c, nb::step], ".", ms=2, color=cols[c], alpha=0.6,
                zorder=2)
        ax.plot(beta[c, -1], rho[c, -1], "o", mfc=cols[c], mec="k", ms=6, zorder=3)
    P.truth_star(ax, beta_true, rho_true)
    ax.set_xscale("log"); ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\rho$")
    ax.set_title("post-adaptation samples per chain (circles: last state)")
    ax.legend(loc="upper right")
    P.save(fig, EXP, "fig_e1_plane")

    # ---------------------------------------------------------- prior spreads
    # Monte Carlo prior of each summary (u ~ N(0, I)): the yardstick for how
    # much the data contract it.  ratio = spread of chain means / prior sd:
    # ~1 for a quantity the data leave free, << 1 for an identified one.
    rng_p = np.random.default_rng(123)
    U = rng_p.standard_normal((4000, prob.r + 2))
    Gp = prob.kle.field(U[:, :prob.r])
    lb_p = cfg.mu_logbeta + cfg.s_logbeta * U[:, prob.r]
    rho_p = np.exp(cfg.mu_logrho + cfg.s_logrho * U[:, prob.r + 1])
    prior_sd = dict(logbeta=lb_p.std(), rho=rho_p.std(),
                    level=(lb_p + rho_p * Gp.mean(1)).std(),
                    amp=(rho_p * Gp.std(1)).std())

    # ---------------------------------------------------------- identified combos
    keys = [("logbeta", r"$\log\beta$", np.log(beta_true)), ("rho", r"$\rho$", rho_true),
            ("level", r"level = mean $\log K$", syn.level_true),
            ("amp", r"amplitude $\rho\,$std$(G)$", rho_true * syn.G_std)]
    fig, ax = plt.subplots(1, 4, figsize=(19, 3.9))
    stats = {}
    for j, (k, lab, tv) in enumerate(keys):
        x = ch[k]
        for c in range(m):
            ax[j].plot(its[::th], x[c][::th], color=cols[c], lw=lw)
        ax[j].axhline(tv, color="r", ls=":", lw=1.5)
        ax[j].axvline(nb, color="gray", ls="--", lw=1)
        spread = float(post(x).mean(1).std()) if m >= 2 else float("nan")
        stats[k] = dict(psrf=psrf_or_nan(post(x)), ess=D.ess_multi(post(x)),
                        chain_means=post(x).mean(1).tolist(),
                        spread_of_chain_means=spread,
                        within_chain_sd=float(post(x).std(1).mean()), truth=float(tv),
                        prior_sd=float(prior_sd[k]),
                        spread_over_prior_sd=spread / float(prior_sd[k]))
        ax[j].set_title(f"{lab}\nchain-mean spread / prior sd = "
                        f"{stats[k]['spread_over_prior_sd']:.2f},  PSRF {fmt(stats[k]['psrf'])}")
        ax[j].set_xlabel("iteration")
    stats["phi"] = dict(psrf=psrf_or_nan(post(ch["phi"])), mean=float(post(ch["phi"]).mean()),
                        floor=prob.noise_floor())
    fig.suptitle("left: not identified by the pressure data;  right: identified "
                 "(red: truth)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    P.save(fig, EXP, "fig_e1_identified")

    # ---------------------------------------------------------- pressure samples
    kept = [(c, i) for c in range(m) for i, it in enumerate(ch["kept_it"][c]) if it >= nb]
    if kept:
        kb = np.array([ch["kept_logbeta"][c][i] for c, i in kept])
        # show chains/samples spanning the beta range
        order = np.argsort(kb)
        pick = [kept[order[int(q)]] for q in np.linspace(0, len(order) - 1, a.n_pfield)]
        rng = np.random.default_rng(0)
        stat_idx = rng.choice(len(kept), min(a.n_pstats, len(kept)), replace=False)
        sel = pick + [kept[s] for s in stat_idx]
        TH = np.array([ch["kept_theta"][c][i] for c, i in sel])
        LB = np.array([ch["kept_logbeta"][c][i] for c, i in sel])
        RH = np.array([ch["kept_rho"][c][i] for c, i in sel])
        lk, pf = reconstruct_fields(prob, TH, LB, RH)
        np.savez_compressed(os.path.join(io.results_dir(EXP), "pfield_samples.npz"),
                            chain=np.array([c for c, _ in sel]),
                            iteration=np.array([ch["kept_it"][c][i] for c, i in sel]),
                            beta=np.exp(LB), rho=RH, logK=lk, p=pf,
                            n_shown=len(pick), p_true=syn.p_true, logK_true=syn.logK_true)
        k_show = len(pick)
        p_mean, p_sd = pf[k_show:].mean(0), pf[k_show:].std(0)
        vmin, vmax = syn.p_true.min(), syn.p_true.max()
        ncol = k_show + 1
        fig, ax = plt.subplots(2, ncol, figsize=(3.3 * ncol, 6.4))
        P.field(ax[0, 0], syn.p_true, n, "true pressure", cmap="magma", vmin=vmin,
                vmax=vmax, obs_mask=syn.obs_mask, cbar=False)
        for j, (c, i) in enumerate(pick):
            P.field(ax[0, j + 1], pf[j], n,
                    f"chain {ch['chain_id'][c]}, it {ch['kept_it'][c][i]}\n"
                    rf"$\beta$={np.exp(LB[j]):.2g}, $\rho$={RH[j]:.2g}",
                    cmap="magma", vmin=vmin, vmax=vmax, cbar=False)
        P.field(ax[1, 0], p_mean, n, f"posterior mean ({len(stat_idx)} samples)",
                cmap="magma", vmin=vmin, vmax=vmax, cbar=False)
        P.field(ax[1, 1], p_sd, n, "posterior sd", cmap="viridis")
        P.field(ax[1, 2], (p_mean - syn.p_true) / syn.sigma_p, n,
                r"(mean $-$ true) / $\sigma_p$", cmap="RdBu_r", vmin=-5, vmax=5)
        for j in range(3, ncol):
            ax[1, j].axis("off")
        P.save(fig, EXP, "fig_e1_pressure")

        fig, ax = plt.subplots(1, ncol, figsize=(3.3 * ncol, 3.4))
        v0, v1 = np.percentile(np.concatenate([lk[:k_show].ravel(), syn.logK_true]), [1, 99])
        P.field(ax[0], syn.logK_true, n, r"true $\log K$", vmin=v0, vmax=v1, cbar=False)
        for j in range(k_show):
            P.field(ax[j + 1], lk[j], n, rf"$\beta$={np.exp(LB[j]):.2g}, $\rho$={RH[j]:.2g}",
                    vmin=v0, vmax=v1, cbar=False)
        P.save(fig, EXP, "fig_e1_logK")

        fig, ax = plt.subplots(figsize=(5.8, 5))
        for c in range(m):
            po = np.array([ch["kept_pobs"][c][i] for i, it in enumerate(ch["kept_it"][c])
                           if it >= nb])
            if len(po):
                ax.plot(np.repeat(syn.d_p[None], len(po), 0).ravel()[::7],
                        po.ravel()[::7], ".", ms=1.5, color=cols[c], alpha=0.3)
        lo, hi = syn.d_p.min(), syn.d_p.max()
        ax.plot([lo, hi], [lo, hi], "k-", lw=1)
        ax.set_xlabel("observed pressure"); ax.set_ylabel("predicted (thinned samples)")
        ax.set_title("all chains fit the data")
        P.save(fig, EXP, "fig_e1_fit")

    # ---------------------------------------------------------- tables
    rows = []
    for c in range(m):
        rows.append(dict(chain=int(ch["chain_id"][c]), iterations=int(ch["it"][c]),
                         acc=float(acc_all[c]), acc_post=float(acc_post[c]),
                         acc_h_post=float(acc_h_post[c]) if blocked else "",
                         step=float(ch["s_final"][c][0]),
                         step_h=float(ch["s_final"][c][1]) if blocked else "",
                         beta_final=float(beta[c, -1]), rho_final=float(rho[c, -1]),
                         beta_mean=float(np.exp(ch["logbeta"][c, nb:].mean())),
                         rho_mean=float(rho[c, nb:].mean()),
                         level_mean=float(ch["level"][c, nb:].mean()),
                         relerr_mean=float(ch["relerr"][c, nb:].mean()),
                         phi_mean=float(ch["phi"][c, nb:].mean())))
    io.write_table(EXP, "e1_chains", rows)
    summ = dict(move=cfg.move, n_chains=m, iterations=int(L), solves_per_iteration=ns,
                n_burn=nb, thin=cfg.thin, r=prob.r,
                acc_mean=float(acc_all.mean()), acc_post_mean=float(acc_post.mean()),
                acc_h_post_mean=float(acc_h_post.mean()) if blocked else None,
                step_median=np.median(ch["s_final"], 0).tolist(),
                noise_level_relerr=float(noise_rel),
                relerr_post_median=float(np.median(ch["relerr"][:, nb:])),
                beta_true=beta_true, rho_true=rho_true, level_true=syn.level_true,
                **{f"{k}_stats": v for k, v in stats.items()})
    io.save_json(io.tables_dir(EXP) + "/e1_summary.json", summ)
    print(f"E1 [{cfg.move}]: {m} chains x {L} it | acc {100 * acc_post.mean():.1f}%"
          + (f"/{100 * acc_h_post.mean():.1f}%" if blocked else "")
          + f" | steps {'/'.join(f'{x:.3g}' for x in np.median(ch['s_final'], 0))} | "
          f"spread/prior sd: logbeta {stats['logbeta']['spread_over_prior_sd']:.2f}, "
          f"rho {stats['rho']['spread_over_prior_sd']:.2f}, "
          f"level {stats['level']['spread_over_prior_sd']:.2f}, "
          f"amp {stats['amp']['spread_over_prior_sd']:.2f} | "
          f"PSRF logbeta {fmt(stats['logbeta']['psrf'])}, rho {fmt(stats['rho']['psrf'])}, "
          f"level {fmt(stats['level']['psrf'])}, amp {fmt(stats['amp']['psrf'])}, "
          f"Phi {fmt(stats['phi']['psrf'])} | relerr {summ['relerr_post_median']:.2e} "
          f"(noise {noise_rel:.2e})")
    return dict(move=cfg.move, ns=ns, nb=nb, L=L, m=m, logbeta=ch["logbeta"], rho=rho,
                level=ch["level"], relerr=ch["relerr"], stats=stats, summ=summ,
                truth=(np.log(beta_true), rho_true, syn.level_true), noise_rel=noise_rel,
                steps=np.median(ch["s_final"], 0))


def compare(done):
    """Blocked vs global on a common axis: PDE solves (blocked costs 2 per
    iteration, global 1), so equal x means equal compute."""
    exp = "e1_compare"
    cmv = {"blocked": "#2b5d8a", "global": "#c0504d"}
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    for mv, d in done.items():
        x = np.arange(d["L"]) * d["ns"]
        th = max(1, d["L"] // 2000)
        for c in range(d["m"]):
            kw = dict(color=cmv[mv], lw=0.4, alpha=0.6,
                      label=f"{mv} (steps {'/'.join(f'{s:.3g}' for s in d['steps'])})"
                      if c == 0 else None)
            ax[0].plot(x[::th], d["logbeta"][c][::th], **kw)
            ax[1].plot(x[::th], d["rho"][c][::th], **kw)
            ax[2].plot(x[::th], d["level"][c][::th], **kw)
        ax[3].semilogy(x[::th], np.median(d["relerr"], 0)[::th], color=cmv[mv], lw=1.6,
                       label=f"{mv}: median relative error")
    lbt, rt, lvt = next(iter(done.values()))["truth"]
    for a_, tv, lab in ((ax[0], lbt, r"$\log\beta$"), (ax[1], rt, r"$\rho$"),
                        (ax[2], lvt, "level (mean $\log K$)")):
        a_.axhline(tv, color="k", ls=":", lw=1.2)
        a_.set_title(lab + " traces, all chains"); a_.set_xlabel("PDE solves")
    ax[3].axhline(next(iter(done.values()))["noise_rel"], color="k", ls=":", lw=1.2,
                  label="noise level")
    ax[3].set_xlabel("PDE solves"); ax[3].set_title("data-to-sample relative error")
    ax[0].legend(); ax[3].legend()
    P.save(fig, exp, "fig_e1_compare_traces")

    keys = [("logbeta", r"$\log\beta$"), ("rho", r"$\rho$"), ("level", "level"),
            ("amp", "amplitude")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    xx = np.arange(len(keys))
    for j, (mv, d) in enumerate(done.items()):
        ax[0].bar(xx + (j - 0.5) * 0.38, [d["stats"][k]["spread_over_prior_sd"] for k, _ in keys],
                  0.38, color=cmv[mv], label=mv)
        ax[1].bar(xx + (j - 0.5) * 0.38, [d["stats"][k]["psrf"] for k, _ in keys],
                  0.38, color=cmv[mv], label=mv)
    for a_, t in ((ax[0], "spread of chain means / prior sd\n(~1: not identified, <<1: identified)"),
                  (ax[1], "PSRF across chains")):
        a_.set_xticks(xx); a_.set_xticklabels([lab for _, lab in keys])
        a_.set_title(t); a_.legend()
    ax[1].axhline(1.1, color="k", ls=":", lw=1)
    P.save(fig, exp, "fig_e1_compare_identifiability")
    rows = []
    for mv, d in done.items():
        sm = d["summ"]
        rows.append(dict(move=mv, chains=d["m"], iterations=d["L"], solves=d["L"] * d["ns"],
                         steps="/".join(f"{s:.3g}" for s in d["steps"]),
                         acc_post=sm["acc_post_mean"], acc_h_post=sm["acc_h_post_mean"] or "",
                         relerr_median=sm["relerr_post_median"],
                         **{f"{k}_spread_over_prior": d["stats"][k]["spread_over_prior_sd"]
                            for k, _ in keys},
                         **{f"{k}_psrf": d["stats"][k]["psrf"] for k, _ in keys}))
    io.write_table(exp, "e1_compare", rows)
    print("E1 compare:", "; ".join(
        f"{r['move']}: steps {r['steps']}, spread/prior logbeta "
        f"{r['logbeta_spread_over_prior']:.2f} level {r['level_spread_over_prior']:.2f}"
        for r in rows))


if __name__ == "__main__":
    main()
