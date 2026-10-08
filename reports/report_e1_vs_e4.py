"""
Naive sampler (E1) against the 3-move curve sampler (E4) at equal cost: same
KLE truth, same data, same sigma_rel, same forward solves per chain.  The two
runs live in different study folders, so both roots are arguments (no BRHO_ROOT).

    python reports/report_e1_vs_e4.py \
        [--e1-root runs/cluster_kle] [--e1-exp e1_blocked] \
        [--e4-root runs/cluster_kle_sig2e3] [--e4-exp e4_sampler] [--out-root <e4-root>]

1. Intervals (needs the E1 chain files, i.e. run on the cluster): pooled
   post-burn-in median and 90% interval of beta, rho, the level and Phi from
   E1's per-iteration traces -> <e1-root>/tables/<e1-exp>/e1_intervals.{json,md}.
   Also <e1-root>/figures/<e1-exp>/fig_e1_field.png, the E1 counterpart of
   fig_e4_field.png (true log K | posterior mean | posterior sd | slice with
   90% band), from E1's kept samples after burn-in.
   Skipped, with a message, when the chains are not there.
2. Convergence (from the committed mpsrf_*.csv tables; runs anywhere), against
   forward solves per chain, into <out-root>/figures/e1_vs_e4/:
     fig_e1_psrf.png         E1: PSRF of every entry (57 theta modes, log beta, log rho)
     fig_e1_mpsrf.png        E1: MPSRF of theta and of all entries
     fig_e1_vs_e4_mpsrf.png  MPSRF of theta, E4 on top of E1
   Burn-in: E4 keeps theta only after its burn-in (30k iterations = 90k solves),
   so its curves start at ~93k solves; E1 keeps everything from iteration 0.
   With --burn-match solves (default) E1's curves are recomputed from its chain
   files with the window starting at the same number of forward solves as E4's
   (--burn-match iterations: the same iteration; none: the committed E1 curves).
   Written to <e1-root>/tables/<e1-exp>/mpsrf_{theta,all}_burn_matched.csv and
   used for the figures when present.  Also fig_e4_psrf.png: E4's PSRF of
   every entry (57 theta modes, log rho).
3. <out-root>/tables/e1_vs_e4/e1_vs_e4.{md,json}: the side-by-side table (E4
   intervals from its beta_upgrade.json; E1 from step 1, or chain-mean
   summaries if step 1 has not been run).
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from _common import io, C, load_problem
from brho import plotting as P
from brho import diagnostics as D
from brho import mpsrf as MP
from brho.beta_upgrade import summarize

E1C, E4C, GR = "#c08a2e", "#2a6f97", "#9aa5ae"
LEVEL = 1.2


def tdir(root, kind, exp):
    d = os.path.join(root, kind, exp)
    os.makedirs(d, exist_ok=True)
    return d


def read_curve(root, exp, which):
    p = os.path.join(root, "tables", exp, f"mpsrf_{which}.csv")
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, data = rows[0], np.array(rows[1:], float)
    return head, data


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def e1_intervals(root, exp, stride):
    d = os.path.join(root, "results", exp)
    paths = io.chain_files(d)
    if not paths:
        print(f"{exp}: no chain files in {d}; intervals skipped "
              "(run this on the cluster, where the chains are)")
        return None
    man = read_json(os.path.join(d, "manifest.json"))
    cfg = C.from_dict(C.NaiveConfig, man["configs"]["NaiveConfig"])
    truth = man["truth"]
    keys = ("logbeta", "rho", "level", "phi")
    pooled = {k: [] for k in keys}
    L = None
    for p in paths:
        with np.load(p) as z:
            n = int(z["it"])
            L = n if L is None else min(L, n)
            for k in keys:
                pooled[k].append(np.asarray(z[k][:n], float))
    nb = min(int(cfg.n_burn), L - 1)
    X = {k: np.concatenate([x[nb:L:stride] for x in pooled[k]]) for k in keys}
    res = dict(exp=exp, n_chains=len(paths), iterations=int(L), n_burn=nb, stride=stride,
               n_pooled=int(len(X["rho"])),
               solves_per_iteration=int(getattr(cfg, "theta_repeat", 1)) + 1,
               prior=dict(log_beta=f"N({cfg.mu_logbeta}, {cfg.s_logbeta}^2)",
                          log_rho=f"N({cfg.mu_logrho}, {cfg.s_logrho}^2)"),
               beta=summarize(np.exp(X["logbeta"]), truth["beta_true"]),
               log_beta=summarize(X["logbeta"], np.log(truth["beta_true"])),
               rho=summarize(X["rho"], truth["rho_true"]),
               level=summarize(X["level"], truth["level_true"]),
               phi=summarize(X["phi"]),
               noise_floor=man["likelihood"]["noise_floor"])
    td = tdir(root, "tables", exp)
    with open(os.path.join(td, "e1_intervals.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(td, "e1_intervals.md"), "w", encoding="utf-8") as f:
        f.write(f"{exp}: {len(paths)} chains x {L} iterations, burn-in {nb}, pooled every "
                f"{stride}th iteration ({res['n_pooled']} draws)\n\n"
                "| quantity | median | 90% interval | sd | truth | covered |\n|---|---|---|---|---|---|\n")
        for name in ("beta", "log_beta", "rho", "level"):
            q = res[name]
            f.write(f"| {name} | {q['median']:.3f} | [{q['lo']:.3f}, {q['hi']:.3f}] | "
                    f"{q['sd']:.3f} | {q['truth']:.3f} | {q['covered']} |\n")
        f.write(f"\nmean Phi {res['phi']['mean']:.1f} (noise floor {res['noise_floor']:.1f})\n")
    print(open(os.path.join(td, "e1_intervals.md"), encoding="utf-8").read())
    return res


def e1_field(root, exp):
    """fig_e1_field.png: same panels as report_sampler.py's fig_e4_field.png."""
    old = os.environ.get("BRHO_ROOT")
    os.environ["BRHO_ROOT"] = os.path.abspath(root)
    try:
        prob, man = load_problem(exp)
        cfg = C.from_dict(C.NaiveConfig, man["configs"]["NaiveConfig"])
        syn, kle, n = prob.syn, prob.kle, prob.pc.n
        LB, RH, TH = [], [], []
        for p in io.chain_files(io.results_dir(exp)):
            with np.load(p) as z:
                keep = np.asarray(z["kept_it"]) >= cfg.n_burn
                LB.append(np.asarray(z["kept_logbeta"], float)[keep])
                RH.append(np.asarray(z["kept_rho"], float)[keep])
                TH.append(np.asarray(z["kept_theta"], float).reshape(len(keep), -1)[keep])
        lb, rh, th = np.concatenate(LB), np.concatenate(RH), np.concatenate(TH)
        fs = D.logK_summaries(lb, rh, th, kle, syn.logK_true)
        m_lk, sd_lk, lo_lk, hi_lk, cov90 = fs["mean"], fs["sd"], fs["lo"], fs["hi"], fs["coverage"]
        fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
        v0 = min(syn.logK_true.min(), m_lk.min()); v1 = max(syn.logK_true.max(), m_lk.max())
        P.field(ax[0], syn.logK_true, n, r"true $\log K$", vmin=v0, vmax=v1)
        P.field(ax[1], m_lk, n, r"posterior mean $\log K$ (E1)", vmin=v0, vmax=v1)
        P.field(ax[2], sd_lk, n, r"posterior sd $\log K$ (E1)", cmap="magma")
        jr = P.slice_row(n); xs = prob.phys.xc
        ax[3].fill_between(xs, P.line(lo_lk, n, jr), P.line(hi_lk, n, jr), color=P.C_BAND,
                           alpha=0.5, label="5–95%")
        ax[3].plot(xs, P.line(m_lk, n, jr), "b-", label="posterior mean")
        ax[3].plot(xs, P.line(syn.logK_true, n, jr), "k-", lw=2, label="truth")
        ax[3].set_xlabel(f"x (y = {xs[jr]:.2f})")
        ax[3].set_title(f"slice; pointwise 90% coverage {cov90:.0%}")
        ax[3].legend()
        P.save(fig, exp, "fig_e1_field")
        return dict(coverage90_logK=float(cov90), n_samples=int(len(rh)),
                    sd_logK_median=float(np.median(sd_lk)))
    finally:
        if old is None:
            os.environ.pop("BRHO_ROOT", None)
        else:
            os.environ["BRHO_ROOT"] = old


def e1_matched_curves(e1_root, e1_exp, e4_root, e4_exp, mode):
    """Recompute E1's MPSRF / PSRF curves with the window starting where E4's
    does (in forward solves, or in iterations).  Needs the E1 chain files."""
    s4 = read_json(os.path.join(e4_root, "tables", e4_exp, "mpsrf_summary.json"))
    s1 = read_json(os.path.join(e1_root, "tables", e1_exp, "mpsrf_summary.json"))
    it4 = s4["first_kept_iteration"] + (s4["iter_burn"] - 1) * s4["thinning"]
    target_it = (it4 * s4["solves_per_iteration"] / s1["solves_per_iteration"]
                 if mode == "solves" else it4)
    old = os.environ.get("BRHO_ROOT")
    os.environ["BRHO_ROOT"] = os.path.abspath(e1_root)
    try:
        import report_mpsrf
        try:
            run_ = report_mpsrf.load_kept(e1_exp)
        except FileNotFoundError as e:
            print(f"{e1_exp}: {e}; burn-matched curves skipped")
            return None
    finally:
        if old is None:
            os.environ.pop("BRHO_ROOT", None)
        else:
            os.environ["BRHO_ROOT"] = old
    its = run_["its"]
    thin = int(its[1] - its[0])
    idx = int(np.searchsorted(its, target_it))       # first window sample (0-based)
    iter_burn = idx + 1                              # MATLAB 1-based
    info = dict(mode=mode, e4_window_start_iteration=int(it4),
                e4_window_start_solves=int(it4 * s4["solves_per_iteration"]),
                e1_window_start_iteration=int(its[idx]),
                e1_window_start_solves=int(its[idx] * s1["solves_per_iteration"]),
                e1_iter_burn=iter_burn, skip=s4["skip"], step=s4["step"])
    td = tdir(e1_root, "tables", e1_exp)
    for name, arr in run_["sets"].items():
        cur = MP.mpsrf_curve(arr, skip=s4["skip"], step=s4["step"], iter_burn=iter_burn)
        x = its[cur["k"] - 1].astype(float) + thin
        extra = run_["extra"][name]
        r = arr.shape[2]
        labels = ([f"θ{i + 1}" for i in range(r - len(extra))] + extra) if extra \
            else [f"theta{i + 1}" for i in range(r)]
        np.savetxt(os.path.join(td, f"mpsrf_{name}_burn_matched.csv"),
                   np.column_stack([x, cur["mpsrf"], cur["psrf"], cur["psrf_i"]]),
                   delimiter=",", header="iteration,mpsrf,psrf_max," + ",".join(labels),
                   comments="", fmt="%.6g")
        conv = MP.convergence_iteration(x, cur["mpsrf"], LEVEL)
        info[name] = dict(mpsrf_final=float(cur["mpsrf"][-1]),
                          psrf_max_final=float(cur["psrf"][-1]),
                          mpsrf_convergence_iteration=conv)
    with open(os.path.join(td, "mpsrf_burn_matched.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(f"E1 burn-matched ({mode}): window starts at iteration {info['e1_window_start_iteration']} "
          f"({info['e1_window_start_solves']} solves; E4: {info['e4_window_start_solves']}); "
          f"theta MPSRF final {info['theta']['mpsrf_final']:.3f}")
    return info


def solves_axis(root, exp, data):
    s = read_json(os.path.join(root, "tables", exp, "mpsrf_summary.json"))
    return data[:, 0] * s["solves_per_iteration"], s


def fmt_q(q, nd=2):
    return f"{q['median']:.{nd}f} [{q['lo']:.{nd}f}, {q['hi']:.{nd}f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1-root", default="runs/cluster_kle")
    ap.add_argument("--e1-exp", default="e1_blocked")
    ap.add_argument("--e4-root", default="runs/cluster_kle_sig2e3")
    ap.add_argument("--e4-exp", default="e4_sampler")
    ap.add_argument("--out-root", default=None, help="default: --e4-root")
    ap.add_argument("--stride", type=int, default=10, help="pool every n-th E1 iteration")
    ap.add_argument("--e4ab-exp", default="e4_sampler_ab",
                    help="A+B ablation run in --e4-root, overlaid when its tables exist")
    ap.add_argument("--burn-match", default="solves", choices=("solves", "iterations", "none"),
                    help="start E1's convergence window where E4's starts (default: same solves)")
    a = ap.parse_args()
    out_root = a.out_root or a.e4_root

    # ---------------------------------------------------------- 1. E1 intervals
    iv = e1_intervals(a.e1_root, a.e1_exp, a.stride)
    if iv is not None:
        iv["field"] = e1_field(a.e1_root, a.e1_exp)
        with open(os.path.join(a.e1_root, "tables", a.e1_exp, "e1_intervals.json"), "w",
                  encoding="utf-8") as f:
            json.dump(iv, f, indent=1)
        print(f"E1 field: pointwise 90% coverage of log K {iv['field']['coverage90_logK']:.1%}")
    if iv is None:
        p = os.path.join(a.e1_root, "tables", a.e1_exp, "e1_intervals.json")
        iv = read_json(p) if os.path.exists(p) else None

    # ---------------------------------------------------------- 2. convergence
    fd = tdir(out_root, "figures", "e1_vs_e4")
    bm = None
    if a.burn_match != "none":
        bm = e1_matched_curves(a.e1_root, a.e1_exp, a.e4_root, a.e4_exp, a.burn_match)
        if bm is None:
            p = os.path.join(a.e1_root, "tables", a.e1_exp, "mpsrf_burn_matched.json")
            bm = read_json(p) if os.path.exists(p) else None
    suf = "_burn_matched" if bm else ""
    if a.burn_match != "none" and not bm:
        print("WARNING: E1 curves NOT burn-matched (no E1 chains here); figures use the "
              "committed E1 curves, which start at iteration 1800")
    h1, all1 = read_curve(a.e1_root, a.e1_exp, "all" + suf)
    _, th1 = read_curve(a.e1_root, a.e1_exp, "theta" + suf)
    h4, all4 = read_curve(a.e4_root, a.e4_exp, "all")
    _, th4 = read_curve(a.e4_root, a.e4_exp, "theta")
    x1, s1 = solves_axis(a.e1_root, a.e1_exp, all1)
    x4, s4 = solves_axis(a.e4_root, a.e4_exp, th4)
    have_ab = os.path.exists(os.path.join(a.e4_root, "tables", a.e4ab_exp, "mpsrf_theta.csv"))
    if have_ab:
        hab, allab = read_curve(a.e4_root, a.e4ab_exp, "all")
        _, thab = read_curve(a.e4_root, a.e4ab_exp, "theta")
        xab, sab = solves_axis(a.e4_root, a.e4ab_exp, thab)
        print(f"overlaying the A+B ablation {a.e4ab_exp} "
              f"(window from {xab[0] / 1e6:.3f}M solves; E4 {x4[0] / 1e6:.3f}M)")
    M = 1e6

    def finish(ax, title):
        ax.axhline(LEVEL, color="k", lw=1, ls="--")
        ax.text(ax.get_xlim()[1], LEVEL, f" {LEVEL}", va="center", fontsize=9)
        ax.set_yscale("log")
        yt = [1, 1.2, 1.5, 2, 3, 5, 10, 20]
        lo, hi = ax.get_ylim()
        yt = [t for t in yt if lo <= t <= hi]
        ax.set_yticks(yt); ax.set_yticklabels([f"{t:g}" for t in yt]); ax.minorticks_off()
        ax.set_xlabel("forward solves per chain (millions)")
        ax.grid(color="0.92", lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, fontsize=11, loc="left")

    burn_note = (f", window from {bm['e1_window_start_solves'] / M:.3f}M solves" if bm else "")

    def psrf_fig(head, data, x, title, fname):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        names = head[3:]
        for i, nm in enumerate(names):
            if nm.startswith("θ"):
                ax.plot(x / M, data[:, 3 + i], color=GR, lw=0.6, alpha=0.7)
        ax.plot([], [], color=GR, lw=1, label="θ modes 1–57 (each)")
        for nm, col, lab in (("logβ", "#b0563f", "log β"), ("logρ", E4C, "log ρ")):
            if nm in names:
                ax.plot(x / M, data[:, 3 + names.index(nm)], color=col, lw=2, label=lab)
        ax.plot(x / M, data[:, 2], color="k", lw=1, ls=":", label="max over entries")
        ax.legend(fontsize=9, frameon=False)
        finish(ax, title)
        ax.set_ylabel("PSRF (log scale)")
        fig.tight_layout(); fig.savefig(os.path.join(fd, fname), dpi=150); plt.close(fig)

    # (a) PSRF of every entry: E1, and E4 for reference
    psrf_fig(h1, all1, x1, f"E1 naive sampler: PSRF of every entry, {s1['n_chains']} chains"
             + burn_note, "fig_e1_psrf.png")
    x4a = all4[:, 0] * s4["solves_per_iteration"]
    psrf_fig(h4, all4, x4a, f"E4 3-move sampler: PSRF of every entry, {s4['n_chains']} chains",
             "fig_e4_psrf.png")
    if have_ab:
        psrf_fig(hab, allab, allab[:, 0] * sab["solves_per_iteration"],
                 f"E4 moves A+B only (no C): PSRF of every entry, {sab['n_chains']} chains",
                 "fig_e4ab_psrf.png")

    # (b) E1, MPSRF
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x1 / M, th1[:, 1], color=E1C, lw=2, label="θ (57 modes)")
    ax.plot(x1 / M, all1[:, 1], color=E1C, lw=1.2, ls="--", label="θ, log β, log ρ (nearly on top)")
    ax.legend(fontsize=9, frameon=False)
    finish(ax, f"E1 naive sampler: MPSRF, final {th1[-1, 1]:.2f} (θ)" + burn_note)
    ax.set_ylabel("MPSRF (log scale)")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_e1_mpsrf.png"), dpi=150); plt.close(fig)

    # (c) MPSRF of theta, E4 on top of E1
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x1 / M, th1[:, 1], color=E1C, lw=2,
            label=f"E1 naive (2 solves/it), final {th1[-1, 1]:.2f}")
    ax.plot(x4 / M, th4[:, 1], color=E4C, lw=2,
            label=f"E4 3-move on the curve (3 solves/it), final {th4[-1, 1]:.2f}")
    if have_ab:
        ax.plot(xab / M, thab[:, 1], color="#7a8691", lw=2, ls="--",
                label=f"E4 moves A+B only, no C (2 solves/it), final {thab[-1, 1]:.2f}")
    if bm:
        ax.axvline(bm["e4_window_start_solves"] / M, color="0.5", lw=0.8, ls="--",
                   label="burn-in end, both runs")
    it_c = s4["theta"].get("mpsrf_convergence_iteration")
    if it_c:
        xc = it_c * s4["solves_per_iteration"] / M
        ax.axvline(xc, color=E4C, lw=0.8, ls=":")
        ax.text(xc, 0.97, f" E4 below {LEVEL}\n at {xc:.2f}M solves", color=E4C,
                fontsize=9, va="top", transform=ax.get_xaxis_transform())
    ax.legend(fontsize=9, frameon=False)
    finish(ax, "MPSRF of θ at equal cost: same truth, data and σ_rel"
           + (" · same burn-in (solves)" if bm and bm["mode"] == "solves" else
              " · same burn-in (iterations)" if bm else ""))
    ax.set_ylabel("MPSRF of θ (log scale)")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_e1_vs_e4_mpsrf.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- 3. table
    up = read_json(os.path.join(a.e4_root, "tables", a.e4_exp, "beta_upgrade.json"))
    e4s = read_json(os.path.join(a.e4_root, "tables", a.e4_exp, "e4_summary.json"))
    e1s = read_json(os.path.join(a.e1_root, "tables", a.e1_exp, "e1_summary.json"))
    td = tdir(out_root, "tables", "e1_vs_e4")
    if iv:
        e1_cells = dict(beta=fmt_q(iv["beta"]), rho=fmt_q(iv["rho"]),
                        level=fmt_q(iv["level"], 3), sd=f"{iv['log_beta']['sd']:.3f}",
                        phi=f"{iv['phi']['mean']:.1f}")
    else:
        lb = np.array(e1s["logbeta_stats"]["chain_means"])
        e1_cells = dict(beta=f"mean {np.exp(lb.mean()):.2f} (run step 1 for intervals)",
                        rho=f"mean {np.mean(e1s['rho_stats']['chain_means']):.2f}",
                        level=f"mean {np.mean(e1s['level_stats']['chain_means']):.3f}",
                        sd="—", phi=f"{e1s['phi_stats']['mean']:.1f}")
    rows = [
        ("Solves per chain", f"{e1s['iterations'] * s1['solves_per_iteration'] / M:.1f}M",
         f"{e4s['n_total'] * s4['solves_per_iteration'] / M:.1f}M", ""),
        ("θ MPSRF, final", f"{th1[-1, 1]:.2f}", f"{th4[-1, 1]:.2f}", ""),
        ("log β PSRF", f"{e1s['logbeta_stats']['psrf']:.2f}", "— (β on the curve)", ""),
        ("ρ", e1_cells["rho"], fmt_q(up["rho_hard"]), "1.5"),
        ("Level", e1_cells["level"], fmt_q(up["level_hard"], 3), f"{up['truth']['level']:.3f}"),
        ("β", e1_cells["beta"], f"{fmt_q(up['beta_hard'])} → {fmt_q(up['beta_upgraded'])}", "2.3"),
        ("sd of log β", e1_cells["sd"],
         f"{up['log_beta_sd']['hard']:.3f} → {up['log_beta_sd']['upgraded']:.3f}", ""),
        (f"Mean Φ (floor {e4s['noise_floor']:.1f})", e1_cells["phi"], f"{e4s['mean_phi']:.1f}", ""),
    ]
    if have_ab:
        upab = read_json(os.path.join(a.e4_root, "tables", a.e4ab_exp, "beta_upgrade.json"))
        e4ab = read_json(os.path.join(a.e4_root, "tables", a.e4ab_exp, "e4_summary.json"))
        ab = ["", f"{e4ab['n_total'] * sab['solves_per_iteration'] / M:.1f}M",
              f"{thab[-1, 1]:.2f}", "— (β on the curve)", fmt_q(upab["rho_hard"]),
              fmt_q(upab["level_hard"], 3),
              f"{fmt_q(upab['beta_hard'])} → {fmt_q(upab['beta_upgraded'])}",
              f"{upab['log_beta_sd']['hard']:.3f} → {upab['log_beta_sd']['upgraded']:.3f}",
              f"{e4ab['mean_phi']:.1f}"]
        rows = [r[:3] + (ab[i + 1],) + r[3:] for i, r in enumerate(rows)]
    with open(os.path.join(td, "e1_vs_e4.md"), "w", encoding="utf-8") as f:
        f.write("E1 naive (log beta ~ N(0,1), log rho ~ N(0,0.75^2)) vs E4 3-move on the curve "
                "(hard -> level swap). Same KLE truth, data, sigma_rel 2e-3, 64 chains.\n\n"
                + ("| | E1 naive | E4 3-move | E4 A+B only | truth |\n|---|---|---|---|---|\n"
                   if have_ab else "| | E1 naive | E4 3-move | truth |\n|---|---|---|---|\n"))
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    with open(os.path.join(td, "e1_vs_e4.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, e1_intervals=iv), f, indent=1)
    print(open(os.path.join(td, "e1_vs_e4.md"), encoding="utf-8").read())
    print("figures:", fd)


if __name__ == "__main__":
    main()
