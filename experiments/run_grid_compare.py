"""
Compare the nine configurations (3 arms x grids 16, 32, 64) of the grid-refinement study from
<root>/<arm>_n<N>/{summary.json, post/post_summary.json, post/theta_stats.npz}.  No chains needed.

    python experiments/run_grid_compare.py [--root runs/grid_arms_2026-11]
Writes <root>/compare/{table.md, table.tex, fig_scaling.png, fig_theta_sd.png}.

Columns: observations, noise sd, acceptance and adapted step size after M, median relative error and Phi
after M (against their values at the truth), iterations to MPSRF < 1.2 (group convention, window from
iteration 1000, 64 chains) with the final MPSRF / max PSRF, the mean posterior sd of theta, the rms error of the
posterior mean of theta, and the Gaussian-approximation KL divergence of the theta posterior from that of arm 1
at 16 x 16 (mean and covariance of the pooled post-M samples).
"""
import _common  # noqa: F401
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = [("arm1_check", "1 check (fixed normalised sigma)"), ("arm2_fixed", "2 fixed 128 obs"),
        ("arm3_abs", "3 abs (fixed sigma_abs)")]
GRIDS = (16, 32, 64)
COL = {"arm1_check": "C0", "arm2_fixed": "C2", "arm3_abs": "C3"}


def gauss_kl(m1, S1, m0, S0):
    """KL( N(m1, S1) || N(m0, S0) )."""
    k = len(m1)
    S0i = np.linalg.inv(S0)
    d = m0 - m1
    return 0.5 * (np.trace(S0i @ S1) + d @ S0i @ d - k + np.linalg.slogdet(S0)[1] - np.linalg.slogdet(S1)[1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs/grid_arms_2026-11")
    a = ap.parse_args(argv)
    R = {}
    for arm, _ in ARMS:
        for n in GRIDS:
            d = os.path.join(a.root, f"{arm}_n{n}")
            if not os.path.exists(os.path.join(d, "post", "post_summary.json")):
                continue
            R[(arm, n)] = dict(S=json.load(open(os.path.join(d, "summary.json"))),
                               P=json.load(open(os.path.join(d, "post", "post_summary.json"))),
                               T=dict(np.load(os.path.join(d, "post", "theta_stats.npz"))))
    if not R:
        raise SystemExit("no runs found under " + a.root)
    ref = R.get(("arm1_check", 16))
    out = os.path.join(a.root, "compare"); os.makedirs(out, exist_ok=True)
    rows, md, tex = [], [], []
    md.append("| arm | grid | n obs | sigma_p | acc | s (adapted) | median relerr (truth) | median Phi (truth) | "
              "iter. to MPSRF<1.2 | MPSRF end | PSRF end | mean sd theta | rms mean err theta | KL to own 16x16 | KL to arm1@16 | s/chain |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for (arm, n), r in sorted(R.items(), key=lambda kv: ([x[0] for x in ARMS].index(kv[0][0]), kv[0][1])):
        S, P, T = r["S"], r["P"], r["T"]
        mp = P["mpsrf"]["theta"]
        kl = gauss_kl(T["mean"], T["cov"], ref["T"]["mean"], ref["T"]["cov"]) if ref is not None else float("nan")
        own = R.get((arm, 16))
        kl_own = gauss_kl(T["mean"], T["cov"], own["T"]["mean"], own["T"]["cov"]) if own is not None else float("nan")
        conv = mp.get("convergence_iteration")
        row = dict(arm=arm, n=n, n_obs=S["data"]["n_obs"], sigma_p=S["data"]["sigma_p"], acc=S["post_acc_mean"],
                   s=S["s_final_median"], relerr=P["relerr_post_M_median"], relerr_truth=P["relerr_at_truth"],
                   phi=P["phi_post_M_median"], phi_truth=P["phi_at_truth"], conv=conv, mpsrf=mp["mpsrf_final"],
                   psrf=mp["psrf_max_final"], sd=P["theta_post"]["sd_mean"], err=P["theta_post"]["rms_mean_error"],
                   kl=kl, kl_own=kl_own, sec=S["sec_per_chain"])
        rows.append(row)
        cs = "none" if conv is None else f"{conv/1e3:.1f}k"
        md.append(f"| {arm} | {n} | {row['n_obs']} | {row['sigma_p']:.2e} | {row['acc']:.3f} | {row['s']:.4f} | "
                  f"{row['relerr']:.4f} ({row['relerr_truth']:.4f}) | {row['phi']:.1f} ({row['phi_truth']:.1f}) | {cs} | "
                  f"{row['mpsrf']:.3f} | {row['psrf']:.3f} | {row['sd']:.4f} | {row['err']:.3f} | {kl_own:.3f} | {kl:.3f} | {row['sec']:.0f} |\n")
        tex.append(f"{arm.replace('_', ' ')} & {n} & {row['n_obs']} & {row['acc']:.2f} & {row['s']:.4f} & "
                   f"{row['relerr']:.4f} & {cs} & {row['mpsrf']:.3f} & {row['sd']:.3f} & {kl_own:.2f} \\\\\n")
    open(os.path.join(out, "table.md"), "w").write("".join(md))
    open(os.path.join(out, "table.tex"), "w").write("".join(tex))
    json.dump(rows, open(os.path.join(out, "table.json"), "w"), indent=1)
    print("".join(md))

    fig, ax = plt.subplots(2, 3, figsize=(14, 7.2), constrained_layout=True)
    spec = [("conv", "iterations to MPSRF < 1.2", 1e-3), ("mpsrf", "MPSRF at the end of the run", 1),
            ("s", "adapted pCN step size s", 1), ("relerr", "median relative error after adaptation", 1),
            ("sd", "mean posterior sd of theta", 1), ("kl_own", "KL(theta posterior || same arm at 16x16)", 1)]
    for a_, (k, ttl, sc) in zip(ax.ravel(), spec):
        for arm, lab in ARMS:
            pts = [(r["n"], r[k]) for r in rows if r["arm"] == arm]
            xs = [p[0] for p in pts if p[1] is not None]; ys = [p[1] * sc for p in pts if p[1] is not None]
            a_.plot(xs, ys, "o-", color=COL[arm], label=lab)
            if k == "conv":
                for p in pts:
                    if p[1] is None:
                        a_.plot([p[0]], [a_.get_ylim()[1] * 0.95], "x", color=COL[arm], ms=9)
        a_.set_xscale("log", base=2); a_.set_xticks(GRIDS); a_.set_xticklabels([f"{n}x{n}" for n in GRIDS])
        a_.set_title(ttl, fontsize=10)
        if k in ("s",): a_.set_yscale("log")
        if k == "mpsrf": a_.axhline(1.2, color="0.5", ls=":")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Global pCN on the Darcy slab (ell = 0.2, 20 modes): grid refinement, three observation arms. "
                 "x = no run reached MPSRF < 1.2")
    fig.savefig(os.path.join(out, "fig_scaling.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True, sharey=True)
    lsty = {16: "-", 32: "--", 64: ":"}
    for a_, (arm, lab) in zip(ax, ARMS):
        for n in GRIDS:
            if (arm, n) in R:
                a_.plot(R[(arm, n)]["T"]["sd"], lsty[n], color=COL[arm], label=f"{n}x{n}")
        if ref is not None:
            a_.plot(ref["T"]["sd"], "k", lw=.6, alpha=.5, label="arm 1, 16x16")
        a_.set_title(lab, fontsize=10); a_.set_xlabel("KLE coordinate"); a_.legend(fontsize=8)
    ax[0].set_ylabel("posterior sd of theta_k (prior sd 1)")
    fig.savefig(os.path.join(out, "fig_theta_sd.png"), dpi=130); plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
