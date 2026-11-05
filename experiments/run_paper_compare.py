"""Compare the eight experiments of the paper-MSM study (+ prior test) from <root>/e*/{summary.json,post/*}.
    python experiments/run_paper_compare.py [--root runs/paper_msm_2026-11]
Writes <root>/compare/{table.json, table.tex, fig_z_sd.png, fig_logk_sd.png, fig_prior.png}."""
import _common  # noqa: F401
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPS = ["e0_global_all", "e1_global_one", "e2_raw_all", "e3_proj_all", "e4_proj_one", "e5_smooth_all", "e6_smooth_one", "e7_smooth_sub"]
LAB = ["E0 global, all", "E1 global, one", "E2 raw, all", "E3 proj, all", "E4 proj, one", "E5 smooth, all", "E6 smooth, one", "E7 smooth, subdomain"]


def kl(m1, S1, m0, S0):
    k, S0i, d = len(m1), np.linalg.inv(S0), m0 - m1
    return 0.5 * (np.trace(S0i @ S1) + d @ S0i @ d - k + np.linalg.slogdet(S0)[1] - np.linalg.slogdet(S1)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/paper_msm_2026-11")
    ap.add_argument("--control", default="runs/grid_arms_2026-11/arm1_check_n16/post")
    a = ap.parse_args()
    out = os.path.join(a.root, "compare"); os.makedirs(out, exist_ok=True)
    Tc = np.load(os.path.join(a.control, "theta_stats.npz")); Fc = np.load(os.path.join(a.control, "fields.npz"))
    rows, T, F = {}, {}, {}
    for e in EXPS:
        d = os.path.join(a.root, e)
        S, P = json.load(open(d + "/summary.json")), json.load(open(d + "/post/post_summary.json"))
        T[e], F[e] = np.load(d + "/post/theta_stats.npz"), np.load(d + "/post/fields.npz")
        m = P["mpsrf"]["common"]
        rows[e] = dict(acc=P["acceptance"]["mean"], accmin=P["acceptance"]["group_min"], accmax=P["acceptance"]["group_max"],
                       relerr=P["relerr_post_M_median"], mpsrf=m["mpsrf_final"], psrf=m["psrf_max_final"], conv=m["convergence_iteration"],
                       mpsrf_nat=P["mpsrf"]["native"]["mpsrf_final"], zsd=P["z_post"]["sd_mean"], zerr=P["z_post"]["rms_mean_error"],
                       klc=float(kl(T[e]["mean"], T[e]["cov"], Tc["mean"], Tc["cov"])),
                       kcl=float(kl(Tc["mean"], Tc["cov"], T[e]["mean"], T[e]["cov"])),
                       lk_diff=float(np.sqrt(np.mean((F[e]["run_logk_mean"] - Fc["run_logk_mean"]) ** 2))),
                       lk_sd=float(F[e]["run_logk_sd"].mean() / Fc["run_logk_sd"].mean()),
                       p_sd=float(F[e]["run_p_sd"].mean() / Fc["run_p_sd"].mean()), sec=S["sec_per_chain"])
    json.dump(rows, open(out + "/table.json", "w"), indent=1)
    n = int(np.sqrt(len(Fc["run_logk_mean"])))
    fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.1
    ax.bar(np.arange(20) - 4 * w, Tc["sd"], w, label="control (arm 1, 16x16)", color="k")
    for i, e in enumerate(EXPS):
        ax.bar(np.arange(20) + (i - 3) * w, T[e]["sd"], w, label=LAB[i])
    ax.set_xlabel("global mode"); ax.set_ylabel("posterior sd of z"); ax.legend(fontsize=6, ncol=3); fig.tight_layout()
    fig.savefig(out + "/fig_z_sd.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(2, 5, figsize=(11, 4.6))
    for ax, (nm, v) in zip(axs.ravel(), [("control", Fc["run_logk_sd"])] + [(LAB[i], F[e]["run_logk_sd"]) for i, e in enumerate(EXPS)]):
        im = ax.imshow(v.reshape(n, n), origin="lower", vmin=0.2, vmax=1.0); ax.set_title(nm, fontsize=8); ax.axis("off")
    axs.ravel()[-1].axis("off"); fig.colorbar(im, ax=axs, shrink=.6, label="posterior sd of log k"); fig.savefig(out + "/fig_logk_sd.png", dpi=150); plt.close(fig)
    pr = json.load(open(a.root + "/prior_test/prior_summary.json"))
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    for j, (k, t) in enumerate([("var_band", "variance in interface band"), ("corr_across", "correlation across interfaces")]):
        for s, mk in ((0.2, "o"), (0.7, "s")):
            axs[j].plot(range(8), [pr["runs"][f"{e}@s={s}"][k] for e in EXPS], mk, label=f"s={s}")
        axs[j].axhline(pr["target"][k], color="k", ls=":"); axs[j].set_xticks(range(8)); axs[j].set_xticklabels([f"E{i}" for i in range(8)]); axs[j].set_title(t, fontsize=9)
    axs[0].legend(); fig.tight_layout(); fig.savefig(out + "/fig_prior.png", dpi=150); plt.close(fig)
    for e in EXPS:
        r = rows[e]; print(f"{e:15s} klc {r['klc']:.2f} kcl {r['kcl']:.2f} lkdiff {r['lk_diff']:.3f} lksd {r['lk_sd']:.2f} psd {r['p_sd']:.2f} sec {r['sec']:.0f}")


main()
