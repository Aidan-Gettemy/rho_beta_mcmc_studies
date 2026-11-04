"""
Side-by-side comparison of the six MSM experiments from <root>/<exp>/post/fields.npz and
post_summary.json: log kappa sd maps and mean maps on shared scales (contourf, jet), pressure
sd maps, and a table.  No chains needed.

    python experiments/run_msm_compare.py [--root runs/msm_local_kle_2026-11] [--cmap-std jet]
Writes <root>/compare/{fig_logk_sd.png, fig_logk_mean.png, fig_p_sd.png, compare.md}.
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

import msm_fieldplots as FP  # noqa: E402

EXPS = [("exp1_global", "1 global pCN", "run"), ("exp2_primal_raw", "2 primal raw", "run"),
        ("exp3a_primal_pinv", "3a pinv", "run"), ("exp3b_primal_aug", "3b + null term", "run"),
        ("exp4_mixture", "4 mixture", "run"), ("exp5_sequence", "5 primal->dual", "pooled")]


def panel_grid(Fs, key, n, path, cmap, title, shared=True):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.6), constrained_layout=True)
    vals = [F[key] for _, F in Fs]
    lo, hi = (min(v.min() for v in vals), max(v.max() for v in vals)) if shared else (None, None)
    for a, (lab, F), v in zip(ax.ravel(), Fs, vals):
        cs = FP.contour(a, v, n, cmap, lo, hi)
        a.set_title(f"{lab}   (mean {v.mean():.3f})" if "sd" in key else lab, fontsize=10)
    fig.colorbar(cs, ax=ax, fraction=.03, label=title)
    fig.suptitle(title); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs/msm_local_kle_2026-11")
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--cmap-std", default=None)
    ap.add_argument("--levels", type=int, default=None)
    a = ap.parse_args(argv)
    if a.levels: FP.LEVELS = a.levels
    out = os.path.join(a.root, "compare"); os.makedirs(out, exist_ok=True)
    Fs, rows = [], []
    for e, lab, ph in EXPS:
        F = dict(np.load(os.path.join(a.root, e, "post", "fields.npz")))
        S = json.load(open(os.path.join(a.root, e, "post", "post_summary.json")))
        Fs.append((lab, {"logk_sd": F[f"{ph}_logk_sd"], "logk_mean": F[f"{ph}_logk_mean"], "p_sd": F[f"{ph}_p_sd"],
                         "truth": F["logk_true"]}))
        n = int(F["grid_n"]); fl = S["fields"][ph]
        rows.append((lab, fl["logk_sd_mean"], fl["p_sd_mean"], fl["logk_mean_error_rms"], S["relerr_post_M_median"]))
    panel_grid(Fs, "logk_sd", n, os.path.join(out, "fig_logk_sd.png"), a.cmap_std or FP.CMAP_STD, "log permeability (std)")
    panel_grid(Fs, "p_sd", n, os.path.join(out, "fig_p_sd.png"), a.cmap_std or FP.CMAP_STD, "pressure (std)")
    panel_grid(Fs, "logk_mean", n, os.path.join(out, "fig_logk_mean.png"), a.cmap or FP.CMAP_MAIN, "log permeability (posterior mean)")
    r0 = rows[0]
    with open(os.path.join(out, "compare.md"), "w") as f:
        f.write("| experiment | mean sd log k | ratio to exp 1 | mean sd p | rms(mean log k - truth) | median rel. err. |\n|---|---|---|---|---|---|\n")
        for lab, sk, sp, er, re in rows:
            f.write(f"| {lab} | {sk:.3f} | {sk / r0[1]:.2f} | {sp:.4f} | {er:.3f} | {re:.4f} |\n")
    print(open(os.path.join(out, "compare.md")).read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
