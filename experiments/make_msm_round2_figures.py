"""Round-2 comparison figures from committed outputs only (no chains):
  fig_conv_bar.png   iterations to MPSRF<1.2 and final MPSRF per run
  fig_sd_ratio.png   mean posterior sd of log k relative to experiment 1 (global pCN)
  fig_logk_sd_grid.png / fig_logk_mean_grid.png   contourf/jet posterior maps, shared scales
    BRHO_ROOT=runs/msm_local_kle_2026-11 python experiments/make_msm_round2_figures.py"""
import _common  # noqa: F401
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msm_fieldplots as FP

R = os.environ.get("BRHO_ROOT", "runs/msm_local_kle_2026-11")
OUT = os.path.join(R, "compare2"); os.makedirs(OUT, exist_ok=True)
# (dir, label, mpsrf key used, field key, correct-stationary-law?)
EX = [("exp1_global_s50", "1 global", "theta_range58_common", "run", True),
      ("exp3b_primal_aug_s50", "3b aug", "theta_range58_common", "run", True),
      ("exp2_primal_raw_s50", "2 raw d4", "theta_range58", "run", False),
      ("exp2_6_primal_raw_d6", "2.6 raw d6", "theta", "run", False),
      ("exp2_8_primal_raw_d8", "2.8 raw d8", "theta", "run", False),
      ("exp3a_primal_pinv_s50", "3a pinv", "theta_range58", "run", False),
      ("exp4_mixture", "4 mix d4", "theta_range58_common", "run", False),
      ("exp4_8_mixture_d8", "4.8 mix d8", "theta", "run", False),
      ("exp4b_mixture_innovation", "4b mix innov", "theta_range58_common", "run", False),
      ("exp5b_sequence_exact", "5b seq (ph2)", "theta_phase2_range58_common", "pooled", False),
      ("exp6a_alt_L1", "6a alt L1", "theta_range58_common", "run", False),
      ("exp6b_alt_L10", "6b alt L10", "theta_range58_common", "run", False),
      ("exp6c_alt_L100", "6c alt L100", "theta_range58_common", "run", False),
      ("exp6d_alt_L1_nooffset", "6d alt L1 no off.", "theta_range58_common", "run", False)]
rows = []
for d, lab, mk, fk, ok in EX:
    P = json.load(open(f"{R}/{d}/post/post_summary.json"))
    m = P["mpsrf"][mk]; sw = 50000 if "seq" in d else 0
    F = dict(np.load(f"{R}/{d}/post/fields.npz"))
    rows.append(dict(d=d, lab=lab, conv=m["convergence_iteration"], fin=m["mpsrf_final"],
                     relerr=P["relerr_post_M_median"], sd=P["fields"][fk]["logk_sd_mean"],
                     err=P["fields"][fk]["logk_mean_error_rms"], F=F, fk=fk, sw=sw))
ref = rows[0]["sd"]
for r in rows: r["ratio"] = r["sd"] / ref
json.dump([{k: v for k, v in r.items() if k != "F"} for r in rows], open(f"{OUT}/summary_rows.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
x = np.arange(len(rows)); lab = [r["lab"] for r in rows]
for i, r in enumerate(rows):
    c = r["conv"]; stuck = (c is None) or r["fin"] >= 1.2
    col = "C2" if i < 2 else ("C3" if stuck else "C0")
    v = (c - r["sw"]) / 1e3 if c else 0
    ax[0].bar(i, v, color=col, hatch="//" if stuck else None, edgecolor="k", lw=.4)
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=60, ha="right", fontsize=8)
ax[0].set_ylabel("iterations to MPSRF < 1.2 (thousands)"); ax[0].set_title("speed of the 58-coordinate MPSRF (common coordinates)")
ax[0].set_ylim(0, 72)
ax[0].text(.01, .99, "green = exact posterior law   blue = other   red hatched = dips below 1.2 but\nfinal MPSRF >= 1.2 (false convergence)", transform=ax[0].transAxes, va="top", fontsize=7.5)
ax[1].bar(x, [r["ratio"] for r in rows], color=["C2", "C2"] + ["C0"] * (len(rows) - 2), edgecolor="k", lw=.4)
ax[1].axhline(1, color="k", ls=":"); ax[1].set_xticks(x); ax[1].set_xticklabels(lab, rotation=60, ha="right", fontsize=8)
ax[1].set_ylabel("mean posterior sd of log k / experiment 1"); ax[1].set_title("fidelity of the posterior spread")
ax[1].set_ylim(0, 1.15)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_conv_bar.png", dpi=140); plt.close(fig)

SEL = [0, 1, 2, 5, 6, 8, 10, 13]
n = int(rows[0]["F"]["grid_n"])
for key, nm, cm, ttl in (("logk_sd", "fig_logk_sd_grid.png", FP.CMAP_STD, "posterior sd of log k"),
                         ("logk_mean", "fig_logk_mean_grid.png", FP.CMAP_MAIN, "posterior mean of log k")):
    vals = [rows[i]["F"][f"{rows[i]['fk']}_{key}"] for i in SEL]
    if key == "logk_mean":
        tr = rows[0]["F"]["logk_true"]; vals_all = vals + [tr]
    else: vals_all = vals
    lo, hi = min(v.min() for v in vals_all), max(v.max() for v in vals_all)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.6), constrained_layout=True)
    for a, i, v in zip(ax.ravel(), SEL, vals):
        cs = FP.contour(a, v, n, cm, lo, hi)
        a.set_title(f"{rows[i]['lab']} (mean {v.mean():.3f})" if key == "logk_sd" else rows[i]["lab"], fontsize=10)
    fig.colorbar(cs, ax=ax, fraction=.025, label=ttl); fig.suptitle(ttl + f" (10,000 pooled samples)")
    fig.savefig(f"{OUT}/{nm}", dpi=130); plt.close(fig)
print("ok", OUT)
for r in rows: print(f"{r['lab']:18s} conv={r['conv']} fin={r['fin']:.3f} sdratio={r['ratio']:.3f} err={r['err']:.3f}")
