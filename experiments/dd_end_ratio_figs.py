#!/usr/bin/env python3
"""Summary figures from dd_end/master_table.csv (truth 2026, eps x 1; the stage 6 truths are shown as spread bars at ell 0.1, 4 x 4):
  ratio_vs_ell.png          DD primal / global error and sd ratio against ell, one line per block count, both data types
  ratio_vs_side_over_ell.png  the same points against block side over ell (1 / (blocks x ell)), coloured by ell
  ladder_bars.png           global = 1, MSM basis, DD primal, DD dual, pooled at 4 x 4 for every ell and data type (error and sd ratio)
The dual and pooled 8 x 8 rows with flux are excluded from conclusions and left out."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "msm_local_kle_2026-11", "dd_end")
d = pd.read_csv(os.path.join(D, "master_table.csv"))
d = d[(d.truth == 2026) & (d.eps_factor == 1.0) & d.sd_ratio.notna() & ~d.label.str.contains("DD-H")].drop_duplicates(["label", "variant"])
d = d[~((d.blocks == 8) & d.variant.isin(["dual", "pooled"]) & d.flux & (d.ell < 0.2))]
t3 = pd.read_csv(os.path.join(D, "three_truth.csv")).set_index(["flux", "family"])
COL = {2: "#1f77b4", 4: "#d95f02", 8: "#1b9e77"}
M = [("err_ratio", "error ratio to global"), ("sd_ratio", "sd ratio to global")]
data = {False: "pressure only", True: "pressure + flux"}

# 1. against ell
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for j, flux in enumerate(data):
    for i, (col, name) in enumerate(M):
        a = ax[i, j]
        for nb in (2, 4, 8):
            g = d[(d.variant == "primal") & (d.flux == flux) & (d.blocks == nb)].sort_values("ell")
            a.plot(g.ell, g[col], marker="o", color=COL[nb], label=f"{nb} x {nb} blocks")
        r = t3.loc[(flux, "primal")]
        a.errorbar([0.1], [r[f"{col}_mean"]], yerr=[r[f"{col}_range"] / 2], color=COL[4], capsize=4, lw=0, elinewidth=3, alpha=.4)
        a.axhline(1, color="k", lw=.8, ls="--"); a.set_xscale("log"); a.set_xticks([.05, .1, .2]); a.set_xticklabels(["0.05", "0.1", "0.2"]); a.minorticks_off()
        a.set_title(data[flux]); a.set_ylabel(name)
        if i == 1: a.set_xlabel("correlation length ell")
ax[0, 0].legend(); fig.text(.5, .005, "faded bar: spread over three truths (4 x 4, ell 0.1)", ha="center", fontsize=8)
plt.tight_layout(rect=(0, .02, 1, 1)); plt.savefig(os.path.join(D, "ratio_vs_ell.png"), dpi=160); plt.close()

# 2. against block side over ell
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
ELLC = {0.2: "#7570b3", 0.1: "#e7298a", 0.05: "#66a61e"}
for j, flux in enumerate(data):
    for i, (col, name) in enumerate(M):
        a = ax[i, j]
        g = d[(d.variant == "primal") & (d.flux == flux)].copy()
        g["x"] = 1.0 / (g.blocks * g.ell)
        for ell, h in g.groupby("ell"):
            a.scatter(h.x, h[col], color=ELLC[ell], label=f"ell {ell}", s=40)
            for _, r in h.iterrows():
                a.annotate(f"{int(r.blocks)}x{int(r.blocks)}", (r.x, r[col]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a.axhline(1, color="k", lw=.8, ls="--"); a.set_xscale("log"); a.set_xticks([1.25, 2.5, 5, 10]); a.set_xticklabels(["1.25", "2.5", "5", "10"]); a.minorticks_off()
        a.set_title(data[flux]); a.set_ylabel(name)
        if i == 1: a.set_xlabel("block side over ell")
ax[0, 0].legend(); plt.tight_layout(); plt.savefig(os.path.join(D, "ratio_vs_side_over_ell.png"), dpi=160); plt.close()

# 3. ladder bars at 4 x 4
fam = [("msm", "MSM basis", "#bbbbbb"), ("primal", "DD primal", "#1f77b4"), ("dual", "DD dual", "#d95f02"), ("pooled", "pooled", "#1b9e77")]
fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for j, flux in enumerate(data):
    for i, (col, name) in enumerate(M):
        a = ax[i, j]
        for k, (v, lab, c) in enumerate(fam):
            ys = []
            for ell in (0.2, 0.1, 0.05):
                g = d[(d.variant == v) & (d.flux == flux) & (d.blocks == 4) & (d.ell == ell)]
                ys.append(float(g[col].iloc[0]) if len(g) else np.nan)
            a.bar(np.arange(3) + (k - 1.5) * 0.2, ys, 0.2, color=c, label=lab)
        a.axhline(1, color="k", lw=.8, ls="--"); a.set_xticks(range(3)); a.set_xticklabels(["ell 0.2", "ell 0.1", "ell 0.05"])
        a.set_title(data[flux] + ", 4 x 4"); a.set_ylabel(name)
ax[0, 0].legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(D, "ladder_bars.png"), dpi=160)
