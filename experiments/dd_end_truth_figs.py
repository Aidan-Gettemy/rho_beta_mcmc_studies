#!/usr/bin/env python3
"""Repeatability figure: the ell 0.1, 4 x 4 column over the three truths (2026, 2027, 2028).
For each data type and metric (error ratio, sd ratio to the global pCN of the same truth) one dot per truth for global pCN in the MSM
basis, DD primal, DD dual and DD pooled, with the mean as a bar.  Reads dd_end/three_truth.csv (dd_end_truth_spread.py);
writes dd_end/truth_spread.png.  The other summary figures show truth 2026 only."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "msm_local_kle_2026-11", "dd_end")
t = pd.read_csv(os.path.join(D, "three_truth.csv")).set_index(["flux", "family"])
FAM = [("msm", "MSM basis"), ("primal", "DD primal"), ("dual", "DD dual"), ("pooled", "DD pooled")]
TR = {2026: ("o", "#1f77b4"), 2027: ("s", "#d95f02"), 2028: ("^", "#1b9e77")}
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for j, flux in enumerate([False, True]):
    for i, (col, name) in enumerate([("err_ratio", "error ratio to global (same truth)"), ("sd_ratio", "sd ratio to global (same truth)")]):
        a = ax[i, j]
        for k, (fam, lab) in enumerate(FAM):
            r = t.loc[(flux, fam)]
            for tr, (mk, c) in TR.items():
                a.plot([k + (tr - 2027) * 0.12], [r[f"{col}_{tr}"]], marker=mk, color=c, ls="", ms=7, label=f"truth {tr}" if (i, j, k) == (0, 0, 0) else None)
            a.hlines(r[f"{col}_mean"], k - 0.22, k + 0.22, color="k", lw=1.5)
        a.axhline(1, color="k", lw=.8, ls="--")
        a.set_xticks(range(len(FAM))); a.set_xticklabels([l for _, l in FAM])
        a.set_title("pressure + flux" if flux else "pressure only"); a.set_ylabel(name)
ax[0, 0].legend(fontsize=8)
fig.text(.5, .005, "ell 0.1, 4 x 4 blocks; black bar: mean over the three truths", ha="center", fontsize=8)
plt.tight_layout(rect=(0, .02, 1, 1)); plt.savefig(os.path.join(D, "truth_spread.png"), dpi=160)
