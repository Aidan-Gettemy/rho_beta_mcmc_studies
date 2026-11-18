#!/usr/bin/env python3
"""Plain MSM-basis pCN vs global pCN: ratio of iterations to MPSRF < 1.2 against the global KLE dimension (from master_table.csv).
    python experiments/dd_end_msm_speedup.py   ->  runs/msm_local_kle_2026-11/dd_end/msm_speedup_vs_dimension.png"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "msm_local_kle_2026-11", "dd_end")
R = {0.2: 19, 0.1: 64, 0.05: 208}                       # global KLE dimension per ell
d = pd.read_csv(os.path.join(D, "master_table.csv"))
d = d[d.conv_it.notna() & d.method.isin(["pcn", "msm"])].drop_duplicates(["label"])
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, (flux, t) in zip(ax, [(False, "pressure only"), (True, "pressure + flux")]):
    for nb, mk in [(2, "o"), (4, "s"), (8, "^")]:
        xs, ys = [], []
        for ell, r in R.items():
            g = d[(d.ell == ell) & (d.flux == flux) & (d.method == "pcn")]
            m = d[(d.ell == ell) & (d.flux == flux) & (d.method == "msm") & (d.blocks == nb)]
            if len(g) and len(m):
                xs.append(r); ys.append(g.conv_it.iloc[0] / m.conv_it.iloc[0])
        a.plot(xs, ys, marker=mk, label=f"{nb}x{nb} blocks")
    a.axhline(1, color="k", lw=.8, ls="--"); a.set_xscale("log"); a.set_title(t)
    a.set_xlabel("global KLE dimension r  (ell 0.2, 0.1, 0.05)")
ax[0].set_ylabel("iterations to MPSRF<1.2: global / MSM basis"); ax[0].legend()
plt.tight_layout(); plt.savefig(os.path.join(D, "msm_speedup_vs_dimension.png"), dpi=160)
