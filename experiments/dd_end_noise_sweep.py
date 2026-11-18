#!/usr/bin/env python3
"""Stage 7: DD to global ratios against the noise level (eps x 0.5, 1, 2) at ell 0.1, 4 x 4, truth 2026, both data types.
Writes dd_end/noise_sweep.csv and noise_sweep.png.  The eps x 1 band is the spread over the three truths (stage 6)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "msm_local_kle_2026-11", "dd_end")
d = pd.read_csv(os.path.join(D, "master_table.csv"))
x = d[(d.ell == 0.1) & (d.truth == 2026) & d.variant.isin(["primal", "dual", "pooled"]) & ((d.blocks == 4) | d.blocks.isna()) & ~d.label.str.contains("DD-H")]
x = x.drop_duplicates(["label", "variant", "eps_factor"]).sort_values(["flux", "variant", "eps_factor"])
x[["flux", "variant", "eps_factor", "sd_mean", "err_rms", "sd_ratio", "err_ratio", "cover90", "sup_mpsrf"]].round(3).to_csv(os.path.join(D, "noise_sweep.csv"), index=False)
t3 = pd.read_csv(os.path.join(D, "three_truth.csv")).set_index(["flux", "family"])
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for j, flux in enumerate([False, True]):
    for i, (col, name) in enumerate([("sd_ratio", "sd ratio to global"), ("err_ratio", "error ratio to global")]):
        a = ax[i, j]
        for var, mk in [("primal", "o"), ("dual", "s"), ("pooled", "^")]:
            g = x[(x.flux == flux) & (x.variant == var)]
            a.plot(g.eps_factor, g[col], marker=mk, label=var)
            r = t3.loc[(flux, var)]
            a.vlines(1.0, r[f"{col}_mean"] - r[f"{col}_range"] / 2, r[f"{col}_mean"] + r[f"{col}_range"] / 2, color="k", lw=3, alpha=.25)
        a.axhline(1, color="k", lw=.8, ls="--"); a.set_xscale("log"); a.set_xticks([.5, 1, 2]); a.set_xticklabels(["0.5", "1", "2"])
        a.set_title(("pressure + flux" if flux else "pressure only")); a.set_ylabel(name)
        if i == 1: a.set_xlabel("noise: eps / eps_c")
ax[0, 0].legend(); fig.text(.5, .005, "grey bars: spread over three truths at eps / eps_c = 1 (mean +- range/2)", ha="center", fontsize=8)
plt.tight_layout(rect=(0, .02, 1, 1)); plt.savefig(os.path.join(D, "noise_sweep.png"), dpi=160)
print(x[["flux", "variant", "eps_factor", "sd_ratio", "err_ratio"]].round(2).to_string(index=False))
