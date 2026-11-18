#!/usr/bin/env python3
"""Stage 6: the ell 0.1, 4 x 4 column over the three truths (2026, 2027, 2028).  Per data type and family: sd ratio and error ratio to the
global pCN of the same truth, per truth, then the mean and the range (max - min) over the three.  A difference between two families
smaller than the larger of their ranges is not read as an effect.   Writes dd_end/three_truth.csv."""
import os
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "msm_local_kle_2026-11", "dd_end")
d = pd.read_csv(os.path.join(D, "master_table.csv"))
x = d[(d.ell == 0.1) & (d.blocks.isna() | (d.blocks == 4)) & d.sd_mean.notna() & d.truth.isin([2026, 2027, 2028])]
x = x[x.variant.isin(["global", "msm", "primal", "dual", "pooled"])].drop_duplicates(["label", "variant", "truth"])
rows = []
for (flux, var), g in x.groupby(["flux", "variant"]):
    r = dict(flux=bool(flux), family=var)
    for col, name in [("sd_ratio", "sd_ratio"), ("err_ratio", "err_ratio"), ("err_rms", "err_rms"), ("sd_mean", "sd")]:
        v = g.sort_values("truth")[col].to_numpy()
        for t, val in zip(sorted(g.truth), v):
            r[f"{name}_{int(t)}"] = round(float(val), 3)
        r[f"{name}_mean"] = round(float(v.mean()), 3); r[f"{name}_range"] = round(float(v.max() - v.min()), 3)
    rows.append(r)
out = pd.DataFrame(rows).sort_values(["flux", "family"])
out.to_csv(os.path.join(D, "three_truth.csv"), index=False)
print(out[["flux", "family", "sd_ratio_mean", "sd_ratio_range", "err_ratio_mean", "err_ratio_range"]].to_string(index=False))
