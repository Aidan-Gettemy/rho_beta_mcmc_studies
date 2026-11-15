#!/usr/bin/env python3
"""dd_end summary figure: box plots of the relative data error of the posterior fields.

    python experiments/dd_end_plots.py [--root DIR] [--truth 2026] [--eps-factor 1]

One figure per correlation length (relerr_ell<ell>.png), two rows: pressure relative error (every cell) and flux relative error
(flux cells only).  Each box is the distribution over the saved posterior fields of ||d - model|| / ||d|| (pressure at the sensors;
flux at the measured interface faces); the dashed line is the noise level of the data (error of the truth against the data).
Reads post/fields.npz of every run in the manifest (DD+ cells: the dual grid and the pooled estimate)."""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "runs/msm_local_kle_2026-11/dd_end"


def _variants(method):
    if method in ("pcn", "msm"):
        return [("", "post", "run")]
    if method == "ddplus":
        return [("(dual)", "post", "dual"), ("(pooled)", "post_pooled", "pooled")]
    return [("", "post", "primal")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--truth", default="2026")
    ap.add_argument("--eps-factor", default="1")
    a = ap.parse_args()
    man = [m for m in csv.DictReader(open(os.path.join(a.root, "manifest.csv")))
           if m["truth"] == a.truth and float(m["eps_factor"]) == float(a.eps_factor)]
    for ell in sorted({m["ell"] for m in man}, key=float):
        items = {"p": [], "q": []}
        noise = {}
        for m in (x for x in man if x["ell"] == ell):
            for tag, sub, prefix in _variants(m["method"]):
                fn = os.path.join(m["dir"], sub, "fields.npz")
                if not os.path.exists(fn):
                    continue
                z = np.load(fn)
                if f"{prefix}_relerr_p" not in z.files:
                    continue
                items["p"].append((m["label"] + tag, z[f"{prefix}_relerr_p"]))
                noise["p"] = float(z["data_noise_relerr_p"])
                if f"{prefix}_relerr_q" in z.files:
                    items["q"].append((m["label"] + tag, z[f"{prefix}_relerr_q"]))
                    noise["q"] = float(z["data_noise_relerr_q"])
        if not items["p"]:
            continue
        rows = [k for k in ("p", "q") if items[k]]
        fig, axs = plt.subplots(len(rows), 1, figsize=(max(7, 0.55 * len(items["p"])), 3.6 * len(rows)), squeeze=False)
        for ax, k in zip(axs[:, 0], rows):
            labs, vals = zip(*items[k])
            ax.boxplot(vals, tick_labels=labs, showfliers=False)
            ax.axhline(noise[k], color="k", ls="--", lw=1, label="noise level of the data")
            ax.set_ylabel("relative error, " + ("pressure" if k == "p" else "flux")); ax.legend(loc="upper left")
            ax.tick_params(axis="x", rotation=70)
        axs[0, 0].set_title(f"posterior-field data misfit, ell = {ell}, truth {a.truth}, eps factor {a.eps_factor}")
        fig.tight_layout()
        out = os.path.join(a.root, f"relerr_ell{ell}" + ("" if a.truth == "2026" and float(a.eps_factor) == 1 else f"_t{a.truth}_e{a.eps_factor}") + ".png")
        fig.savefig(out, dpi=130); plt.close(fig); print(out)


if __name__ == "__main__":
    main()
