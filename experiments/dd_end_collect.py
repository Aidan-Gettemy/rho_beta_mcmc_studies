#!/usr/bin/env python3
"""Collect the dd_end results into one master table (committed summaries only: post/post_summary.json, post/fields.npz, summary.json).

    python experiments/dd_end_collect.py                      # reads runs/msm_local_kle_2026-11/dd_end/manifest.csv
    python experiments/dd_end_collect.py --root DIR --out master_table.csv

One row per run (a DD+ cell gives two rows: the dual grid alone, and the pooled primal + dual estimate).  Every metric is computed from the
stored posterior mean / sd fields of log k and the truth, so global, MSM-basis and DD runs are measured the same way:
  sd_mean       mean over the grid of the posterior sd of log k
  err_rms       RMS over the grid of (posterior mean - truth)
  err_over_sd   err_rms / sd_mean
  cover90       fraction of cells whose truth lies within mean +- 1.645 sd (pointwise, Gaussian marginals; nominal 0.90)
  sup_mpsrf     sup over blocks of the MPSRF at the end of the run (window from iteration 0 on log-spaced points; formula in the post-processing)
  conv_it, stays_below_it   first iteration with sup-MPSRF < 1.2, and the iteration from which it stays below 1.2 (a growing window from
                iteration 0 includes the transient, so these are upper bounds on the time to stationarity); burn, burn_ok: the burn-in
                used for the fields and whether it is at or beyond stays_below_it
  sd_ratio, err_ratio   to the global pCN reference of the same ell, data type, truth and eps factor
  wall_sec, sec_per_chain   from the sampler's summary.json (a DD run's tasks are blocks x chains)."""
import argparse
import csv
import json
import os

import numpy as np

ROOT = "runs/msm_local_kle_2026-11/dd_end"


def _field_metrics(npz, prefix):
    z = np.load(npz)
    truth, mean, sd = z["logk_true"], z[f"{prefix}_logk_mean"], z[f"{prefix}_logk_sd"]
    err = mean - truth
    e, s = float(np.sqrt(np.mean(err ** 2))), float(np.mean(sd))
    return dict(sd_mean=s, err_rms=e, err_over_sd=e / s, cover90=float(np.mean(np.abs(err) <= 1.645 * sd)),
                n_fields=int(z[f"{prefix}_n"]))


def _mpsrf(post_summary, method, variant):
    """(sup-MPSRF at the end, max PSRF at the end, first iteration with MPSRF < 1.2, iteration from which it stays below 1.2)."""
    d = json.load(open(post_summary))
    if method in ("pcn", "msm"):
        mm = d["mpsrf"]
        if "theta" in mm:                                  # global pCN: the theta MPSRF
            m = mm["theta"]
            return (float(m["mpsrf_final"]), float(m["psrf_max_final"]), m.get("convergence_iteration"), m.get("stays_below_from_iteration"))
        ms = list(mm.values())                             # MSM basis: theta_range58 and c_local68; report the worse
        def worst_m(key):
            x = [v.get(key) for v in ms]
            return None if any(t is None for t in x) else max(x)
        return (float(max(v["mpsrf_final"] for v in ms)), float(max(v["psrf_max_final"] for v in ms)),
                worst_m("convergence_iteration"), worst_m("stays_below_from_iteration"))
    g = d["grids"]
    vals = [(v["mpsrf"]["sup"]["mpsrf_final"], v["mpsrf"]["sup"]["psrf_max_final"], v["mpsrf"]["sup"].get("convergence_iteration"),
             v["mpsrf"]["sup"].get("stays_below_from_iteration")) for v in g.values()]
    def worst(i):                      # the latest crossing over the grids; None if a grid never crossed
        x = [v[i] for v in vals]
        return None if any(t is None for t in x) else max(x)
    return float(max(x[0] for x in vals)), float(max(x[1] for x in vals)), worst(2), worst(3)


def collect_run(d, method):
    """Rows (list of dicts) of one run folder `d`."""
    rows = []
    summ = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else {}
    st = summ.get("settings", {})
    timing = dict(wall_sec=summ.get("wall_sec"), sec_per_chain=summ.get("sec_per_chain"), n_iter=st.get("n_iter"), chains=st.get("chains"))
    if method in ("pcn", "msm"):
        variants = [("global" if method == "pcn" else "msm", "post", "run")]
    elif method == "ddplus":
        variants = [("dual", "post", "dual"), ("pooled", "post_pooled", "pooled")]
    else:
        variants = [("primal", "post", "primal")]
    for name, sub, prefix in variants:
        ps, fn = os.path.join(d, sub, "post_summary.json"), os.path.join(d, sub, "fields.npz")
        if not (os.path.exists(ps) and os.path.exists(fn)):
            rows.append(dict(variant=name, status="missing"))
            continue
        r = dict(variant=name, status="done", **_field_metrics(fn, prefix), **timing)
        r["sup_mpsrf"], r["psrf_max"], r["conv_it"], r["stays_below_it"] = _mpsrf(ps, method, name)
        burn = json.load(open(ps)).get("burn_used", st.get("burn"))      # the burn the post actually used (a re-post may override the run's own)
        r["burn"] = burn
        r["burn_ok"] = (r["stays_below_it"] is not None and burn is not None and burn >= r["stays_below_it"])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    man = list(csv.DictReader(open(os.path.join(a.root, "manifest.csv"))))
    table = []
    for m in man:
        for r in collect_run(m["dir"], m["method"]):
            table.append(dict(label=m["label"], ell=m["ell"], flux=m["flux"], method=m["method"], blocks=m["blocks"], truth=m["truth"],
                              eps_factor=m["eps_factor"], dir=m["dir"], **r))
    ref = {(t["ell"], t["flux"], t["truth"], t["eps_factor"]): t for t in table if t["method"] == "pcn" and t["status"] == "done"}
    for t in table:
        g = ref.get((t["ell"], t["flux"], t["truth"], t["eps_factor"]))
        if t["status"] == "done" and g:
            t["sd_ratio"], t["err_ratio"] = t["sd_mean"] / g["sd_mean"], t["err_rms"] / g["err_rms"]
    cols = ["label", "variant", "ell", "flux", "method", "blocks", "truth", "eps_factor", "status", "sd_mean", "err_rms", "err_over_sd", "cover90",
            "sup_mpsrf", "psrf_max", "conv_it", "stays_below_it", "burn", "burn_ok", "sd_ratio", "err_ratio", "n_iter", "chains", "wall_sec", "sec_per_chain", "n_fields", "dir"]
    out = a.out or os.path.join(a.root, "master_table.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    done = sum(t["status"] == "done" for t in table)
    bad = [t["label"] for t in table if t["status"] == "done" and t["sup_mpsrf"] > 1.2]
    print(f"{done} of {len(table)} rows done; sup-MPSRF above 1.2: {sorted(set(bad)) or 'none'}; table: {out}")


if __name__ == "__main__":
    main()
