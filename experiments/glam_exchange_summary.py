"""Per-exchange summary of a GLAM run's archive and restarts (no MPI, no solver), from <results>/exchanges.pkl:

    python experiments/glam_exchange_summary.py --exp-tag r24m-pilot
    python experiments/glam_exchange_summary.py --exp-tag r24m --every 5

Prints one line per exchange (every --every-th) and writes tables/<study>/exchanges.csv with all of them:
  iteration, phase, cycle                 where the exchange happened
  anchors, eligible                       archive size after the rebuild; candidates that passed the quality tolerance
  phi_best, phi_med, phi_max              Phi of the anchors
  dist_med, eff_rank                      median distance between anchors; effective rank of their covariance
  arch_s                                  wall seconds of the archive rebuild (watch it grow with the archive)
  ok_exploit, ok_frontier, ok_explore     fraction of restarts accepted in the role
  sc_exploit, sc_frontier, sc_explore     mean radius finally used (shrinks by restart_scale_decay on a rejected attempt)
  best_role                               role that produced the best anchor
exchanges.pkl is written when a run finishes (not while it runs).
"""
import _common  # noqa: F401
import argparse
import os
import pickle

import numpy as np

from brho import io

COLUMNS = [("iteration", None), ("phase", None), ("cycle", None), ("anchors", "anchor_count"),
           ("eligible", "candidate_count_quality_eligible"), ("phi_best", "anchor_phi_best"),
           ("phi_med", "anchor_phi_median"), ("phi_max", "anchor_phi_max"), ("dist_med", "distance_median"),
           ("eff_rank", "anchor_effective_rank"), ("arch_s", "archive_exchange_wall_seconds_local")]
RESTART = [("ok_exploit", "exploit_restart_success_rate"), ("ok_frontier", "frontier_restart_success_rate"),
           ("ok_explore", "explore_restart_success_rate"), ("sc_exploit", "exploit_restart_final_scale_mean"),
           ("sc_frontier", "frontier_restart_final_scale_mean"), ("sc_explore", "explore_restart_final_scale_mean")]


def rows(exchanges):
    out = []
    for e in exchanges:
        a, r = e.get("archive", {}), e.get("restart", {})
        row = dict(iteration=e["iteration"], phase=e["phase"], cycle=e["cycle"])
        for name, key in COLUMNS[3:]:
            row[name] = float(a.get(key, np.nan))
        for name, key in RESTART:
            row[name] = float(r.get(key, np.nan))
        row["best_role"] = str(a.get("best_source_role", ""))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--every", type=int, default=1, help="print every k-th exchange")
    args = ap.parse_args()
    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")
    path = os.path.join(io.results_dir(exp), "exchanges.pkl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found (written when the run finishes)")
    with open(path, "rb") as f:
        R = rows(pickle.load(f))
    names = [c[0] for c in COLUMNS] + [c[0] for c in RESTART] + ["best_role"]
    odir = io.tables_dir(exp)
    os.makedirs(odir, exist_ok=True)
    with open(os.path.join(odir, "exchanges.csv"), "w") as f:
        f.write(",".join(names) + "\n")
        for r in R:
            f.write(",".join(f"{r[n]:.6g}" if isinstance(r[n], float) else str(r[n]) for n in names) + "\n")
    print(" ".join(f"{n:>10}" for n in names))
    for i, r in enumerate(R):
        if i % max(args.every, 1) == 0 or i == len(R) - 1:
            print(" ".join(f"{r[n]:>10.4g}" if isinstance(r[n], float) else f"{str(r[n]):>10}" for n in names))
    a = np.array([r["anchors"] for r in R]); t = np.array([r["arch_s"] for r in R])
    print(f"\n{len(R)} exchanges; anchors max {np.nanmax(a):.0f}; archive rebuild: mean {np.nanmean(t):.2f} s, max {np.nanmax(t):.2f} s,"
          f" total {np.nansum(t) / 60:.1f} min (per rank)")


if __name__ == "__main__":
    main()
