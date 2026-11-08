"""GLAM on a synthetic high-dimensional Gaussian posterior (no solver, no cluster): the real sampler and anchor manager,
24 chains (ThreadWorld threads), state = 12 blocks x 500 + 1 (d = 6001), Phi = |y - A x|^2 / (2 s^2), prior N(0, I).
The exact posterior is known, so the run is scored against it.

    python experiments/run_glam_gaussian.py --n-iter 20000 --tag g1
    python experiments/run_glam_gaussian.py --n-iter 600 --tag smoke        # timing check

Output: runs/gaussian/<tag>/ (chain files, exchanges.pkl) and <tag>_summary.json next to it.
"""
import _common  # noqa: F401
import argparse
import json
import os
import pickle
import time

import numpy as np

from brho import config as C
from brho.glam.comm import ThreadWorld
from brho.glam.post import load_chain
from brho.glam.sampler import run_glam
from brho.glam.toy import LinearGaussianModel


def expand(ch, n_iter):
    """per-iteration index of the record that holds the state after the move at that iteration"""
    idx = np.full(n_iter, -1)
    for j, (s, c) in enumerate(zip(ch["start"], ch["count"])):
        idx[s:s + c] = j
    return idx


def make_model(n_sub=12, m=500, n_data=300, phi0=7000.0, seed=0):
    # prior draw: Phi0 ~ n (2 + s^2) / (2 s^2)  ->  s^2 = 2 n / (2 phi0 - n)
    s = float(np.sqrt(2.0 * n_data / (2.0 * phi0 - n_data)))
    return LinearGaussianModel(n_sub=n_sub, m=m, n_data=n_data, s=s, seed=seed,
                               x_true=np.random.default_rng(seed + 1).standard_normal(n_sub * m + 1))


def exact_posterior(model):
    """mean, informed basis V (d x r), posterior variances along V (1/(1+S^2/s^2)); null directions have variance 1."""
    A, s = model.A, model.s
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    var = 1.0 / (1.0 + S ** 2 / s ** 2)
    prec_rhs = A.T @ model.y / s ** 2
    mean = V @ (var * (V.T @ prec_rhs))            # null component of the mean is 0
    r = model.y - A @ mean
    e_phi = (r @ r + np.sum(var * S ** 2)) / (2 * s ** 2)
    return mean, V, var, e_phi


def config(n_iter, window1):
    return C.GlamConfig(
        n_iter=n_iter, window1=window1, window2=n_iter - window1, burnin_length=500, global_stage_length=200,
        local_stage_length=200, min_global_stage_length=50, min_local_stage_length=50,
        beta_g0=0.05, beta_l0=0.1, target=0.25, adapt_interval=50, adapt_gain=2.0, adapt_gain_exponent=0.5, adapt_gain_hold=40,
        local_block_schedule="round_robin", save_every=1000, log_every=1000,
        # proposed r24m3 archive
        anchor_include_previous=True, anchor_use_best_state=True, max_anchors=100,
        restart_scale=0.05, exploit_restart_scale=0.02, frontier_restart_scale=0.05, exploration_restart_scale=0.06,
        restart_phi_tolerance=100.0, quality_tolerance=100.0, minimum_anchor_distance=12.0,
        covariance_regularization=0.03, logdet_gain_scale=1.0, minimum_diversity_score=0.5, diversity_alpha=0.85,
        archive_selection_temperature=0.3, exploit_fraction=0.25, frontier_fraction=0.5, explore_fraction=0.25,
        frontier_target_phi=-2.0,  # replaced below once E_post[Phi] is known
        successful_region_distance=20.0, successful_region_merge_distance=15.0, frontier_minimum_distance=12.0,
        frontier_relative_quality_tolerance=300.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=20000)
    ap.add_argument("--window1", type=int, default=0, help="default n_iter/4")
    ap.add_argument("--chains", type=int, default=24)
    ap.add_argument("--n-data", type=int, default=300)
    ap.add_argument("--phi0", type=float, default=7000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="g1")
    ap.add_argument("--out-root", default=os.path.join(_common.REPO if hasattr(_common, "REPO") else ".", "runs", "gaussian"))
    ap.add_argument("--preset", default="", choices=["", "stages", "r24m3b"],
                    help="stages: g1 settings with 1000-iteration stages; r24m3b: stages + eps 1e-2, alpha 0.7, gain 8, cap 300, "
                         "min score 0.5, 1000-iteration stages, matched explore restarts")
    ap.add_argument("--score-only", action="store_true", help="skip sampling, score an existing run")
    ap.add_argument("--set", nargs="*", default=[], help="GlamConfig overrides name=value")
    args = ap.parse_args()
    model = make_model(n_data=args.n_data, phi0=args.phi0, seed=args.seed)
    mean, V, var, e_phi = exact_posterior(model)
    cfg = config(args.n_iter, args.window1 or args.n_iter // 4)
    d = dict(vars(cfg))
    d["frontier_target_phi"] = 1.3 * e_phi
    if args.preset in ("stages", "r24m3b"):
        d.update(global_stage_length=1000, local_stage_length=1000, min_global_stage_length=500, min_local_stage_length=500)
    if args.preset == "r24m3b":
        d.update(covariance_regularization=0.01, diversity_alpha=0.7, logdet_gain_scale=8.0, max_anchors=300,
                 minimum_diversity_score=0.5)
    for kv in args.set:
        k, v = kv.split("=")
        d[k] = type(d[k])(v) if not isinstance(d[k], bool) else v.lower() == "true"
    cfg = C.GlamConfig(**d)
    out = os.path.join(args.out_root, args.tag)
    os.makedirs(out, exist_ok=True)
    print(f"d={model.d} s={model.s:.4f} E_post[Phi]={e_phi:.1f} frontier_target={cfg.frontier_target_phi:.1f}", flush=True)
    t0 = time.time()
    P = args.chains
    if not args.score_only:
      ThreadWorld(P).run(lambda comm: run_glam(model, cfg, comm, out, log=(print if comm.Get_rank() == 0 else (lambda *_: None))))
    wall = time.time() - t0
    # ---- score against the exact posterior
    tail = cfg.window1
    Zs, phis, sx, sq, n = [], [], np.zeros(model.d), 0.0, 0
    for r in range(P):                      # one chain at a time (the full tail does not fit in memory)
        ch = load_chain(out, r)
        idx = expand(ch, cfg.n_iter)[tail::10]
        Xr = np.hstack([ch["theta"][idx], ch["w"][idx][:, None]])
        Zs.append(Xr @ V); phis.append(ch["phi"][idx])
        sx += Xr.sum(0); sq += float((Xr ** 2).sum()); n += len(Xr)
        del ch, Xr
    Z = np.vstack(Zs); phis = np.concatenate(phis)
    mu_s = sx / n
    var_inf = Z.var(0)
    total_var = sq / n - float(mu_s @ mu_s)
    var_null = float((total_var - var_inf.sum()) / (model.d - V.shape[1]))
    X = np.empty((n, 0))
    err = mu_s - mean
    err_inf = V.T @ err
    summary = dict(wall_s=wall, d=model.d, s=model.s, e_post_phi=e_phi, phi_tail_mean=float(phis.mean()),
                   phi_tail_median=float(np.median(phis)),
                   mean_err_total=float(np.linalg.norm(err)), mean_err_informed=float(np.linalg.norm(err_inf)),
                   post_mean_norm=float(np.linalg.norm(mean)),
                   var_ratio_informed_median=float(np.median(var_inf / var)), var_ratio_informed_min=float((var_inf / var).min()),
                   var_ratio_informed_max=float((var_inf / var).max()), var_ratio_null=var_null,
                   n_tail_samples=int(len(X)))
    with open(os.path.join(out, "exchanges.pkl"), "rb") as f:
        ex = pickle.load(f)
    rows = []
    for e in ex:
        a = e.get("archive", {})
        rows.append(dict(iteration=e["iteration"], phase=e["phase"],
                         **{k: a.get(k) for k in ("anchor_count", "anchor_effective_rank", "anchor_cov_largest_eigenvalue_fraction",
                                                  "distance_min", "distance_median", "distance_max", "candidate_count_quality_eligible",
                                                  "rejected_below_score_count", "selected_score_min", "selected_logdet_novelty_mean",
                                                  "selected_distance_novelty_mean", "archive_hit_cap", "frontier_status",
                                                  "successful_region_count")},
                         **{k: e.get("restart", {}).get(k) for k in ("exploit_restart_success_rate", "explore_restart_success_rate",
                                                                     "explore_restart_attempts_mean", "frontier_restart_success_rate")}))
    summary["exchanges"] = rows[:: max(1, len(rows) // 40)]
    summary["preset"] = args.preset
    summary["n_exchanges"] = len(rows)
    with open(os.path.join(args.out_root, f"{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "exchanges"}, indent=1, default=float))


if __name__ == "__main__":
    main()
