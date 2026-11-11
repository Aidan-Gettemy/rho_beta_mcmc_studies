"""Stage 1 of the subspace-transplant study: five samplers on a synthetic d = 6001 Gaussian whose sensitivity decays with the
coefficient index (as the KLE ordering of the whitening basis is expected to), scored against the exact posterior.

    python experiments/transplant_synthetic.py --n-iter 20000 --out-root <dir>          # all arms, ~1 h in a 2-core sandbox
    python experiments/transplant_synthetic.py --arms pcn,single --n-iter 600 --out-root <dir>    # smoke test

State: 12 blocks x 500 coefficients + the whitened rho (the real layout), 24 chains (ThreadWorld threads), Phi = |y - A x|^2 / (2 s^2),
prior N(0, I).  A = G diag(c), c_j = 1 / (1 + (j / j0)^2) in the index j of the block, c = 1 for rho, so the first ~j0 coefficients of every
block are constrained and the rest are not.  Exchanges stop at --stop-iter (default n_iter / 2); after that every arm is plain pCN, and the
statistics are taken from the last quarter of the run, so they are not contaminated by restarts.

Arms:  pcn       plain pCN (burn-in never ends): also the data for the mask (Stage 0)
       single    GLAM, one anchor, stages 200/200, exploit restarts at scale 0.02
       multi     GLAM, open archive with the r24m3 settings (cap 100 here for memory), stages 1000/500
       single_t  single + transplant with the mask estimated from the pcn arm
       multi_t   multi + transplant + archive selection on the transplanted coordinates only
       single_o  single + transplant with the oracle mask (exact marginal posterior variance < 0.8 -> installed)
Scores (window = last quarter): Phi against E_post[Phi]; for the constrained and the unconstrained coordinates (oracle split) the variance
ratio sampled/exact (median) and the mean error; coordinate-wise PSRF median; archive size and effective rank at the last exchange.
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
from brho.glam.sampler import load_keep_mask, run_glam
from brho.glam.toy import LinearGaussianModel
from coef_variance_by_index import propose_mask, tail_moments

N_SUB, M, N_DATA = 12, 500, 300


def make_model(j0=30, phi0=7000.0, seed=0):
    d = N_SUB * M + 1
    model = LinearGaussianModel(n_sub=N_SUB, m=M, n_data=N_DATA, s=1.0, seed=seed)
    rng = np.random.default_rng(seed + 11)
    j = np.arange(M)
    c = np.concatenate([np.tile(1.0 / (1.0 + (j / j0) ** 2), N_SUB), [1.0]])
    G = rng.standard_normal((N_DATA, d))
    A = G * c[None, :]
    A /= np.sqrt(np.mean(np.sum(A ** 2, axis=1)))                 # unit mean squared row norm
    x_true = rng.standard_normal(d)
    # prior-draw Phi0 ~ n (2 rn2 + s^2) / (2 s^2) with rn2 = 1  ->  s^2 = 2 n / (2 phi0 - n)
    s = float(np.sqrt(2.0 * N_DATA / (2.0 * phi0 - N_DATA)))
    model.A, model.s = A, s
    model.y = A @ x_true + s * rng.standard_normal(N_DATA)
    return model


def exact(model):
    """mean, marginal variance (diag of the covariance), E_post[Phi], via the SVD of A."""
    A, s = model.A, model.s
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    shrink = 1.0 - 1.0 / (1.0 + S ** 2 / s ** 2)
    var_diag = 1.0 - (Vt ** 2).T @ shrink
    mean = Vt.T @ ((1.0 / (1.0 + S ** 2 / s ** 2)) * (Vt @ (A.T @ model.y / s ** 2)))
    r = model.y - A @ mean
    e_phi = (r @ r + np.sum(S ** 2 / (1.0 + S ** 2 / s ** 2))) / (2 * s ** 2)
    return mean, var_diag, float(e_phi)


def expand(ch, n_iter):
    idx = np.full(n_iter, -1)
    for j, (s0, c) in enumerate(zip(ch["start"], ch["count"])):
        idx[s0:s0 + c] = j
    return idx


def base_cfg(n_iter, stop, **kw):
    d = dict(n_iter=n_iter, window1=min(5000, n_iter // 4), window2=0, burnin_length=500, global_stage_length=200, local_stage_length=200,
             min_global_stage_length=50, min_local_stage_length=50, beta_g0=0.05, beta_l0=0.1, target=0.25, adapt_interval=50,
             adapt_gain=2.0, adapt_gain_exponent=0.5, adapt_gain_hold=40, local_block_schedule="round_robin", save_every=1000,
             log_every=0, anchor_stop_iteration=stop, random_reach_attempts=0, restart_phi_tolerance=100.0, max_restart_attempts=10)
    d.update(kw)
    return C.GlamConfig(**d)


def arm_cfg(arm, n_iter, stop, e_phi):
    window2 = n_iter                                          # local stages allowed for the whole run (they stop with the exchanges)
    if arm == "pcn":
        big = 10 ** 8
        return base_cfg(n_iter, -1, burnin_length=big, global_stage_length=big, local_stage_length=big,
                        min_global_stage_length=big, min_local_stage_length=big)
    if arm in ("single", "single_t", "single_o"):
        return base_cfg(n_iter, stop, window2=window2, max_anchors=1, restart_scale=0.02, exploit_restart_scale=0.02,
                        frontier_restart_scale=0.02, exploration_restart_scale=0.02, exploit_fraction=1.0, frontier_fraction=0.0,
                        explore_fraction=0.0, quality_tolerance=50.0, minimum_anchor_distance=0.0, anchor_include_previous=False,
                        anchor_use_best_state=False)
    if arm in ("multi", "multi_t"):
        return base_cfg(n_iter, stop, window2=window2, global_stage_length=1000, local_stage_length=500, min_global_stage_length=500,
                        min_local_stage_length=500, max_anchors=100, restart_scale=0.05, exploit_restart_scale=0.02,
                        frontier_restart_scale=0.05, exploration_restart_scale=0.06, restart_phi_tolerance=100.0,
                        quality_tolerance=100.0, minimum_anchor_distance=12.0, covariance_regularization=0.01, logdet_gain_scale=8.0,
                        minimum_diversity_score=0.5, diversity_alpha=0.7, archive_selection_temperature=0.3, exploit_fraction=0.10,
                        frontier_fraction=0.35, explore_fraction=0.55, frontier_target_phi=1.3 * e_phi,
                        successful_region_distance=20.0, successful_region_merge_distance=15.0, frontier_minimum_distance=12.0,
                        frontier_relative_quality_tolerance=300.0)
    raise SystemExit(f"unknown arm {arm}")


def score(out, model, mean_ex, var_ex, e_phi, n_iter, window, constrained):
    P = 24
    s1 = np.zeros(model.d); s2 = np.zeros(model.d); n = 0; phis = []; cm = []
    for r in range(P):
        ch = load_chain(out, r)
        sel = expand(ch, n_iter)[window::10]
        X = np.hstack([ch["theta"][sel], ch["w"][sel][:, None]])
        s1 += X.sum(0); s2 += (X ** 2).sum(0); n += len(X); phis.append(ch["phi"][sel]); cm.append(X.mean(0))
        del ch, X
    mu = s1 / n; var = s2 / n - mu ** 2
    cm = np.asarray(cm); within = np.mean(s2 / n - 0 * mu) if False else None
    chain_var_between = cm.var(axis=0, ddof=1)
    phis = np.concatenate(phis)
    err = mu - mean_ex
    out_d = dict(phi_tail_mean=float(phis.mean()), e_post_phi=e_phi)
    for name, sel_ in (("constrained", constrained), ("unconstrained", ~constrained)):
        out_d[f"var_ratio_median_{name}"] = float(np.median(var[sel_] / var_ex[sel_]))
        out_d[f"mean_err_{name}"] = float(np.linalg.norm(err[sel_]))
        out_d[f"between_chain_var_{name}"] = float(np.median(chain_var_between[sel_]))
    out_d["n_constrained"] = int(constrained.sum())
    return out_d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-iter", type=int, default=20000)
    ap.add_argument("--stop-iter", type=int, default=0, help="exchanges stop here (default n_iter/2)")
    ap.add_argument("--arms", default="pcn,single,multi,single_t,multi_t,single_o")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--j0", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.8)
    args = ap.parse_args()
    stop = args.stop_iter or args.n_iter // 2
    window = (3 * args.n_iter) // 4
    os.makedirs(args.out_root, exist_ok=True)
    model = make_model(j0=args.j0, seed=args.seed)
    mean_ex, var_ex, e_phi = exact(model)
    constrained = var_ex < args.thr
    print(f"d={model.d} s={model.s:.4f} E_post[Phi]={e_phi:.1f}; constrained coordinates (exact marginal variance < {args.thr}): "
          f"{constrained.sum()} ({constrained[:M].sum()} in block 0)", flush=True)
    # oracle mask: keep own where the exact marginal posterior variance is ~ the prior one
    omasks = [~constrained[b * M:(b + 1) * M] for b in range(N_SUB)] + [np.zeros(1, bool)]
    np.savez(os.path.join(args.out_root, "mask_oracle.npz"), **{f"m{b}": mk for b, mk in enumerate(omasks)})
    summary = {}
    sfile = os.path.join(args.out_root, "summary.json")
    if os.path.exists(sfile):
        summary = json.load(open(sfile))
    for arm in [a for a in args.arms.split(",") if a]:
        out = os.path.join(args.out_root, arm)
        cfg = arm_cfg(arm, args.n_iter, stop, e_phi)
        keep, sel_only = None, False
        if arm in ("single_t", "multi_t"):
            mean_s, var_s, between = tail_moments(os.path.join(args.out_root, "pcn"), 24, window, 10, N_SUB)
            masks, cuts = propose_mask(var_s, N_SUB, M, thr=args.thr)
            np.savez(os.path.join(args.out_root, "mask_est.npz"), **{f"m{b}": mk for b, mk in enumerate(masks)})
            keep = masks
            sel_only = arm == "multi_t"
            print(f"[{arm}] estimated cut-offs (first K installed): {cuts}", flush=True)
        if arm == "single_o":
            keep = omasks
        t0 = time.time()
        if os.path.isdir(out):
            import shutil; shutil.rmtree(out)
        ThreadWorld(24).run(lambda comm: run_glam(model, cfg, comm, out, log=lambda *_: None, keep_mask=keep,
                                                  selection_constrained_only=sel_only))
        wall = time.time() - t0
        res = score(out, model, mean_ex, var_ex, e_phi, args.n_iter, window, constrained)
        res["wall_s"] = wall
        try:
            ex = pickle.load(open(os.path.join(out, "exchanges.pkl"), "rb"))
            if ex:
                a = ex[-1]["archive"]
                res.update(anchors=a.get("anchor_count"), eff_rank=a.get("anchor_effective_rank"), exchanges=len(ex),
                           restart_success={k: ex[-1]["restart"].get(k) for k in ("exploit_success_rate", "explore_success_rate")})
        except Exception:
            pass
        if keep is not None:
            res["kept_fraction"] = float(np.mean(np.concatenate([m_.ravel() for m_ in keep])))
        summary[arm] = res
        json.dump(summary, open(sfile, "w"), indent=1, default=float)
        print(f"[{arm}] {wall:.0f}s " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
