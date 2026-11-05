"""
Post-processing of one configuration of the grid-refinement study, from <dir>/chains.npz and
<dir>/summary.json (experiments/run_grid_chains.py).  Writes <dir>/post/:
    ar_hist.png, relerr_traces.png, mpsrf_theta.png, fields_run.png, theta_stats.npz,
    post_summary.json, fields.npz
1. acceptance histogram (500-iteration blocks after M);  2. relative-error traces;
3. MPSRF / PSRF of theta (20 coordinates) in the group's convention (window from --mpsrf-start, default 1000,
   grows to the end of the run; see mpsrf_method in post_summary.json);
4. posterior of theta (mean, sd, covariance of the pooled post-M samples), and posterior fields of log kappa
   and pressure from --n-fields post-M samples (mean, sd).

    python experiments/run_grid_post.py --dir runs/grid_arms_2026-11/arm3_abs_n64 --n-fields 10000
"""
import _common  # noqa: F401
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brho import gridstudy as G  # noqa: E402
import msm_fieldplots as FP  # noqa: E402
from run_msm_post import ar_blocks, mpsrf_of, mpsrf_method  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n-fields", type=int, default=10000)
    ap.add_argument("--mpsrf-start", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--cmap-std", default=None)
    a = ap.parse_args(argv)
    summ = json.load(open(os.path.join(a.dir, "summary.json")))
    st = summ["settings"]
    arm, n, N, M, se = st["arm"], st["n"], st["n_iter"], st["adapt"], st["save_every"]
    Z = np.load(os.path.join(a.dir, "chains.npz"))
    fdir = os.path.join(a.dir, "post"); os.makedirs(fdir, exist_ok=True)
    S = G.GridSetup(n)
    data = G.GridData(S, arm, st["eps"], st["sigma_abs"], st["data_seed"])
    theta = Z["theta"].astype(float)                       # (m, N/se, 20)
    m = theta.shape[0]
    res = dict(arm=arm, n=n, n_obs=data.n_obs, chains=int(m))
    # 1. acceptance
    ab = ar_blocks(Z["acc"], Z["move"], M, N)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(ab.ravel(), bins=30, color="C0"); ax.axvline(0.25, color="k", ls=":")
    ax.set_title(f"AR (500-it blocks), {arm} n={n}"); ax.set_xlabel("acceptance rate")
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "ar_hist.png"), dpi=130); plt.close(fig)
    res["acceptance"] = dict(mean=float(ab.mean()), sd=float(ab.std()), min=float(ab.min()), max=float(ab.max()))
    # 2. relative-error traces
    rel = Z["relerr"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(rel.shape[1])
    ax.semilogy(x, np.median(rel, 0), label="median over chains")
    ax.fill_between(x, np.percentile(rel, 5, 0), np.percentile(rel, 95, 0), alpha=.3, label="5-95%")
    ax.axvline(M, color="k", ls=":"); ax.axhline(data.relerr_truth, color="C3", ls="--", label="noise level")
    ax.set_xlabel("iteration"); ax.set_ylabel("relative error"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(fdir, "relerr_traces.png"), dpi=130); plt.close(fig)
    res["relerr_post_M_median"] = float(np.median(rel[:, M:]))
    res["relerr_at_truth"] = float(data.relerr_truth)
    res["phi_post_M_median"] = float(np.median(Z["phi"][:, M:]))
    res["phi_at_truth"] = float(data.phi_truth)
    # 3. MPSRF of theta
    mp = {}
    j0 = a.mpsrf_start // se
    mpsrf_of(theta[:, j0:], j0 * se, se, "theta", fdir, mp)
    res["mpsrf"] = mp
    res["mpsrf_method"] = mpsrf_method(a.mpsrf_start, se).replace(
        "Dimension: theta (64), or the 58 range coordinates V_+^T theta for runs whose theta lives in the rank-58 range; "
        "local coefficients as stated per entry.", "Dimension: theta (20 KLE coefficients).")
    print(res["mpsrf_method"])
    # 4. posterior of theta and fields
    post = theta[:, M // se:].reshape(-1, S.r)
    np.savez_compressed(os.path.join(fdir, "theta_stats.npz"), mean=post.mean(0), sd=post.std(0),
                        cov=np.cov(post.T), theta_true=data.theta_true, n_samples=len(post),
                        chain_means=theta[:, M // se:].mean(1))
    res["theta_post"] = dict(sd_mean=float(post.std(0).mean()), sd_min=float(post.std(0).min()),
                             sd_max=float(post.std(0).max()),
                             rms_mean_error=float(np.sqrt(np.mean((post.mean(0) - data.theta_true) ** 2))))
    rng = np.random.default_rng(a.seed)
    sel = post[rng.choice(len(post), min(a.n_fields, len(post)), replace=False)]
    lk = S.kle.field(sel)
    pr = np.empty_like(lk)
    for j in range(len(sel)):
        pr[j] = S.phys.solve(np.exp(lk[j]))
    lk_true = S.kle.field(data.theta_true)
    np.savez_compressed(os.path.join(fdir, "fields.npz"), grid_n=n, logk_true=lk_true, p_true=data.p_true,
                        phases=np.array(["run"]), run_logk_mean=lk.mean(0), run_logk_sd=lk.std(0),
                        run_p_mean=pr.mean(0), run_p_sd=pr.std(0), run_n=len(sel))
    F = dict(np.load(os.path.join(fdir, "fields.npz")))
    FP.plot_fields(F, "run", n, os.path.join(fdir, "fields_run.png"), a.cmap, a.cmap_std)
    res["fields"] = dict(run=dict(n=len(sel), logk_sd_mean=float(lk.std(0).mean()), p_sd_mean=float(pr.std(0).mean()),
                                  logk_mean_error_rms=float(np.sqrt(np.mean((lk.mean(0) - lk_true) ** 2)))))
    json.dump(res, open(os.path.join(fdir, "post_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "mpsrf_method"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
