"""
Post-processing of one experiment of the paper-MSM study, from <dir>/chains.npz and <dir>/summary.json
(experiments/run_paper_chains.py).  Writes <dir>/post/:
    ar_hist.png, relerr_traces.png, mpsrf_native.png, mpsrf_common.png, fields_run.png, theta_stats.npz (common
    coordinates), fields.npz, post_summary.json
1. acceptance (500-iteration blocks after M) and per-group acceptance;  2. relative-error traces;
3. MPSRF / PSRF in the group's convention (window from --mpsrf-start, default 1000, grows to the end of the run) on
   the NATIVE state (20 coordinates: global theta for e0/e1, local coefficients c otherwise) and on the COMMON
   coordinates z = theta-coordinates (L2 projection onto the 20 global modes) of the field;
4. posterior of z and of the fields (log kappa, pressure) from the thinned post-M fields, and their comparison with
   the control (arm 1 of the grid study at 16 x 16): Gaussian-approximation KL of z, rms difference of the posterior
   mean fields, ratio of the posterior sd fields.

    python experiments/run_paper_post.py --dir runs/paper_msm_2026-11/e7_smooth_sub
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

from brho import msmpaper as M  # noqa: E402
import msm_fieldplots as FP  # noqa: E402
from run_msm_post import ar_blocks, mpsrf_of, mpsrf_method  # noqa: E402


def gauss_kl(m0, S0, m1, S1):
    d = len(m0)
    S1i = np.linalg.inv(S1)
    dm = m1 - m0
    return 0.5 * (np.trace(S1i @ S0) + dm @ S1i @ dm - d + np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--control", default="runs/grid_arms_2026-11/arm1_check_n16/post")
    ap.add_argument("--n-fields", type=int, default=10000)
    ap.add_argument("--mpsrf-start", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--cmap-std", default=None)
    a = ap.parse_args(argv)
    summ = json.load(open(os.path.join(a.dir, "summary.json")))
    st = summ["settings"]
    exp, N, Mad, se, fe = st["exp"], st["n_iter"], st["adapt"], st["save_every"], st["field_every"]
    Z = np.load(os.path.join(a.dir, "chains.npz"))
    fdir = os.path.join(a.dir, "post"); os.makedirs(fdir, exist_ok=True)
    S = M.PaperSetup()
    data = M.PaperData(S, st["eps"], st["data_seed"])
    c, z = Z["c"].astype(float), Z["z"].astype(float)
    m = c.shape[0]
    res = dict(exp=exp, spec=summ["spec"], n_obs=data.n_obs, chains=int(m), n_iter=N, adapt=Mad)
    # 1. acceptance
    ab = ar_blocks(Z["acc"], Z["move"], Mad, N)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(ab.ravel(), bins=30, color="C0"); ax.axvline(0.25, color="k", ls=":")
    ax.set_title(f"AR (500-it blocks), {exp}"); ax.set_xlabel("acceptance rate")
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "ar_hist.png"), dpi=130); plt.close(fig)
    res["acceptance"] = dict(mean=float(ab.mean()), sd=float(ab.std()), min=float(ab.min()), max=float(ab.max()),
                             group_min=summ["post_acc_group_min"], group_max=summ["post_acc_group_max"])
    res["step_final_median_per_group"] = summ["s_final_median_per_group"]
    # 2. relative-error traces
    rel = Z["relerr"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(rel.shape[1])
    ax.semilogy(x, np.median(rel, 0), label="median over chains")
    ax.fill_between(x, np.percentile(rel, 5, 0), np.percentile(rel, 95, 0), alpha=.3, label="5-95%")
    ax.axvline(Mad, color="k", ls=":"); ax.axhline(data.relerr_truth, color="C3", ls="--", label="noise level")
    ax.set_xlabel("iteration"); ax.set_ylabel("relative error"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(fdir, "relerr_traces.png"), dpi=130); plt.close(fig)
    res["relerr_post_M_median"] = float(np.median(rel[:, Mad:]))
    res["relerr_at_truth"] = float(data.relerr_truth)
    res["phi_post_M_median"] = float(np.median(Z["phi"][:, Mad:]))
    # 3. MPSRF, native and common coordinates
    mp, j0 = {}, a.mpsrf_start // se
    mpsrf_of(c[:, j0:], j0 * se, se, "native", fdir, mp)
    mpsrf_of(z[:, j0:], j0 * se, se, "common", fdir, mp)
    res["mpsrf"] = mp
    res["mpsrf_method"] = mpsrf_method(a.mpsrf_start, se).replace(
        "Dimension: theta (64), or the 58 range coordinates V_+^T theta for runs whose theta lives in the rank-58 range; "
        "local coefficients as stated per entry.",
        "Dimension 20: 'native' = the sampler's state (global theta for e0/e1, local coefficients c for e2-e7), "
        "'common' = theta-coordinates (L2 projection onto the 20 global modes) of the field.")
    print(res["mpsrf_method"])
    # 4. posterior of the common coordinates and of the fields
    zp = z[:, Mad // se:]
    pool = zp.reshape(-1, S.r)
    mean, cov = pool.mean(0), np.cov(pool.T)
    np.savez_compressed(os.path.join(fdir, "theta_stats.npz"), mean=mean, sd=pool.std(0), cov=cov,
                        theta_true=data.theta_true, n_samples=len(pool), chain_means=zp.mean(1))
    res["z_post"] = dict(sd_mean=float(pool.std(0).mean()), sd_min=float(pool.std(0).min()), sd_max=float(pool.std(0).max()),
                         rms_mean_error=float(np.sqrt(np.mean((mean - data.theta_true) ** 2))))
    ET = Z["eta_thin"].astype(float)                         # (m, n_thin, N): field before iteration k * fe
    k0 = int(np.ceil(Mad / fe))
    eta = ET[:, k0:].reshape(-1, S.N)
    rng = np.random.default_rng(a.seed)
    eta = eta[rng.choice(len(eta), min(a.n_fields, len(eta)), replace=False)]
    pr = np.empty_like(eta)
    for j in range(len(eta)):
        pr[j] = S.phys.solve(np.exp(eta[j]))
    lk_true = S.T @ data.theta_true
    np.savez_compressed(os.path.join(fdir, "fields.npz"), grid_n=S.n, logk_true=lk_true, p_true=data.g.p_true,
                        phases=np.array(["run"]), run_logk_mean=eta.mean(0), run_logk_sd=eta.std(0),
                        run_p_mean=pr.mean(0), run_p_sd=pr.std(0), run_n=len(eta))
    F = dict(np.load(os.path.join(fdir, "fields.npz")))
    FP.plot_fields(F, "run", S.n, os.path.join(fdir, "fields_run.png"), a.cmap, a.cmap_std)
    res["fields"] = dict(n=len(eta), logk_sd_mean=float(eta.std(0).mean()), p_sd_mean=float(pr.std(0).mean()),
                         logk_mean_error_rms=float(np.sqrt(np.mean((eta.mean(0) - lk_true) ** 2))))
    # comparison with the control (arm 1, 16 x 16)
    cdir = a.control
    if os.path.exists(os.path.join(cdir, "fields.npz")):
        Fc = np.load(os.path.join(cdir, "fields.npz"))
        Tc = np.load(os.path.join(cdir, "theta_stats.npz"))
        res["vs_control"] = dict(
            kl_z_to_control=float(gauss_kl(mean, cov, Tc["mean"], Tc["cov"])),
            kl_control_to_z=float(gauss_kl(Tc["mean"], Tc["cov"], mean, cov)),
            logk_mean_rms_diff=float(np.sqrt(np.mean((eta.mean(0) - Fc["run_logk_mean"]) ** 2))),
            logk_sd_ratio_mean=float(eta.std(0).mean() / Fc["run_logk_sd"].mean()),
            logk_sd_ratio_band=float(eta.std(0)[S.band].mean() / Fc["run_logk_sd"][S.band].mean()),
            logk_sd_ratio_interior=float(eta.std(0)[~S.band].mean() / Fc["run_logk_sd"][~S.band].mean()),
            p_mean_rms_diff=float(np.sqrt(np.mean((pr.mean(0) - Fc["run_p_mean"]) ** 2))),
            p_sd_ratio_mean=float(pr.std(0).mean() / Fc["run_p_sd"].mean()))
    json.dump(res, open(os.path.join(fdir, "post_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "mpsrf_method"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
