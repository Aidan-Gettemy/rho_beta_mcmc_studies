"""
Post-processing of the domain-decomposed (DD-MCMC, spline interface) study from the run folders written by
experiments/run_dd_chains.py (brho/msm_dd.py, docs/dd_mcmc_experiment.md).  One or two grids:

    test A (primal only):    --dirs runs/.../exp7_dd_primal            --out runs/.../exp7_dd_primal/post
    test B (primal + dual):  --dirs runs/.../exp7_dd_primal runs/.../exp7_dd_dual --out runs/.../exp7_dd_pooled/post

Writes to <out>/ : ar_hist.png, mpsrf_<grid>.png (+ mpsrf_sup_all.png for two grids), fields_<label>.png,
field_pairs.png, fields.npz (schema of experiments/run_msm_post.py, so experiments/msm_fieldplots.py plots it),
post_summary.json.

1. MPSRF per subdomain: the 64 chains of block k, local coefficients c_k (dimension r_k), group convention
   (experiments/run_msm_post.py: window from --mpsrf-start, skip 2, step 15 saved samples, first 10 curve points
   ignored).  The reported curve is the SUPREMUM over subdomains at every curve point; all subdomains are said to have
   converged when it first drops below 1.2 (linearly interpolated, as brho.mpsrf.convergence_iteration).
2. Acceptance-rate histogram (500-iteration blocks after M, all blocks and chains).
3. Global fields: for every subdomain an independent random subset of --n-fields saved post-M states (all chains
   pooled; blocks are independent given the frozen interface), assembled theta = B c_all (the projection onto the global
   KLE), log kappa = T theta, pressure from a full forward solve; pointwise mean and sd.  With two grids,
   label "pooled" = the same number of assembled fields from each grid.  The relative error of the assembled fields
   against the data is reported.
"""
import _common  # noqa: F401
import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brho import msm_dd as DD, msm_flux as MF  # noqa: E402
from brho.mpsrf import mpsrf_curve, convergence_iteration  # noqa: E402
import run_msm_post as RP  # noqa: E402
from run_dd_chains import build, GRID_TAG  # noqa: E402

REF = "runs/msm_local_kle_2026-11/exp1_global/post/post_summary.json"


def dd_mpsrf_method(start, se):
    return ("DD-MCMC convergence (sup over subdomains). For each subdomain k the MPSRF is computed from the saved "
            "local coefficients c_k (dimension r_k) of its independent chains; the reported curve is the maximum over "
            "subdomains of the MPSRF at every curve point, the same for the max-PSRF curve, and all subdomains are said "
            "to have converged at the first point of the sup curve (ignoring the first 10 curve points) below "
            f"{RP.MPSRF_LEVEL}, linearly interpolated. Per-subdomain computation: " + RP.mpsrf_method(start, se))


def mpsrf_blocks(Z, nb, start, se, level=RP.MPSRF_LEVEL):
    """Per-block and sup MPSRF curves.  Returns (its, mp (nb, nk), ps (nb, nk), dims)."""
    j0 = start // se
    its0 = j0 * se
    mp, ps, dims = [], [], []
    for k in range(nb):
        x = Z[f"c_{k:02d}"][:, j0:].astype(float)
        cur = mpsrf_curve(x, skip=RP.MPSRF_SKIP, step=RP.MPSRF_STEP, iter_burn=1)
        mp.append(cur["mpsrf"]); ps.append(cur["psrf"]); dims.append(int(x.shape[2]))
        its = its0 + (cur["k"] - 1) * se
    return its.astype(float), np.array(mp), np.array(ps), dims


def curve_stats(its, curve, level=RP.MPSRF_LEVEL):
    return dict(final=float(curve[-1]), convergence_iteration=convergence_iteration(its, curve, level))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="run folder(s) of run_dd_chains.py (primal, then dual)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--n-fields", type=int, default=5000, help="assembled global fields PER GRID")
    ap.add_argument("--mpsrf-start", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--ref", default=REF, help="post_summary.json of the global-pCN run (sd ratio), optional")
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--cmap-std", default=None)
    ap.add_argument("--levels", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = dict(kind="dd_mcmc", grids={}, fields={})
    S = data = None
    fields_lk, fields_pr, out_f = {}, {}, {}
    fig_sup, ax_sup = plt.subplots(figsize=(6.4, 3.8))
    fig_ar, axs_ar = plt.subplots(1, len(a.dirs), figsize=(5 * len(a.dirs), 3.4), squeeze=False)
    for gi, d in enumerate(a.dirs):
        summ = json.load(open(os.path.join(d, "summary.json")))
        st = summ["settings"]
        grid_name, n, M, se = st["grid"], st["n_iter"], st["adapt"], st["save_every"]
        Z = np.load(os.path.join(d, "chains.npz"))
        S, data, spline, grid = build(Namespace(**st))
        nb, m = len(grid.blocks), Z["c_00"].shape[0]
        gres = dict(directory=d, n_blocks=nb, r_k=grid.r_k, chains=int(m),
                    interface=summ["interface"], spline=summ["spline"])
        # ---- 1. MPSRF per subdomain and sup
        its, mp, ps, dims = mpsrf_blocks(Z, nb, a.mpsrf_start, se)
        sup_mp, sup_ps = np.max(mp, 0), np.max(ps, 0)              # NaN propagates: a NaN point is not converged
        gres["mpsrf"] = dict(window_start_iteration=int(a.mpsrf_start // se * se),
                             n_samples=int(Z["c_00"].shape[1] - a.mpsrf_start // se),
                             sup=dict(mpsrf_final=float(sup_mp[-1]), psrf_max_final=float(sup_ps[-1]),
                                      convergence_iteration=convergence_iteration(its, sup_mp, RP.MPSRF_LEVEL),
                                      psrf_convergence_iteration=convergence_iteration(its, sup_ps, RP.MPSRF_LEVEL)),
                             blocks=[dict(block=k, dim=dims[k], mpsrf_final=float(mp[k][-1]),
                                          psrf_max_final=float(ps[k][-1]),
                                          convergence_iteration=convergence_iteration(its, mp[k], RP.MPSRF_LEVEL))
                                     for k in range(nb)],
                             worst_block_final=int(np.argmax(mp[:, -1])))
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for k in range(nb):
            ax.plot(its, mp[k], color="0.75", lw=0.7)
        ax.plot(its, sup_mp, color="C0", lw=2, label="sup over subdomains of MPSRF")
        ax.plot(its, sup_ps, color="C1", lw=1.2, alpha=.8, label="sup of max PSRF")
        ax.axhline(RP.MPSRF_LEVEL, color="0.4", ls=":")
        top = np.nanmax(sup_mp[10:]) if len(sup_mp) > 10 else 3
        ax.set_ylim(1, min(3, max(1.3, top * 1.05)))
        ax.set_xlabel("iteration"); ax.set_title(f"{grid_name}: {nb} subdomains, {m} chains each"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"mpsrf_{grid_name}.png"), dpi=130); plt.close(fig)
        ax_sup.plot(its, sup_mp, label=f"{grid_name} (sup MPSRF)")
        # ---- 2. acceptance
        ab = np.concatenate([np.stack([Z[f"acc_{k:02d}"][:, M:M + ((n - M) // 500) * 500].reshape(m, -1, 500).mean(2)
                                       for k in range(nb)]).ravel()])
        axs_ar[0, gi].hist(ab, bins=40, color="C0"); axs_ar[0, gi].axvline(0.25, color="k", ls=":")
        axs_ar[0, gi].set_title(f"AR, {grid_name} (500-it blocks, all subdomains)"); axs_ar[0, gi].set_xlabel("acceptance rate")
        gres["acceptance"] = dict(mean=float(ab.mean()), sd=float(ab.std()), min=float(ab.min()), max=float(ab.max()),
                                  by_block=[b["acc_post_M_mean"] for b in summ["blocks"]])
        # ---- 3. assembled global fields: independent per-block draws of saved post-M states
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, GRID_TAG[grid_name]]))
        cb = []
        for k in range(nb):
            flat = Z[f"c_{k:02d}"][:, M // se:].reshape(-1, grid.r_k[k]).astype(float)
            cb.append(flat[rng.choice(len(flat), min(a.n_fields, len(flat)), replace=False)])
        theta = grid.theta(cb)
        lk = S.kle.field(theta)
        pr = np.empty_like(lk)
        for j in range(len(lk)):
            pr[j] = S.phys.solve(np.exp(lk[j]))
        r2 = ((pr[:, S.phys.obs_mask] - data.d[None]) ** 2).sum(1)
        rel = np.sqrt(r2) / data.dnorm
        fl = getattr(grid, "flux_data", None)
        if fl is not None:                                                      # flux studies: flux misfit of the assembled fields
            nn = S.n
            qm = np.stack([MF.face_flux(np.exp(lk[j]).reshape(nn, nn), pr[j].reshape(nn, nn), fl.ia, fl.ja, fl.ib, fl.jb)
                           for j in range(len(lk))])
            e_q = np.linalg.norm(qm - fl.q[None], axis=1) / fl.qnorm
            gres["assembled_flux_relerr"] = dict(median=float(np.median(e_q)), q05=float(np.percentile(e_q, 5)),
                                                 q95=float(np.percentile(e_q, 95)), noise_level=fl.relerr_truth,
                                                 phi_q_median=float(np.median(0.5 * fl.w * ((qm - fl.q[None]) ** 2).sum(1))))
            print(f"  [{grid_name}] assembled flux relerr median {np.median(e_q):.4f}", flush=True)
        gres["assembled_global_relerr"] = dict(median=float(np.median(rel)), q05=float(np.percentile(rel, 5)),
                                               q95=float(np.percentile(rel, 95)),
                                               phi_median=float(np.median(0.5 * data.w * r2)),
                                               noise_level=float(data.relerr_truth))
        fields_lk[grid_name], fields_pr[grid_name] = lk, pr
        res["grids"][grid_name] = gres
        print(f"{grid_name}: sup MPSRF final {sup_mp[-1]:.3f}, convergence iteration "
              f"{gres['mpsrf']['sup']['convergence_iteration']}, assembled relerr median {np.median(rel):.4f}", flush=True)
    ax_sup.axhline(RP.MPSRF_LEVEL, color="0.4", ls=":"); ax_sup.set_xlabel("iteration"); ax_sup.legend()
    ax_sup.set_ylim(1, 3)
    fig_sup.tight_layout(); fig_sup.savefig(os.path.join(a.out, "mpsrf_sup_all.png"), dpi=130); plt.close(fig_sup)
    fig_ar.tight_layout(); fig_ar.savefig(os.path.join(a.out, "ar_hist.png"), dpi=130); plt.close(fig_ar)
    res["mpsrf_method"] = dd_mpsrf_method(a.mpsrf_start, se)
    print(res["mpsrf_method"])
    # ---- field statistics, per grid and pooled (equal numbers from each grid)
    labels = list(fields_lk)
    sets = {l: (fields_lk[l], fields_pr[l]) for l in labels}
    if len(labels) > 1:
        sets["pooled"] = (np.concatenate([fields_lk[l] for l in labels]), np.concatenate([fields_pr[l] for l in labels]))
    lk_true = S.kle.field(data.theta_true)
    p_true = S.phys.solve(np.exp(lk_true))
    for label, (lk, pr) in sets.items():
        out_f[label] = dict(logk_mean=lk.mean(0), logk_sd=lk.std(0), p_mean=pr.mean(0), p_sd=pr.std(0), n=len(lk))
    ref = None
    if os.path.exists(a.ref):
        ref = json.load(open(a.ref))["fields"]["run"]["logk_sd_mean"]
    for label, f in out_f.items():
        res["fields"][label] = dict(n=f["n"], logk_sd_mean=float(f["logk_sd"].mean()), p_sd_mean=float(f["p_sd"].mean()),
                                    logk_mean_error_rms=float(np.sqrt(np.mean((f["logk_mean"] - lk_true) ** 2))))
        if ref:
            res["fields"][label]["logk_sd_ratio_to_global_pcn"] = res["fields"][label]["logk_sd_mean"] / ref
    # selected (log kappa, pressure) pairs: truth and the first four assembled samples of the first grid
    first = labels[0]
    rows = [("truth", lk_true)] + [(f"{first} sample {j}", fields_lk[first][j]) for j in range(4)]
    pair_lk = np.stack([r[1] for r in rows])
    pair_p = np.stack([S.phys.solve(np.exp(x)) for x in pair_lk])
    pair_e = np.array([np.linalg.norm(data.d - q[S.phys.obs_mask]) / data.dnorm for q in pair_p])
    np.savez_compressed(os.path.join(a.out, "fields.npz"), grid_n=S.n, logk_true=lk_true, p_true=p_true,
                        pair_labels=np.array([r[0] for r in rows]), pair_logk=pair_lk, pair_p=pair_p,
                        pair_relerr=pair_e, phases=np.array(list(out_f)),
                        **{f"{l}_{q}": v for l, d in out_f.items() for q, v in d.items()})
    RP.replot(a.out, a.cmap, a.cmap_std, a.levels)
    json.dump(res, open(os.path.join(a.out, "post_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("mpsrf_method",)}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
