"""
Post-processing of one experiment of the local-KLE (MSM) study, from <dir>/chains.npz and
<dir>/summary.json written by experiments/run_msm_chains.py.  Writes <dir>/post/:
    ar_hist.png, relerr_traces.png, mpsrf_*.png, fields_*.png, post_summary.json, fields.npz

1. Acceptance-rate histograms: all chains, blocks of 500 iterations after M (experiment 4:
   per grid; experiment 5: per phase, after M within each phase).
2. Relative-error traces (every iteration): median and 5-95% band over chains.
3. MPSRF (brho/mpsrf.py, the group's MATLAB convention: skip 2, step 15, burn 10) on the saved
   samples after M (every 100th iteration, from iteration 0 saved):
     global theta; local coefficients c (3b: joint (c, zeta); 4: both projections; 5: per phase).
   theta that lives in the 58-dim range of the rank-deficient primal Sigma_ms (experiments 2, 3a,
   phase 1 of 5) is taken in range coordinates V_+^T theta (brho.msm_local.range_coords).
4. Posterior fields: n_fields saved post-M global thetas (experiment 5: equal numbers from each
   phase) -> log kappa = T theta, pressure from a full forward solve; pointwise mean and sd.

    python experiments/run_msm_post.py --dir runs/msm_local_kle_2026-11/exp2_primal_raw
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

from brho import msm_local as ML, msm_sampler as MS  # noqa: E402
from brho.mpsrf import mpsrf_curve, convergence_iteration  # noqa: E402

RANGE_KINDS = ("primal_raw", "primal_pinv")


def ar_blocks(acc, move, lo, hi, g=None, block=500):
    """(chains, n_blocks) acceptance per block; g selects a grid for experiment 4."""
    a, m = acc[:, lo:hi].astype(float), move[:, lo:hi]
    nb = (hi - lo) // block
    a, m = a[:, :nb * block].reshape(len(a), nb, block), m[:, :nb * block].reshape(len(a), nb, block)
    if g is None:
        return a.mean(2)
    sel = (m == g)
    return (a * sel).sum(2) / np.maximum(sel.sum(2), 1)


def mpsrf_of(x, x0_iter, save_every, name, fdir, res, level=1.2):
    """x (m, n, dim) saved samples; x0_iter iteration of its first sample."""
    try:
        cur = mpsrf_curve(x, skip=2, step=15, iter_burn=10)
    except Exception as e:                       # pragma: no cover
        res[name] = dict(error=str(e)); return
    if len(cur["k"]) == 0 or np.all(np.isnan(cur["mpsrf"])):
        res[name] = dict(dim=int(x.shape[2]), n_samples=int(x.shape[1]), note="too few samples"); return
    its = x0_iter + cur["k"] * save_every
    conv = convergence_iteration(its.astype(float), cur["mpsrf"], level)
    res[name] = dict(dim=int(x.shape[2]), n_samples=int(x.shape[1]),
                     mpsrf_final=float(cur["mpsrf"][-1]), psrf_max_final=float(cur["psrf"][-1]),
                     convergence_iteration=conv)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(its, cur["mpsrf"], label="MPSRF"); ax.plot(its, cur["psrf"], label="max PSRF", alpha=.7)
    ax.axhline(level, color="0.5", ls=":"); ax.set_ylim(1, min(3, max(1.3, np.nanmax(cur["mpsrf"][10:]) * 1.05)) if len(cur["mpsrf"]) > 10 else 3)
    ax.set_xlabel("iteration"); ax.set_title(f"{name} (dim {x.shape[2]}, {x.shape[0]} chains)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(fdir, f"mpsrf_{name}.png"), dpi=130); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n-fields", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    summ = json.load(open(os.path.join(a.dir, "summary.json")))
    st = summ["settings"]
    kind, n, M, se, K = st["kind"], st["n_iter"], st["adapt"], st["save_every"], st.get("phase_k")
    Z = np.load(os.path.join(a.dir, "chains.npz"))
    fdir = os.path.join(a.dir, "post"); os.makedirs(fdir, exist_ok=True)
    S = ML.MSMSetup(); P, D = S.primal, S.dual
    if P.r_plus != 58 or D.r_plus != 64:
        raise SystemExit("GUARD: rank outcome changed (58 / 64 expected)")
    data = MS.Data(S, st["sigma_rel"], st["data_seed"], data_relerr=st.get("data_relerr"))
    m = Z["theta"].shape[0]
    res = dict(kind=kind, chains=int(m))
    theta = Z["theta"].astype(float)                              # (m, n/se, r)
    # ---- phases: (name, first saved index, last+1, first iteration, acceptance window)
    if kind == "sequence":
        ph = [("phase1", M // se, K // se, M, (M, K)), ("phase2", K // se + M // se, n // se, K + M, (K + M, n))]
    else:
        ph = [("run", M // se, n // se, M, (M, n))]
    # ---- 1. acceptance histograms
    fig, axs = plt.subplots(1, len(ph) * (2 if kind == "mixture" else 1), figsize=(5 * len(ph) * (2 if kind == "mixture" else 1), 3.4), squeeze=False)
    k = 0
    for name, i0, i1, it0, (lo, hi) in ph:
        for g in ((0, 1) if kind == "mixture" else (None,)):
            ab = ar_blocks(Z["acc"], Z["move"], lo, hi, g)
            axs[0, k].hist(ab.ravel(), bins=30, color="C0"); axs[0, k].axvline(0.25, color="k", ls=":")
            lab = name + ("" if g is None else (" primal" if g == 0 else " dual"))
            axs[0, k].set_title(f"AR, {lab} (500-it blocks)"); axs[0, k].set_xlabel("acceptance rate")
            res.setdefault("acceptance", {})[lab] = dict(mean=float(ab.mean()), sd=float(ab.std()),
                                                         min=float(ab.min()), max=float(ab.max()))
            k += 1
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "ar_hist.png"), dpi=130); plt.close(fig)
    # ---- 2. relative-error traces
    rel = Z["relerr"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(rel.shape[1])
    ax.semilogy(x, np.median(rel, 0), label="median over chains")
    ax.fill_between(x, np.percentile(rel, 5, 0), np.percentile(rel, 95, 0), alpha=.3, label="5-95%")
    ax.axvline(M, color="k", ls=":"); ax.axhline(data.relerr_truth, color="C3", ls="--", label="noise level")
    if kind == "sequence": ax.axvline(K, color="k", ls="-.")
    ax.set_xlabel("iteration"); ax.set_ylabel("relative error"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(fdir, "relerr_traces.png"), dpi=130); plt.close(fig)
    res["relerr_post_M_median"] = float(np.median(rel[:, M:]))
    # ---- 3. MPSRF
    mp = {}
    for name, i0, i1, it0, _ in ph:
        th = theta[:, i0:i1]
        tag = "" if kind != "sequence" else f"_{name}"
        use_range = kind in RANGE_KINDS or (kind == "sequence" and name == "phase1")
        mpsrf_of(ML.range_coords(P, th) if use_range else th, it0, se, "theta" + tag + ("_range58" if use_range else ""), fdir, mp)
        if kind in ("primal_raw", "primal_pinv", "primal_aug"):
            c = Z["c"][:, i0:i1].astype(float)
            if kind == "primal_aug":
                mpsrf_of(np.concatenate([c, Z["zeta"][:, i0:i1].astype(float)], 2), it0, se, "c_zeta_joint", fdir, mp)
            else:
                mpsrf_of(c, it0, se, "c", fdir, mp)
        if kind == "mixture":
            mpsrf_of(Z["c_primal"][:, i0:i1].astype(float), it0, se, "c_primal_proj", fdir, mp)
            mpsrf_of(Z["c_dual"][:, i0:i1].astype(float), it0, se, "c_dual_proj", fdir, mp)
        if kind == "sequence":
            h = K // se
            c = (Z["c1"][:, i0:i1] if name == "phase1" else Z["c2"][:, i0 - h:i1 - h]).astype(float)
            mpsrf_of(c, it0, se, f"c_{name}", fdir, mp)
    if kind == "sequence":                       # theta across the full run after M
        mpsrf_of(theta[:, M // se:], M, se, "theta_full_run", fdir, mp)
    res["mpsrf"] = mp
    # ---- 4. posterior fields
    rng = np.random.default_rng(a.seed)
    groups = [(i0, i1) for _, i0, i1, _, _ in ph]
    per = a.n_fields // len(groups)
    pools = []
    for (i0, i1) in groups:
        flat = theta[:, i0:i1].reshape(-1, S.r)
        pools.append(flat[rng.choice(len(flat), per, replace=False)])
    out_f = {}
    for label, th_s in list(zip([p[0] for p in ph], pools)) + ([("pooled", np.concatenate(pools))] if len(pools) > 1 else []):
        lk = S.kle.field(th_s)                                  # (n, N)
        pr = np.empty_like(lk)
        for j in range(len(th_s)):
            pr[j] = S.phys.solve(np.exp(lk[j]))
        out_f[label] = dict(logk_mean=lk.mean(0), logk_sd=lk.std(0), p_mean=pr.mean(0), p_sd=pr.std(0), n=len(th_s))
    lk_true = S.kle.field(data.theta_true); p_true = S.phys.solve(np.exp(lk_true))
    np.savez_compressed(os.path.join(fdir, "fields.npz"), logk_true=lk_true, p_true=p_true,
                        **{f"{l}_{q}": v for l, d in out_f.items() for q, v in d.items()})
    n_ = S.n
    for label, d in out_f.items():
        fig, ax = plt.subplots(1, 5, figsize=(19, 3.4))
        panels = [("log k truth", lk_true), ("log k mean", d["logk_mean"]), ("log k sd", d["logk_sd"]),
                  ("p mean", d["p_mean"]), ("p sd", d["p_sd"])]
        for axx, (t, f) in zip(ax, panels):
            im = axx.imshow(f.reshape(n_, n_).T, origin="lower", extent=(0, 1, 0, 1)); axx.set_title(f"{t} ({label})")
            fig.colorbar(im, ax=axx, fraction=.046)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, f"fields_{label}.png"), dpi=120); plt.close(fig)
    res["fields"] = {l: dict(n=d["n"], logk_sd_mean=float(d["logk_sd"].mean()), p_sd_mean=float(d["p_sd"].mean()),
                             logk_mean_error_rms=float(np.sqrt(np.mean((d["logk_mean"] - lk_true) ** 2))))
                     for l, d in out_f.items()}
    json.dump(res, open(os.path.join(fdir, "post_summary.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
