"""
Spectrum check before E6: is the theta posterior strongly anisotropic, and is
pCN slow in the broad (prior-like) directions?

    python reports/report_spectrum.py [--exp e5_soft] [--chains 64]

Uses the kept (post-burn-in) theta samples of each run: E5 keeps only those;
for E1 the samples before n_burn are dropped.  For each run:

    Sigma   pooled posterior covariance of theta (all chains, all samples)
    W       mean within-chain covariance (what one chain explores)
    v_k     eigenvectors of Sigma, sd_k = sqrt(eigenvalue): the posterior sd
            along principal direction k (prior sd = 1 in every direction)
    R_k     PSRF of the projection theta . v_k (the getMSPRF V/W formula,
            whole post-burn-in window)
    cos_k   |cos| between v_k and c (KLE coordinates of the constant field,
            the direction that trades against log beta)

Also, because the theta move keeps (beta, rho) fixed: the conditional
covariance of theta given (log beta, log rho) (linear / Schur-complement
estimate), its spectrum, the widths along the level direction (Gbar), and
sum_k (s / sd_k)^2 for the marginal and conditional widths (a Gaussian
random walk accepts ~25% when this is ~2.38^2 = 5.7).

The E5 report predicts:
  (1) sd_k spans orders of magnitude: a few tight, data-informed directions
      and a prior-like remainder (sd near 1);
  (2) the pCN step s is of the order of the tightest sd (s / sd_min ~ 1), so
      the step relative to the broad directions (s / sd_max) is tiny;
  (3) E5: R_k grows with sd_k (the broad directions are the unmixed ones);
      E1: the largest R_k sits on a direction aligned with c (the beta ridge).
If sd_k is flat (sd_max / sd_min of a few), the E6 argument is wrong.

Caveat: Sigma comes from chains that have not converged (MPSRF 1.56 in E5),
so the broad directions are, if anything, under-estimated; the ordering and
the order of magnitude are what matter here.

Writes figures/<exp>/fig_spectrum.png, tables/<exp>/spectrum.json,
tables/<exp>/spectrum.npz (mean, Sigma, eigenvalues and vectors, R_k, cos_k),
and with two runs figures/spectrum_compare/fig_spectrum_compare.png.
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from _common import io
from brho import mpsrf as M
from report_mpsrf import _chain_paths

COLORS = {"e1_blocked": "#c0392b", "e1_global": "#e67e22", "e5_soft": "#2e8b57"}
LABELS = {"e1_blocked": "E1 naive pCN (blocked)", "e1_global": "E1 naive pCN (global)",
          "e5_soft": "E5 soft-curve sampler"}


# --------------------------------------------------------------- analysis
def spectrum(theta, c=None, step=None, hyper=None, u_level=None):
    """theta: (m chains, n samples, r); hyper: (m, n, h) = (log beta, log rho)
    at the same samples; u_level: the vector with Gbar = u_level . theta.
    Pure function (tested in tests/)."""
    th = np.asarray(theta, float)
    m, n, r = th.shape
    flat = th.reshape(-1, r)
    mean = flat.mean(0)
    Sigma = np.cov(flat, rowvar=False)
    W = np.mean([np.cov(th[j], rowvar=False) for j in range(m)], axis=0)
    lam, V = np.linalg.eigh(Sigma)
    order = np.argsort(lam)[::-1]                    # broadest first
    lam, V = lam[order], V[:, order]
    sd = np.sqrt(np.clip(lam, 0, None))
    sd_w = np.sqrt(np.clip(np.sort(np.linalg.eigvalsh(W))[::-1], 0, None))
    y = th @ V                                       # samples in the eigenbasis
    _, _, R = M.get_mpsrf(y)
    out = dict(r=r, m=m, n=n, mean=mean, Sigma=Sigma, lam=lam, V=V, sd=sd,
               sd_within=sd_w, R=R)
    out["cos_c"] = (np.abs(V.T @ c) / np.linalg.norm(c)) if c is not None \
        else np.full(r, np.nan)
    s = dict(
        sd_max=float(sd[0]), sd_min=float(sd[-1]),
        sd_ratio=float(sd[0] / sd[-1]),
        n_sd_below_0p1=int((sd < 0.1).sum()), n_sd_below_0p3=int((sd < 0.3).sum()),
        n_sd_0p3_to_0p8=int(((sd >= 0.3) & (sd < 0.8)).sum()),
        n_sd_above_0p8=int((sd >= 0.8).sum()),
        trace_ratio=float(lam.sum() / r),            # 1 = prior
        R_max=float(R.max()), R_argmax=int(R.argmax()),
        sd_at_R_max=float(sd[R.argmax()]),
        n_R_above_1p2=int((R > 1.2).sum()),
        # does R grow with sd?  Spearman rank correlation, no scipy needed
        spearman_R_vs_sd=float(np.corrcoef(np.argsort(np.argsort(sd)),
                                           np.argsort(np.argsort(R)))[0, 1]),
    )
    if c is not None:
        k = int(np.argmax(out["cos_c"]))
        s.update(c_direction=k, c_direction_cos=float(out["cos_c"][k]),
                 c_direction_sd=float(sd[k]), c_direction_R=float(R[k]),
                 cos_c_at_R_max=float(out["cos_c"][R.argmax()]))
    if step is not None:
        s.update(pcn_step=float(step),
                 step_over_sd_min=float(step / sd[-1]),
                 step_over_sd_max=float(step / sd[0]),
                 # AR(1) estimate tau ~ 4 / (a s_k^2) in direction k, with s_k
                 # the step relative to that direction's sd; a = 0.25
                 tau_broadest_estimate=float(4 / (0.25 * (step / sd[0]) ** 2)))
    if hyper is not None:
        # The theta move acts with (beta, rho) FIXED, so what limits its step is
        # the CONDITIONAL width of theta given (log beta, log rho).  Linear
        # (Schur complement) estimate from the joint samples: it is an upper
        # bound on the average conditional covariance (nonlinear dependence
        # left in the residual only adds variance).
        hh = np.asarray(hyper, float).reshape(-1, np.shape(hyper)[-1])
        J = np.cov(np.hstack([flat, hh]), rowvar=False)
        Stt, Sth, Shh = J[:r, :r], J[:r, r:], J[r:, r:]
        B = np.linalg.solve(Shh, Sth.T).T                # theta ~ a + B h
        Sc = Stt - B @ Sth.T
        Sc = 0.5 * (Sc + Sc.T)
        lc, Vc = np.linalg.eigh(Sc)
        oc = np.argsort(lc)[::-1]
        lc, Vc = lc[oc], Vc[:, oc]
        sd_c = np.sqrt(np.clip(lc, 0, None))
        out.update(Sigma_cond=Sc, B_cond=B, mean_h=hh.mean(0), lam_cond=lc,
                   V_cond=Vc, sd_cond=sd_c)
        s.update(sd_cond_max=float(sd_c[0]), sd_cond_min=float(sd_c[-1]),
                 sd_cond_ratio=float(sd_c[0] / sd_c[-1]))
        if u_level is not None:
            u = np.asarray(u_level, float) / np.linalg.norm(u_level)
            s.update(sd_marg_along_level=float(np.sqrt(u @ Sigma @ u)),
                     sd_cond_along_level=float(np.sqrt(u @ Sc @ u)),
                     cos_level_tightest_cond=float(abs(Vc[:, -1] @ u)))
        if step is not None:
            # Gaussian random-walk rule of thumb: ~25% acceptance when
            # sum_k (s / sd_k)^2 ~ 2.38^2 = 5.7 (sd_k: widths the move sees)
            s.update(step_over_sd_cond_min=float(step / sd_c[-1]),
                     sum_step_over_sd2_marginal=float(np.sum((step / sd) ** 2)),
                     sum_step_over_sd2_conditional=float(np.sum((step / sd_c) ** 2)),
                     rule_of_thumb_sum=2.38 ** 2)
    out["summary"] = s
    return out


# --------------------------------------------------------------- loading
def load_theta(exp, n_chains=None):
    """Post-burn-in theta samples (m, n, r), (log beta, log rho) at the same
    samples (m, n, 2), the constant-field vector c, the level vector
    u_level (Gbar = u_level . theta) and the median final theta pCN step."""
    paths = _chain_paths(exp, n_chains)
    zs = [np.load(p) for p in paths]
    man = io.read_manifest(exp)
    L = min(len(z["kept_theta"]) for z in zs)
    theta = np.stack([z["kept_theta"][:L] for z in zs])
    lr = np.log(np.stack([z["kept_rho"][:L] for z in zs]))
    if "kept_logbeta" in zs[0].files:
        lb = np.stack([z["kept_logbeta"][:L] for z in zs])
    else:
        lb = np.log(np.stack([z["kept_beta"][:L] for z in zs]))
    hyper = np.stack([lb, lr], axis=2)
    if "kept_it" in zs[0].files:                     # E1: kept from iteration 0
        nb = man["configs"]["NaiveConfig"]["n_burn"]
        keep = np.asarray(zs[0]["kept_it"][:L]) >= nb
        theta, hyper = theta[:, keep], hyper[:, keep]
    key = "s" if "s" in zs[0].files else ("s_final" if "s_final" in zs[0].files else None)
    step = float(np.median([np.atleast_1d(z[key])[0] for z in zs])) if key else None
    kle = os.path.join(io.results_dir(exp), "kle.npz")
    c = u_level = None
    if os.path.exists(kle):
        T = np.load(kle)["T"]
        c, *_ = np.linalg.lstsq(T, np.ones(T.shape[0]), rcond=None)
        u_level = T.mean(0)
    return theta, hyper, c, u_level, step


# --------------------------------------------------------------- figures
def fig_one(exp, res, path):
    sd, R, cos = res["sd"], res["R"], res["cos_c"]
    s = res["summary"]
    k = np.arange(1, len(sd) + 1)
    f, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    ax[0].semilogy(k, sd, "o-", ms=4, lw=1.5, color="#1f4e79",
                   label=r"pooled posterior: $\sqrt{\mathrm{eig}(\Sigma)}$")
    ax[0].semilogy(k, res["sd_within"], "s--", ms=3, lw=1, color="#7f9fbf",
                   label=r"within-chain: $\sqrt{\mathrm{eig}(W)}$")
    if "sd_cond" in res:
        ax[0].semilogy(k, res["sd_cond"], "^-", ms=3, lw=1.2, color="#d4a017",
                       label=r"given $(\beta,\rho)$: $\sqrt{\mathrm{eig}(\Sigma_{\theta|\beta,\rho})}$")
    ax[0].axhline(1.0, color="k", lw=1, ls=":", label="prior sd = 1")
    if "pcn_step" in s:
        ax[0].axhline(s["pcn_step"], color="r", lw=1.5, ls="--",
                      label=f"pCN step s = {s['pcn_step']:.4f}")
    ax[0].set_xlabel("principal direction k (broadest first)")
    ax[0].set_ylabel("posterior sd along direction k")
    ax[0].set_title(f"spectrum: sd from {s['sd_min']:.3g} to {s['sd_max']:.3g} "
                    f"(ratio {s['sd_ratio']:.0f})")
    ax[0].legend(fontsize=10)
    sc = ax[1].scatter(sd, R, c=cos, cmap="viridis", vmin=0, vmax=1, s=36,
                       edgecolor="k", lw=0.3)
    ax[1].axhline(1.2, color="r", ls=":", lw=1.5, label="1.2")
    ax[1].axhline(1.0, color="r", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("posterior sd along direction k")
    ax[1].set_ylabel(r"PSRF of $\theta\cdot v_k$")
    ax[1].set_title(f"mixing per direction (rank corr. with sd: "
                    f"{s['spearman_R_vs_sd']:+.2f})")
    f.colorbar(sc, ax=ax[1], label=r"$|\cos(v_k, c)|$  (alignment with the $\beta$ ridge)")
    ax[1].legend(fontsize=10)
    f.suptitle(f"{LABELS.get(exp, exp)}: {res['m']} chains x {res['n']} "
               f"post-burn-in samples, r = {res['r']}")
    f.tight_layout()
    f.savefig(path, dpi=150)
    plt.close(f)


def fig_compare(results, path):
    f, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    for exp, res in results.items():
        col = COLORS.get(exp, None)
        k = np.arange(1, res["r"] + 1)
        lab = LABELS.get(exp, exp)
        ax[0].semilogy(k, res["sd"], "o-", ms=3, color=col, label=lab)
        if "sd_cond" in res:
            ax[0].semilogy(k, res["sd_cond"], "^:", ms=3, color=col,
                           label=lab + r": given $(\beta,\rho)$")
        if "pcn_step" in res["summary"]:
            ax[0].axhline(res["summary"]["pcn_step"], color=col, ls="--", lw=1,
                          label=f"{lab}: pCN step")
        ax[1].scatter(res["sd"], res["R"], color=col, s=24, label=lab)
    ax[0].axhline(1.0, color="k", ls=":", lw=1, label="prior sd = 1")
    ax[0].set_xlabel("principal direction k (broadest first)")
    ax[0].set_ylabel("posterior sd along direction k")
    ax[0].legend(fontsize=9)
    ax[1].axhline(1.2, color="r", ls=":", lw=1.5)
    ax[1].axhline(1.0, color="r", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("posterior sd along direction k")
    ax[1].set_ylabel(r"PSRF of $\theta\cdot v_k$")
    ax[1].legend(fontsize=9)
    f.suptitle("theta posterior spectrum and per-direction mixing")
    f.tight_layout()
    f.savefig(path, dpi=150)
    plt.close(f)


def run(exp, n_chains=None):
    theta, hyper, c, u_level, step = load_theta(exp, n_chains)
    res = spectrum(theta, c, step, hyper=hyper, u_level=u_level)
    fdir, tdir = io.figures_dir(exp), io.tables_dir(exp)
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    fig_one(exp, res, os.path.join(fdir, "fig_spectrum.png"))
    io.save_json(os.path.join(tdir, "spectrum.json"), res["summary"])
    np.savez_compressed(os.path.join(tdir, "spectrum.npz"),
                        **{k: res[k] for k in ("mean", "Sigma", "lam", "V", "sd",
                                               "sd_within", "R", "cos_c", "Sigma_cond",
                                               "B_cond", "mean_h", "lam_cond", "V_cond",
                                               "sd_cond") if k in res})
    print(f"[{exp}] {res['m']} chains x {res['n']} samples")
    for key, v in res["summary"].items():
        print(f"    {key:24s} {v:.4g}" if isinstance(v, float) else f"    {key:24s} {v}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=None, help="one run (default: E1 blocked and E5)")
    ap.add_argument("--chains", type=int, default=None, help="use the first N chains")
    a = ap.parse_args(argv)
    exps = [a.exp] if a.exp else [
        e for e in (io.exp_e1("blocked"), io.EXP_SOFT)
        if glob.glob(os.path.join(io.results_dir(e), "chain*.npz"))]
    if not exps:
        raise SystemExit("no chain files found (is BRHO_ROOT set?)")
    results = {e: run(e, a.chains) for e in exps}
    if len(results) > 1:
        d = io.figures_dir("spectrum_compare")
        os.makedirs(d, exist_ok=True)
        fig_compare(results, os.path.join(d, "fig_spectrum_compare.png"))
        print(f"  wrote {os.path.relpath(d, io.root())}/fig_spectrum_compare.png")


if __name__ == "__main__":
    main()
