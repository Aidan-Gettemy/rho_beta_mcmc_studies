"""
Prior test (likelihood OFF, every proposal accepted, no PDE solves) for the six
samplers of the local-KLE (MSM) experiment, docs/msm_local_kle_experiment.md
Section 6, step 3:

    1  global pCN            theta ~ N(0, I)                          (must PASS)
    2  primal raw            theta = B c                              (variance deficit)
    3a primal whitened, pinv theta = W_+ B c  (6 null directions out) (cov err ~0.306)
    3b primal whitened + V0 zeta                                       (must PASS)
    4  mixture primal/dual   theta -> Pi_g theta -> pCN -> B_g, p=0.5 (not prior-preserving)
    5  sequence primal -> dual (K = n_iter/2 each)                    (not prior-preserving)

Compared with direct draws theta ~ N(0, I_64): Frobenius error of cov(theta) - I,
law of the spatial mean and empirical variance of the field, pointwise variance
ratio against the exact truncated-KLE prior variance diag(T T^T), and MPSRF
(range coordinates V_+^T theta for the runs whose theta lives in the 58-dim range).

Guard: primal r_+ must be 58 and dual r_+ 64 (the decision of 2026-11-02);
anything else stops the run.

    BRHO_ROOT=runs/msm_local_kle_2026-11 python experiments/run_msm_prior_test.py
Writes figures/msm_prior_test/*.png and tables/msm_prior_test/msm_prior_test.{json,md}.
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brho import io  # noqa: E402
from brho import diagnostics as D  # noqa: E402
from brho.mpsrf import get_mpsrf  # noqa: E402
from brho import msm_local as ML  # noqa: E402

EXP = "msm_prior_test"
ORDER = ("reference", "1_global", "2_primal_raw", "3a_pinv", "3b_aug", "4_mixture",
         "5_phase1", "5_phase2")
COL = dict(zip(ORDER, ("k", "C0", "C3", "C1", "C2", "C4", "C5", "C6")))


def stats(th, kle, mp_coords=None):
    flat = th.reshape(-1, th.shape[-1])
    r = flat.shape[1]
    Cv = np.cov(flat.T)
    G = kle.field(flat)
    pmean, pvar = D.pointwise_moments(flat, kle.T)
    target = (kle.T ** 2).sum(1)
    ratio = pvar / target
    mp = None
    if mp_coords is not None:
        mp = float(get_mpsrf(mp_coords(th))[0])
    return dict(frob=float(np.linalg.norm(Cv - np.eye(r)) / np.sqrt(r)),
                diag=np.sort(np.diag(Cv))[::-1], mean=G.mean(1), var=G.var(1),
                ratio=ratio, pmean_max=float(np.abs(pmean).max()), mpsrf=mp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", type=int, default=64)
    ap.add_argument("--n-iter", type=int, default=40000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=20)
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=51)
    a = ap.parse_args(argv)

    S = ML.MSMSetup()
    P, Dg, kle = S.primal, S.dual, S.kle
    print(f"r={S.r}; primal p={P.p} r_+={P.r_plus}; dual p={Dg.p} r_+={Dg.r_plus}")
    if P.r_plus != 58 or Dg.r_plus != 64:
        print("GUARD: rank outcome differs from the recorded decision (58 / 64); stop.")
        return 1

    def rng(j):
        return np.random.default_rng(a.seed + j)

    args = (a.chains, a.n_iter, a.thin, a.burn, a.s)
    rng_coords = lambda th: ML.range_coords(P, th)
    res = {}
    n_keep = (a.n_iter - a.burn + a.thin - 1) // a.thin
    ref = rng(99).standard_normal((a.chains, n_keep, S.r))
    res["reference"] = stats(ref, kle, None)
    th = ML.prior_chain("global", S, *args, rng(0))["theta"]
    res["1_global"] = stats(th, kle, lambda t: t)
    th = ML.prior_chain("primal_raw", S, *args, rng(1))["theta"]
    res["2_primal_raw"] = stats(th, kle, rng_coords)
    th = ML.prior_chain("primal_pinv", S, *args, rng(2))["theta"]
    res["3a_pinv"] = stats(th, kle, rng_coords)
    o = ML.prior_chain("primal_aug", S, *args, rng(3))
    res["3b_aug"] = stats(o["theta"], kle, lambda t: t)
    th = ML.prior_chain("mixture", S, *args, rng(4))["theta"]
    res["4_mixture"] = stats(th, kle, lambda t: t)
    o = ML.prior_chain("sequence", S, a.chains, a.n_iter, a.thin, a.burn, a.s, rng(5))
    res["5_phase1"] = stats(o["theta1"], kle, rng_coords)
    res["5_phase2"] = stats(o["theta2"], kle, lambda t: t)
    switch = o["switch_field_change"]
    print(f"phase-switch relative field change (primal -> dual projection): {switch:.4f}")

    fdir, tdir = io.figures_dir(EXP), io.tables_dir(EXP)
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for k in ORDER:
        ax[0].plot(res[k]["diag"], color=COL[k], lw=1.6, label=k)
    ax[0].axhline(1, color="0.5", ls=":")
    ax[0].set_xlabel("sorted index"); ax[0].set_title(r"diagonal of cov($\theta$) (target 1)")
    ax[0].legend(fontsize=7)
    rm, rv = res["reference"]["mean"], res["reference"]["var"]
    b = np.linspace(-4 * rm.std(), 4 * rm.std(), 50)
    for k in ORDER:
        ax[1].hist(res[k]["mean"], b, density=True, histtype="step", color=COL[k],
                   label=f"{k} (sd {res[k]['mean'].std():.3f})")
    ax[1].set_title("law of the spatial mean of the field"); ax[1].legend(fontsize=7)
    b = np.linspace(0, rv.mean() + 5 * rv.std(), 50)
    for k in ORDER:
        ax[2].hist(res[k]["var"], b, density=True, histtype="step", color=COL[k],
                   label=f"{k} ({res[k]['var'].mean():.3f})")
    ax[2].set_title("law of the empirical variance"); ax[2].legend(fontsize=7)
    fig.suptitle(f"MSM prior test (likelihood off): s={a.s}, {a.chains} chains x {a.n_iter} it.")
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "fig_msm_prior_test.png"), dpi=150)
    plt.close(fig)

    table = {k: dict(frob=res[k]["frob"], spatial_mean_sd=float(res[k]["mean"].std()),
                     field_var_mean=float(res[k]["var"].mean()),
                     theta_var_max=float(res[k]["diag"][0]),
                     theta_var_min=float(res[k]["diag"][-1]),
                     pvar_ratio_min=float(res[k]["ratio"].min()),
                     pvar_ratio_max=float(res[k]["ratio"].max()),
                     pvar_ratio_mean=float(res[k]["ratio"].mean()),
                     pmean_max_abs=res[k]["pmean_max"], mpsrf=res[k]["mpsrf"])
             for k in ORDER}
    io.save_json(os.path.join(tdir, "msm_prior_test.json"),
                 dict(settings=vars(a), primal=P.summary(), dual=Dg.summary(),
                      switch_field_change=switch, results=table))
    with open(os.path.join(tdir, "msm_prior_test.md"), "w") as f:
        f.write("| sampler | cov err (Frob/√r) | θ var max / min | sd spatial mean | "
                "field var mean | pointwise var ratio min–max (mean) | MPSRF |\n"
                "|---|---|---|---|---|---|---|\n")
        for k in ORDER:
            t = table[k]
            mp = "-" if t["mpsrf"] is None else f"{t['mpsrf']:.3f}"
            f.write(f"| {k} | {t['frob']:.3f} | {t['theta_var_max']:.2f} / {t['theta_var_min']:.2f}"
                    f" | {t['spatial_mean_sd']:.3f} | {t['field_var_mean']:.3f} | "
                    f"{t['pvar_ratio_min']:.2f}–{t['pvar_ratio_max']:.2f} ({t['pvar_ratio_mean']:.3f})"
                    f" | {mp} |\n")
    print(open(os.path.join(tdir, "msm_prior_test.md")).read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
