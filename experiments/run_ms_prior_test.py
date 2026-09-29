"""
Multiscale prior test on the 2D problem (E7, step 1): the 2D counterpart of
F. Pereira, "Variance control in distribution for a multiscale (local-KLE)
pCN sampler", Sec. 3 (3D plug, multiscale_whitening.py).

Pure pCN on theta with the likelihood switched OFF (no PDE solves), for
three innovations built from local KLEs on ms_blocks x ms_blocks subdomains
(brho/multiscale.py):
    raw         xi = B eta                 law N(0, Sigma_ms)   -> must FAIL
    whitened    xi = W B eta + V0 zeta     law N(0, I_r)        -> must PASS
    normalized  whitened, then each proposed FIELD forced to spatial mean 0
                and variance 1 and re-projected (a nonlinear map whose
                Jacobian never enters the ratio)          -> must FAIL
against direct prior draws theta ~ N(0, I_r).  Checks: cov(theta) = I_r,
the law of the spatial mean of G = T theta, the law of its empirical variance.

    python experiments/run_ms_prior_test.py [--from-exp e5_soft] [--blocks 2]
           [--local-energy 0.95] [--chains 16] [--n-iter 40000] [--s 0.5]

The KLE (and grid) is read from a frozen run of the study (default e5_soft),
so the test uses exactly the basis the samplers use.
Writes figures/ms_prior_test/fig_ms_prior_test.png and
tables/ms_prior_test/ms_prior_test.{json,md}.
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import C  # noqa: E402
from brho import io  # noqa: E402
from brho.problem import Problem  # noqa: E402
from brho.multiscale import from_problem  # noqa: E402
from brho.mpsrf import get_mpsrf  # noqa: E402

EXP = "ms_prior_test"
VARIANTS = ("raw", "whitened", "normalized")
COL = {"reference": "k", "raw": "C3", "whitened": "C0", "normalized": "C2"}


def run_chains(kind, ms, kle, m, n_iter, thin, s, seed):
    rng = np.random.default_rng(seed)
    r = kle.r
    th = rng.standard_normal((m, r))                 # start at prior draws
    innov = "ms_raw" if kind == "raw" else "ms_white"
    kept = []
    for it in range(n_iter):
        xi = np.stack([ms.draw(innov, rng) for _ in range(m)])
        th = np.sqrt(1 - s * s) * th + s * xi        # Phi = 0: always accepted
        if kind == "normalized":
            G = kle.field(th)
            G = (G - G.mean(1, keepdims=True)) / G.std(1, keepdims=True)
            th = kle.project(G.T).T
        if it % thin == 0:
            kept.append(th.copy())
    return np.stack(kept, 1)                          # (m, n, r)


def stats(th, kle):
    flat = th.reshape(-1, th.shape[-1])
    r = flat.shape[1]
    Cv = np.cov(flat.T)
    G = kle.field(flat)
    return dict(frob=float(np.linalg.norm(Cv - np.eye(r)) / np.sqrt(r)),
                diag=np.sort(np.diag(Cv))[::-1], mean=G.mean(1), var=G.var(1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-exp", default=io.EXP_SOFT)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--local-energy", type=float, default=0.95)
    ap.add_argument("--chains", type=int, default=16)
    ap.add_argument("--n-iter", type=int, default=40000)
    ap.add_argument("--thin", type=int, default=4)
    ap.add_argument("--s", type=float, default=0.5, help="pCN step (Pereira: 0.5)")
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args(argv)

    man = io.read_manifest(a.from_exp)
    pc = C.from_dict(C.ProblemConfig, man["configs"]["ProblemConfig"])
    prob = Problem(pc, frozen=io.results_dir(a.from_exp))
    kle = prob.kle
    ms = from_problem(prob, a.blocks, a.local_energy)
    msum = ms.summary()
    print(f"KLE from {a.from_exp}: {pc.n}x{pc.n}, ell={pc.ell}, r={kle.r}; multiscale "
          f"{a.blocks}x{a.blocks} subdomains, local modes {msum['r_k']} (p={msum['p']}); "
          f"Sigma_ms eigenvalues [{msum['sigma_ms_eig_min']:.3f}, "
          f"{msum['sigma_ms_eig_max']:.3f}]; r_plus={msum['r_plus']}; "
          f"whitening check {msum['operator_check']:.1e}")

    res, psrf = {}, {}
    for j, kind in enumerate(VARIANTS):
        th = run_chains(kind, ms, kle, a.chains, a.n_iter, a.thin, a.s, a.seed + j)
        res[kind] = stats(th, kle)
        psrf[kind] = float(get_mpsrf(th)[1])
        print(f"  {kind:10s} done")
    n_ref = a.chains * (a.n_iter // a.thin)
    ref = np.random.default_rng(a.seed + 99).standard_normal((1, n_ref, kle.r))
    res["reference"] = stats(ref, kle)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    order = ("reference", "raw", "whitened", "normalized")
    for k in order:
        ax[0].plot(res[k]["diag"], color=COL[k], lw=1.8, label=k)
    ax[0].axhline(1, color="0.5", ls=":")
    ax[0].set_xlabel("sorted index"); ax[0].set_ylabel(r"variance of $\theta_i$ (chain)")
    ax[0].set_title(r"diagonal of cov($\theta$) (target 1)")
    ax[0].legend(fontsize=9)
    ref_m, ref_v = res["reference"]["mean"], res["reference"]["var"]
    bins = np.linspace(-4 * ref_m.std(), 4 * ref_m.std(), 50)
    for k in order:
        ax[1].hist(res[k]["mean"], bins, density=True, histtype="step", color=COL[k],
                   lw=1.8, label=f"{k} (sd {res[k]['mean'].std():.3f})")
    ax[1].set_ylim(0, 1.6 * np.histogram(ref_m, bins, density=True)[0].max())
    ax[1].text(0.02, 0.95, "normalized: a spike at 0\n(off the top of the axis)",
               transform=ax[1].transAxes, va="top", fontsize=8, color=COL["normalized"])
    ax[1].set_xlabel("spatial mean of the field G"); ax[1].set_title("law of the spatial mean")
    ax[1].legend(fontsize=8)
    bins = np.linspace(0, ref_v.mean() + 5 * ref_v.std(), 50)
    for k in order:
        ax[2].hist(res[k]["var"], bins, density=True, histtype="step", color=COL[k], lw=1.8,
                   label=f"{k} ({res[k]['var'].mean():.3f}±{res[k]['var'].std():.3f})")
    ax[2].set_ylim(0, 1.6 * np.histogram(ref_v, bins, density=True)[0].max())
    ax[2].text(0.02, 0.95, "normalized: a spike at 1\n(off the top of the axis)",
               transform=ax[2].transAxes, va="top", fontsize=8, color=COL["normalized"])
    ax[2].set_xlabel("empirical variance of the field G")
    ax[2].set_title("law of the empirical variance")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"Multiscale prior test, 2D ({pc.n}x{pc.n}, ell={pc.ell}, r={kle.r}): "
                 f"pCN s={a.s}, {a.chains} chains x {a.n_iter} it., likelihood off")
    fig.tight_layout()
    fdir = io.figures_dir(EXP)
    os.makedirs(fdir, exist_ok=True)
    fig.savefig(os.path.join(fdir, "fig_ms_prior_test.png"), dpi=150)
    plt.close(fig)

    table = {k: dict(frob=res[k]["frob"], spatial_mean_sd=float(res[k]["mean"].std()),
                     field_var_mean=float(res[k]["var"].mean()),
                     field_var_sd=float(res[k]["var"].std()),
                     theta_var_max=float(res[k]["diag"][0]),
                     theta_var_min=float(res[k]["diag"][-1]),
                     worst_psrf=psrf.get(k))
             for k in order}
    out = dict(settings=vars(a), multiscale=msum, results=table)
    tdir = io.tables_dir(EXP)
    os.makedirs(tdir, exist_ok=True)
    io.save_json(os.path.join(tdir, "ms_prior_test.json"), out)
    with open(os.path.join(tdir, "ms_prior_test.md"), "w") as f:
        f.write("| innovation | cov(θ) error (Frob./√r) | θ variances (max / min) | "
                "sd of spatial mean | empirical variance (mean ± sd) |\n|---|---|---|---|---|\n")
        for k in order:
            t = table[k]
            f.write(f"| {k} | {t['frob']:.3f} | {t['theta_var_max']:.2f} / "
                    f"{t['theta_var_min']:.2f} | {t['spatial_mean_sd']:.3f} | "
                    f"{t['field_var_mean']:.3f} ± {t['field_var_sd']:.3f} |\n")
    print(open(os.path.join(tdir, "ms_prior_test.md")).read())


if __name__ == "__main__":
    main()
