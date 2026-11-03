"""
Sigma_ms rank test for the local-KLE (MSM) experiment
(docs/msm_local_kle_experiment.md, Section 6, step 2).  Runs BEFORE any sampling.
No PDE solves.

What it does, exactly:
  1. Builds the global KLE (32 x 32 cells, squared-exponential kernel, ell = 0.1,
     truncated at r = 64 modes: energy 0.966 with the tie-aware truncation of
     brho/kle.py) and checks that r = 64.
  2. Builds the PRIMAL decomposition (4 x 4 blocks of 8 x 8 cells, exactly 4
     local KLE modes per block) and the DUAL decomposition (5 x 5 blocks,
     interfaces at (i + 1/2)/4, local dimension of every block from the 95%
     local-energy rule, tied eigenvalues kept together), and for each the
     assemble-and-project matrix B (r x p), column (k, j) = projection onto the
     global KLE of the zero-extended local mode sqrt(mu_j^k) psi_j^k.
  3. Forms Sigma_ms = B B^T and its eigenvalues d_1 >= ... >= d_r (recomputed
     here from B, independently of brho/multiscale.py).
  4. Reports, for each grid: p, the block dimensions, the singular values of B,
     d_1, d_min, cond(Sigma_ms) = d_1/d_min, cond(B), the numerical rank of B,
     r_+ = #{d_i > tau d_1} with tau = 1e-10, and whether any block's last kept
     local mode is tied with the first dropped one (a cut tie group).  For the
     primal grid it also reports the whitening operator check
     ||W Sigma_ms W^T + V_0 V_0^T - I||_F / sqrt(r).
  5. PASS iff r_+ = r = 64 for BOTH grids.  On a FAIL the script prints the
     numbers and exits with status 1: nothing is changed or "fixed" (no coarse
     move, no change of local dimension) and no sampling should follow.

    python experiments/run_msm_rank_test.py            # BRHO_ROOT=runs/msm_local_kle_2026-11
Writes figures/msm_rank_test/fig_msm_rank_test.png and
tables/msm_rank_test/msm_rank_test.{json,md} under BRHO_ROOT.
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
from brho.msm_local import MSMSetup, rank_report, RANK_TOL, KLE_ENERGY_R64  # noqa: E402

EXP = "msm_rank_test"


def _fmt(x):
    return f"{x:.3e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ell", type=float, default=0.1)
    ap.add_argument("--kle-energy", type=float, default=KLE_ENERGY_R64)
    ap.add_argument("--r", type=int, default=64, help="required KLE dimension")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--primal-local-dim", type=int, default=4)
    ap.add_argument("--dual-energy", type=float, default=0.95)
    ap.add_argument("--tol", type=float, default=RANK_TOL)
    a = ap.parse_args(argv)

    S = MSMSetup(a.n, a.ell, "sqexp", a.kle_energy, a.r, a.blocks,
                 a.primal_local_dim, a.dual_energy)
    kle = S.kle
    lam = kle.lam_all
    print(f"Global KLE: {a.n}x{a.n} cells, sqexp ell={a.ell}, r = {kle.r} "
          f"(energy {kle.energy:.5f}); lambda_63 = {lam[62]:.6f}, lambda_64 = {lam[63]:.6f}, "
          f"lambda_65 = {lam[64]:.6f}")
    reps = {"primal": rank_report(S.primal, a.tol), "dual": rank_report(S.dual, a.tol)}
    for name, rep in reps.items():
        print(f"\n== {name} grid: {rep['grid_blocks']} blocks, p = {rep['p']}  "
              f"(B is {rep['B_shape'][0]} x {rep['B_shape'][1]})")
        print(f"   local dimensions r_k = {rep['r_k']}")
        print(f"   blocks whose last kept mode is tied with the first dropped one: "
              f"{rep['tie_cut_blocks']}")
        print(f"   singular values of B: max {_fmt(rep['sv_max'])}, min {_fmt(rep['sv_min'])}, "
              f"cond(B) = {_fmt(rep['cond_B'])}, rank {rep['matrix_rank_B']} "
              f"(singular values > sqrt(tau) sigma_1)")
        print(f"   eigenvalues of Sigma_ms: d_1 = {_fmt(rep['d1'])}, d_min = {_fmt(rep['d_min'])}, "
              f"cond = {_fmt(rep['cond_Sigma'])}, trace/r = {rep['trace_over_r']:.4f}")
        print(f"   r_+ = #{{d_i > {a.tol:g} d_1}} = {rep['r_plus']}  (r = {rep['r']})  "
              f"->  {'PASS' if rep['passed'] else 'FAIL'}")
        print(f"   whitening operator check ||W S W' + V0 V0' - I||_F/sqrt(r) = "
              f"{_fmt(rep['operator_check'])}")
    ok = all(r["passed"] for r in reps.values())

    # ---------------------------------------------------------------- outputs
    fdir, tdir = io.figures_dir(EXP), io.tables_dir(EXP)
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    col = {"primal": "C0", "dual": "C3"}
    for name, rep in reps.items():
        sv, d = rep["singular_values_B"], rep["eig_Sigma"]
        ax[0].semilogy(np.arange(1, len(sv) + 1), sv, "o-", ms=3, color=col[name],
                       label=f"{name} (p = {rep['p']})")
        ax[1].semilogy(np.arange(1, len(d) + 1), np.maximum(d, 1e-300), "o-", ms=3,
                       color=col[name], label=f"{name}: r_+ = {rep['r_plus']}")
        ax[1].axhline(a.tol * rep["d1"], color=col[name], ls=":", lw=1)
    ax[0].set_xlabel("index"); ax[0].set_ylabel("singular value of B")
    ax[0].set_title("singular values of B (r x p)")
    ax[1].set_xlabel("index"); ax[1].set_ylabel(r"eigenvalue of $\Sigma_{ms}$")
    ax[1].set_title(rf"eigenvalues of $\Sigma_{{ms}}$ (dotted: $\tau d_1$, $\tau$ = {a.tol:g})")
    for x in ax:
        x.legend()
    fig.suptitle(f"Sigma_ms rank test: r = {kle.r}, 4 x 4 primal (4 modes/block) and 5 x 5 dual "
                 f"(95% energy): {'PASS' if ok else 'FAIL'}")
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "fig_msm_rank_test.png"), dpi=150)
    plt.close(fig)

    out = dict(settings=vars(a), kle=dict(r=kle.r, energy=kle.energy, lam_63_64_65=lam[62:65]),
               passed=ok, **reps)
    io.save_json(os.path.join(tdir, "msm_rank_test.json"), out)
    with open(os.path.join(tdir, "msm_rank_test.md"), "w") as f:
        f.write("| grid | blocks | p | r_+ (tau = %g) | d_1 | d_min | cond(Sigma_ms) | "
                "cond(B) | cut tie groups | result |\n|---|---|---|---|---|---|---|---|---|---|\n"
                % a.tol)
        for name, rep in reps.items():
            f.write(f"| {name} | {rep['grid_blocks']} | {rep['p']} | {rep['r_plus']} / {rep['r']} | "
                    f"{_fmt(rep['d1'])} | {_fmt(rep['d_min'])} | {_fmt(rep['cond_Sigma'])} | "
                    f"{_fmt(rep['cond_B'])} | {rep['tie_cut_blocks']} | "
                    f"{'PASS' if rep['passed'] else 'FAIL'} |\n")
        f.write(f"\nPrimal block dimensions: {reps['primal']['r_k']}\n\n"
                f"Dual block dimensions (row-major over the 5 x 5 tiling): {reps['dual']['r_k']}\n\n"
                f"Whitening operator check (primal): {_fmt(reps['primal']['operator_check'])}\n")
    print("\n" + open(os.path.join(tdir, "msm_rank_test.md")).read())
    if not ok:
        print("RANK TEST FAILED: Sigma_ms is rank-deficient for at least one grid. "
              "Nothing was changed; do not start sampling.")
        return 1
    print("RANK TEST PASSED for both grids.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
