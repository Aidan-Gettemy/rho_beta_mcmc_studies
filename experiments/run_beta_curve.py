"""
E3 - offline Monte Carlo construction of beta_bar(rho) ("global" sampling:
realizations from the truncated global KLE, same r modes as the sampler).

Only beta = 1 solves (scaling lemma); stores keff1[a, i] so any statistic
(ratio_of_mean / mean_of_ratio / median) and any K* can be applied later.
Also runs the full-rank Cholesky sweep with the same seed for the
KLE-vs-Cholesky consistency figure unless --no-cholesky.

    python experiments/run_beta_curve.py [--quick] [--n-real 256] [--n-rho 40]

Output: results/e3_curve/{curve_kle.npz, curve_cholesky.npz, truth.npz,
        manifest.json}
"""
import os
import time
import _common  # noqa: F401  (first: pins threads before NumPy loads)
from dataclasses import replace

from _common import parser, configs, C
from brho import io
from brho.problem import Problem
from brho.curve import sweep, BetaCurve


def main():
    ap = parser(__doc__, C.CurveConfig)
    ap.add_argument("--no-cholesky", action="store_true")
    args = ap.parse_args()
    pc, ccfg = configs(args, C.CurveConfig)
    prob = Problem(pc)
    out_dir = io.results_dir(io.EXP_CURVE)
    prob.freeze(out_dir)
    t0 = time.time()

    kinds = ["kle"] if args.no_cholesky else ["kle", "cholesky"]
    for kind in kinds:
        cfg = replace(ccfg, field=kind)
        print(f"E3 sweep [{kind}]: {cfg.n_rho} rho x {cfg.n_real} realizations")
        rhos, keff1, G = sweep(
            prob, cfg,
            progress=lambda a, rho, k: print(f"  rho={rho:5.2f}  mean Keff(beta=1)="
                                             f"{k.mean():.3f}  rel.std={k.std() / k.mean():.2f}"))
        curve = BetaCurve(rhos, keff1, cfg.statistic)
        curve.save(os.path.join(out_dir, f"curve_{kind}.npz"),
                   Gbar=G.mean(1), Gvar=G.var(1), field=kind)

    io.write_manifest(io.EXP_CURVE, C.to_dict(pc, ccfg),
                      dict(minutes=(time.time() - t0) / 60, r=prob.r,
                           Kstar=prob.syn.keff_obs, truth=prob.syn.summary()))
    c = BetaCurve.load(os.path.join(out_dir, "curve_kle.npz"))
    print(f"K* = {prob.syn.keff_obs:.3f};  beta_bar(rho_true={pc.rho_true}) = "
          f"{c.beta_bar(pc.rho_true, prob.syn.keff_obs):.3f}  "
          f"(beta_true = {pc.beta_true})")


if __name__ == "__main__":
    main()
