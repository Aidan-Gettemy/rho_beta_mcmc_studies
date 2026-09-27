"""
E4 - sample (theta, rho) with beta eliminated by the E3 curve, using the
3-step sampler (A: pCN theta, B: pCN w, C: ridge move), 4 chains from prior
draws, checkpointed and resumable (re-run the same command to continue).

    python experiments/run_curve_sampler.py                # production
    python experiments/run_curve_sampler.py --prior-test 1 # likelihood off
    python experiments/run_curve_sampler.py --quick
    python experiments/run_curve_sampler.py --fresh        # delete checkpoints

Requires results/e3_curve/curve_<field>.npz (run_beta_curve.py first).
Output: results/e4_sampler/  (or results/e4_prior_test/ with --prior-test 1)
        chain{k}.npz checkpoints, truth.npz, curve.npz (copy), manifest.json
"""
import glob
import os
import shutil
import time
from multiprocessing import Pool

from _common import parser, configs, C
from brho import io
from brho.problem import Problem
from brho.curve import BetaCurve
from brho.samplers import ThreeStepSampler, run_chunk


def _run_chain(args):
    pc, scfg, curve_path, ckpt, cid = args
    prob = Problem(pc)
    sampler = ThreeStepSampler(prob, BetaCurve.load(curve_path), scfg)
    t0 = time.time()
    it = 0
    while it < scfg.n_total:
        it = run_chunk(sampler, cid, ckpt)
        print(f"  chain {cid}: {it}/{scfg.n_total}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return cid, it


def check_curve_matches(pc):
    man = io.read_manifest(io.EXP_CURVE)["configs"]["ProblemConfig"]
    for k in ("n", "ell", "energy", "q_in"):
        if man[k] != getattr(pc, k):
            raise SystemExit(f"curve was built with {k}={man[k]}, "
                             f"sampler uses {getattr(pc, k)}: rebuild E3")
    if man["truth_mode"] != pc.truth_mode:
        print("note: curve built under a different truth_mode; the curve "
              "itself is truth-independent (K* is applied at run time).")


def main():
    ap = parser(__doc__, C.SamplerConfig)
    ap.add_argument("--curve-field", default="kle", choices=["kle", "cholesky"])
    ap.add_argument("--curve-statistic", default=None,
                    help="override the statistic stored with the curve")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    pc, scfg = configs(args, C.SamplerConfig)
    exp = io.EXP_PRIOR if scfg.prior_test else io.EXP_SAMPLER
    out_dir = io.results_dir(exp)
    if args.fresh:
        for f in glob.glob(os.path.join(out_dir, "chain*.npz")):
            os.remove(f)

    check_curve_matches(pc)
    src = os.path.join(io.results_dir(io.EXP_CURVE), f"curve_{args.curve_field}.npz")
    curve = BetaCurve.load(src, args.curve_statistic)
    curve_path = os.path.join(out_dir, "curve.npz")
    if src != curve_path:
        shutil.copy(src, curve_path)
    if args.curve_statistic:
        curve.save(curve_path)

    prob = Problem(pc)
    prob.syn.save(os.path.join(out_dir, "truth.npz"))
    io.write_manifest(exp, C.to_dict(pc, scfg),
                      dict(r=prob.r, Kstar=prob.syn.keff_obs,
                           curve=dict(source=src, statistic=curve.statistic,
                                      rho_max=curve.rho_max),
                           truth=prob.syn.summary(), status="running"))
    print(f"E4{' PRIOR TEST' if scfg.prior_test else ''}: {scfg.n_chains} chains "
          f"x {scfg.n_total} it (burn {scfg.n_burn}, thin {scfg.thin}); "
          f"r={prob.r}; K*={prob.syn.keff_obs:.3f}; curve={curve.statistic}")
    t0 = time.time()
    jobs = [(pc, scfg, curve_path, os.path.join(out_dir, f"chain{c}.npz"), c)
            for c in range(scfg.n_chains)]
    with Pool(min(scfg.n_workers, scfg.n_chains)) as pool:
        pool.map(_run_chain, jobs)
    man = io.read_manifest(exp)
    man.update(status="complete", minutes=(time.time() - t0) / 60)
    io.save_json(os.path.join(out_dir, "manifest.json"), man)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
