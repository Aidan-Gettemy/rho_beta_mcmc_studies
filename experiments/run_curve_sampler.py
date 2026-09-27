"""
E4 - sample (theta, rho) with beta eliminated by the E3 curve, using the
3-step sampler (A: pCN theta, B: pCN w, C: ridge move).  Chains start from
prior draws, are checkpointed every `chunk` iterations and resume when the
same command is rerun.

Laptop / single node (setup + all chains in a process pool):
    python experiments/run_curve_sampler.py                # production
    python experiments/run_curve_sampler.py --prior-test 1 # likelihood off
    python experiments/run_curve_sampler.py --quick
    python experiments/run_curve_sampler.py --fresh        # new run
Cluster (one chain per array task; see cluster/):
    python experiments/run_curve_sampler.py --setup-only --n-chains 64 [...]
    python experiments/run_curve_sampler.py --chain-id env # in each task
    python experiments/run_curve_sampler.py --status

Requires results/e3_curve/curve_<field>.npz (run_beta_curve.py first).
Output: results/e4_sampler/  (or results/e4_prior_test/ with --prior-test 1)
        chainK.npz checkpoints, truth.npz, curve.npz (copy), manifest.json
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import glob
import os
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np

from _common import parser, configs, configs_from_manifest, task_indices, C
from brho import io
from brho.problem import Problem
from brho.curve import BetaCurve
from brho.kle import KLE
from _common import kle_summary
from brho.samplers import ThreeStepSampler, run_chunk


def exp_name(prior_test):
    return io.EXP_PRIOR if prior_test else io.EXP_SAMPLER


def chain_path(exp, cid):
    return os.path.join(io.results_dir(exp), f"chain{cid}.npz")


def run_chain(pc, scfg, exp, cid):
    prob = Problem(pc, frozen=io.results_dir(exp))
    curve = BetaCurve.load(os.path.join(io.results_dir(exp), "curve.npz"))
    sampler = ThreeStepSampler(prob, curve, scfg)
    t0 = time.time()
    it = 0
    while it < scfg.n_total:
        it = run_chunk(sampler, cid, chain_path(exp, cid))
        print(f"  chain {cid}: {it}/{scfg.n_total}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return cid, it


def _pool_job(args):
    return run_chain(*args)


def check_curve_matches(prob):
    """The curve must be built from the same KLE basis the sampler uses
    (multiscale report, consistency requirement 1)."""
    man = io.read_manifest(io.EXP_CURVE)
    e3_kle = os.path.join(io.results_dir(io.EXP_CURVE), "kle.npz")
    if man.get("kle_key") is not None:
        if man["kle_key"] != prob.kle_key():
            raise SystemExit(f"E3 curve was built with KLE {man['kle_key']}, "
                             f"the sampler asks for {prob.kle_key()}: rebuild E3")
        if not np.array_equal(KLE.load(e3_kle).T, prob.kle.T):
            raise SystemExit("E3 and E4 KLE bases differ numerically (built on "
                             "different stores/machines): rebuild E3 in this study")
    else:                                      # E3 run from before the store
        pe3 = man["configs"]["ProblemConfig"]
        for k in ("n", "ell", "energy", "q_in"):
            if pe3[k] != getattr(prob.pc, k):
                raise SystemExit(f"curve was built with {k}={pe3[k]}, "
                                 f"sampler uses {getattr(prob.pc, k)}: rebuild E3")
    if man["configs"]["ProblemConfig"]["truth_mode"] != prob.pc.truth_mode:
        print("note: curve built under a different truth_mode; the curve "
              "itself is truth-independent (K* is applied at run time).")


def setup(pc, scfg, curve_field, curve_statistic, exp):
    out_dir = io.results_dir(exp)
    prob = Problem(pc, store=io.store_dir())      # build once per study, else load
    check_curve_matches(prob)                     # validate BEFORE touching the run
    src = os.path.join(io.results_dir(io.EXP_CURVE), f"curve_{curve_field}.npz")
    curve = BetaCurve.load(src, curve_statistic)
    for f in glob.glob(os.path.join(out_dir, "chain*.npz")):
        os.remove(f)                              # a new setup = a new run
    curve.save(os.path.join(out_dir, "curve.npz"))   # frozen copy for this run
    prob.freeze(out_dir)                          # one KLE basis + data per run
    prob.phys.warmup()                            # compile numba once, before ranks start
    io.write_manifest(exp, C.to_dict(pc, scfg),
                      dict(r=prob.r, Kstar=prob.syn.keff_obs,
                           kle_key=prob.kle_key(), kle=kle_summary(prob),
                           curve=dict(source=src, statistic=curve.statistic,
                                      rho_max=curve.rho_max),
                           truth=prob.syn.summary(), status="set up"))
    print(f"E4{' PRIOR TEST' if scfg.prior_test else ''}: {scfg.n_chains} chains "
          f"x {scfg.n_total} it (burn {scfg.n_burn}, thin {scfg.thin}); "
          f"r={prob.r}; K*={prob.syn.keff_obs:.3f}; curve={curve.statistic}; "
          f"seeds={scfg.seed_scheme}")


def status(exp):
    _, scfg = configs_from_manifest(exp, C.SamplerConfig)
    done = 0
    for cid in range(scfg.n_chains):
        p = chain_path(exp, cid)
        it = int(np.load(p)["it"]) if os.path.exists(p) else 0
        done += it >= scfg.n_total
        print(f"chain {cid:3d}: {it:8d}/{scfg.n_total}")
    print(f"{done}/{scfg.n_chains} chains complete")
    return done == scfg.n_chains


def main():
    ap = parser(__doc__, C.SamplerConfig)
    ap.add_argument("--curve-field", default="kle", choices=["kle", "cholesky"])
    ap.add_argument("--curve-statistic", default=None,
                    help="override the statistic stored with the curve")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--setup-only", action="store_true")
    mode.add_argument("--chain-id", help="int, or 'env' for the array index")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--warmup-only", action="store_true",
                      help="compile numba for an existing run (resumed node jobs)")
    ap.add_argument("--fresh", action="store_true",
                    help="(pool mode) discard existing chains and start over")
    args = ap.parse_args()
    prior_test = bool(args.prior_test) if args.prior_test is not None else False
    exp = exp_name(prior_test)

    if args.status:
        sys.exit(0 if status(exp) else 1)
    if args.warmup_only:
        pc, _ = configs_from_manifest(exp, C.SamplerConfig)
        Problem(pc, frozen=io.results_dir(exp)).phys.warmup()
        print("numba warm-up done")
        return
    if args.chain_id is not None:
        pc, scfg = configs_from_manifest(exp, C.SamplerConfig)
        for cid in task_indices(args.chain_id, scfg.n_chains):
            run_chain(pc, scfg, exp, cid)
        return

    pc, scfg = configs(args, C.SamplerConfig)
    have_run = os.path.exists(os.path.join(io.results_dir(exp), "manifest.json"))
    if args.setup_only or args.fresh or not have_run:
        setup(pc, scfg, args.curve_field, args.curve_statistic, exp)
    else:                                   # resume: settings must not change
        rec = configs_from_manifest(exp, C.SamplerConfig)
        if C.to_dict(*rec) != C.to_dict(pc, scfg):
            a, b = C.to_dict(*rec), C.to_dict(pc, scfg)
            diff = {f"{c}.{k}": (a[c][k], b[c][k]) for c in a for k in a[c]
                    if a[c][k] != b[c].get(k)}
            raise SystemExit(f"an {exp} run with different settings exists "
                             f"(recorded, requested): {diff}\n"
                             "rerun with --fresh to start a new run, or drop "
                             "the flags to resume the existing one")
        print(f"resuming {exp} (use --fresh for a new run)")
    if args.setup_only:
        print(f"set up; now run --chain-id env over {scfg.n_chains} chains "
              f"(array 0-{scfg.n_chains - 1}, or srun/mpirun ranks)")
        return
    t0 = time.time()
    with Pool(min(scfg.n_workers, scfg.n_chains)) as pool:
        pool.map(_pool_job, [(pc, scfg, exp, c) for c in range(scfg.n_chains)])
    man = io.read_manifest(exp)
    man.update(status="complete", minutes=(time.time() - t0) / 60)
    io.save_json(os.path.join(io.results_dir(exp), "manifest.json"), man)
    print(f"done in {man['minutes']:.1f} min -> {io.results_dir(exp)}")


if __name__ == "__main__":
    main()
