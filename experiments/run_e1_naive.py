"""
E1 - naive joint (theta, beta, rho) model sampled by pCN, one chain per MPI
rank.  Demonstrates the failure: every chain fits the pressure data, but beta
and rho drift and different chains end in different places.

    --move blocked   (default, the standard) A) pCN on theta, B) pCN on
                     (xi_beta, xi_rho), separate adapted step sizes
    --move global    one pCN move on everything, one step size
Each move writes its own run folder (results/e1_blocked, results/e1_global),
so both can live in one study and report_e1.py compares them.

Cluster (see cluster/slurm/mpi_e1_naive.sbatch):
    mpirun -np 64 python experiments/run_e1_naive.py [--move global] [config flags]
    mpirun -np 64 python experiments/run_e1_naive.py --resume [--move global]
Laptop / no mpi4py:
    python experiments/run_e1_naive.py --quick --local-chains 4
Post-processing (one process):
    python reports/report_e1.py

Flow: rank 0 sets up the run (KLE basis + reference field from the study
store, frozen copies in results/e1_naive, manifest, numba compiled once) and
broadcasts success; every rank then loads the frozen problem and runs chain
<rank> (checkpointed every `checkpoint_every` iterations); summaries are
gathered on rank 0 (mean acceptance rate, results/e1_naive/chains_summary.json).
All settings come from the manifest written by rank 0.
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import glob
import os
import sys
import time
import traceback

import numpy as np

from _common import parser, configs, configs_from_manifest, C, kle_summary
from brho import io
from brho.problem import Problem
from brho.naive import run_chain

EXP = io.exp_e1("blocked")        # set from --move in main()

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
except ImportError:                           # laptop / tests: no MPI
    MPI, COMM = None, None
RANK = COMM.Get_rank() if COMM else 0
SIZE = COMM.Get_size() if COMM else 1


def log(msg):
    print(f"[rank {RANK}] {msg}", flush=True)


def chain_path(cid):
    return os.path.join(io.results_dir(EXP), f"chain{cid}.npz")


def setup(args, n_chains):
    """Rank 0 only.  Returns (ok, message)."""
    d = io.results_dir(EXP)
    existing = glob.glob(os.path.join(d, "chain*.npz"))
    if args.resume:
        if not os.path.exists(os.path.join(d, "manifest.json")):
            return False, "--resume: no E1 run in this BRHO_ROOT"
        man = io.read_manifest(EXP)
        if man["configs"]["NaiveConfig"].get("move", "global") != args.move_resolved:
            return False, "--resume: --move does not match the run in this folder"
        if man["n_chains"] != n_chains:
            return False, (f"--resume: the run has {man['n_chains']} chains, this "
                           f"launch has {n_chains}; use the same number of ranks")
        pc, _ = configs_from_manifest(EXP, C.NaiveConfig)
        Problem(pc, frozen=d).phys.warmup()           # numba cache for this job
        return True, f"resuming {len(existing)} chain files"
    if existing and not args.fresh:
        return False, (f"{len(existing)} E1 chain files exist in {d}: pass --resume "
                       "to continue them or --fresh to start a new run")
    pc, cfg = configs(args, C.NaiveConfig)
    prob = Problem(pc, store=io.store_dir(), log=log)
    for f in existing:
        os.remove(f)
    prob.freeze(d)
    prob.phys.warmup()                                # compile numba once
    io.write_manifest(EXP, C.to_dict(pc, cfg),
                      dict(n_chains=n_chains, mpi_size=SIZE, r=prob.r,
                           kle_key=prob.kle_key(), kle=kle_summary(prob),
                           likelihood=prob.likelihood_summary(),
                           truth=prob.syn.summary(), status="running"))
    lk = prob.likelihood_summary()
    lk_txt = (f"relative, sigma_rel={lk['sigma_rel']:.3g}" if "sigma_rel" in lk
              else "absolute")
    return True, (f"E1 [{cfg.move}]: {n_chains} chains x {cfg.n_iter} it (adapt {cfg.n_burn}, "
                  f"thin {cfg.thin}); r={prob.r}; truth={pc.truth_mode}; likelihood {lk_txt}; "
                  f"beta~logN({cfg.mu_logbeta},{cfg.s_logbeta}), "
                  f"rho~logN({cfg.mu_logrho},{cfg.s_logrho})")


def one_chain(cid):
    try:
        pc, cfg = configs_from_manifest(EXP, C.NaiveConfig)
        prob = Problem(pc, frozen=io.results_dir(EXP))
        return run_chain(prob, cfg, cid, chain_path(cid), log=log)
    except Exception:
        return dict(chain_id=cid, error=traceback.format_exc())


def finish(summaries, t0):
    errors = [s for s in summaries if "error" in s]
    ok = [s for s in summaries if "error" not in s]
    for s in errors:
        print(f"chain {s['chain_id']} FAILED:\n{s['error']}", flush=True)
    io.save_json(os.path.join(io.results_dir(EXP), "chains_summary.json"), summaries)
    man = io.read_manifest(EXP)
    man.update(status="complete" if not errors else f"{len(errors)} chains failed",
               minutes=(time.time() - t0) / 60)
    io.save_json(os.path.join(io.results_dir(EXP), "manifest.json"), man)
    if ok:
        print("=" * 53)
        for s in sorted(ok, key=lambda s: s["chain_id"]):
            acc_h = (f", hyper {100 * s['acc_h_post']:5.1f}%" if "acc_h" in s else "")
            print(f"chain {s['chain_id']:3d}: acc {100 * s['acc']:5.1f}% "
                  f"(post-adapt {100 * s['acc_post']:5.1f}%{acc_h})  "
                  f"steps {'/'.join(f'{x:.3g}' for x in s['s_final'])}  "
                  f"beta={s['beta_final']:.3g}  "
                  f"rho={s['rho_final']:.3g}  relerr={s['relerr_final']:.2e}")
        print(f"MEAN ACCEPTANCE RATE ........................: "
              f"{100 * np.mean([s['acc'] for s in ok]):5.2f}%")
        print("=" * 53, flush=True)
    return not errors


def main():
    ap = parser(__doc__, C.NaiveConfig)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="discard an existing E1 run in this BRHO_ROOT")
    ap.add_argument("--local-chains", type=int, default=1,
                    help="without MPI: run this many chains in a process pool")
    args = ap.parse_args()
    global EXP
    args.move_resolved = args.move or C.NaiveConfig().move
    EXP = io.exp_e1(args.move_resolved)
    t0 = time.time()
    n_chains = SIZE if COMM else args.local_chains

    ok, msg = (setup(args, n_chains) if RANK == 0 else (None, None))
    if COMM:
        ok, msg = COMM.bcast((ok, msg), root=0)
    if RANK == 0:
        print(msg, flush=True)
    if not ok:
        sys.exit(1)

    if COMM:
        summaries = COMM.gather(one_chain(RANK), root=0)
    else:
        from multiprocessing import Pool
        with Pool(min(n_chains, os.cpu_count() or 1)) as pool:
            summaries = pool.map(one_chain, range(n_chains))
    if RANK == 0:
        good = finish(summaries, t0)
        print(f"E1 done in {(time.time() - t0) / 60:.1f} min -> {io.results_dir(EXP)}")
        if COMM and not good:
            COMM.Abort(1)
        if not good:
            sys.exit(1)


if __name__ == "__main__":
    main()
