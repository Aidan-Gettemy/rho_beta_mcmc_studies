"""
E2 - multiple-MCMC identifiability study (the 5x5 (beta, rho) picture).

For every pair on the grid, run one pCN chain in theta-space with (beta, rho)
held fixed and record the posterior-mean fit.  Reproduces Figs. 2-5 of the
non-uniqueness report.

Laptop / single node (setup + all pairs in a process pool + collect):
    python experiments/run_grid_study.py                    # standardized truth
    python experiments/run_grid_study.py --truth-mode raw --seed-scheme legacy
                                                            # July data + seeds
    python experiments/run_grid_study.py --quick            # smoke test
Cluster (one pair per array task; see cluster/):
    python experiments/run_grid_study.py --setup-only [config flags]
    python experiments/run_grid_study.py --pair-index env   # in each task
    python experiments/run_grid_study.py --collect

Output: results/e2_grid/{manifest.json, truth.npz, pairs/pair_KKK.npz, grid.npz}
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import glob
import os
import time
from multiprocessing import Pool

import numpy as np

from _common import parser, configs, configs_from_manifest, task_index, C
from brho import io
from brho.problem import Problem
from brho.samplers import pcn_fixed_pair

EXP = io.EXP_GRID
KEYS = ("beta", "rho", "rms", "acc", "best_phi", "mean_phi", "s_final",
        "theta_norm2_mean", "trace", "mean_pobs", "mean_G", "mean_theta",
        "seconds")


def pairs_of(gcfg):
    return [(b, r) for b in gcfg.betas for r in gcfg.rhos]


def pair_path(k):
    d = os.path.join(io.results_dir(EXP), "pairs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"pair_{k:03d}.npz")


def setup(pc, gcfg):
    prob = Problem(pc)
    prob.freeze(io.results_dir(EXP))             # one KLE basis + data per run
    io.write_manifest(EXP, C.to_dict(pc, gcfg),
                      dict(r=prob.r, n_pairs=len(pairs_of(gcfg)),
                           truth=prob.syn.summary(), status="set up"))
    return prob


def run_pair(prob, gcfg, k, force=False):
    path = pair_path(k)
    if os.path.exists(path) and not force:
        print(f"pair {k}: already done")
        return
    beta, rho = pairs_of(gcfg)[k]
    t0 = time.time()
    out = pcn_fixed_pair(prob, beta, rho, gcfg,
                         C.chain_seed(gcfg.seed0, k, gcfg.seed_scheme, 17))
    out["seconds"] = time.time() - t0
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **{key: np.asarray(out[key]) for key in KEYS})
    os.replace(tmp, path)
    print(f"pair {k}: beta={beta} rho={rho} RMS/sigma="
          f"{out['rms'] / prob.syn.sigma_p:.2f} acc={out['acc']:.2f} "
          f"({out['seconds']:.0f} s)", flush=True)


def collect(gcfg):
    n = len(pairs_of(gcfg))
    missing = [k for k in range(n) if not os.path.exists(pair_path(k))]
    if missing:
        raise SystemExit(f"missing pairs {missing}: rerun those array tasks")
    zs = [np.load(pair_path(k)) for k in range(n)]
    st = lambda key: np.array([z[key] for z in zs])
    r = int(zs[0]["mean_theta"].shape[0])
    np.savez_compressed(
        os.path.join(io.results_dir(EXP), "grid.npz"),
        betas=np.array(gcfg.betas), rhos=np.array(gcfg.rhos),
        beta=st("beta"), rho=st("rho"), rms=st("rms"), acc=st("acc"),
        best_phi=st("best_phi"), mean_phi=st("mean_phi"),
        s_final=st("s_final"), theta_norm2_mean=st("theta_norm2_mean"),
        traces=st("trace"), mean_pobs=st("mean_pobs"), mean_G=st("mean_G"),
        mean_theta=st("mean_theta"), seconds=st("seconds"), r=r)
    man = io.read_manifest(EXP)
    man.update(status="complete",
               cpu_minutes=float(st("seconds").sum() / 60))
    io.save_json(os.path.join(io.results_dir(EXP), "manifest.json"), man)
    print(f"collected {n} pairs -> grid.npz")


_PROB = None


def _init(pc):
    global _PROB
    _PROB = Problem(pc, frozen=io.results_dir(EXP))


def _job(args):
    gcfg, k = args
    run_pair(_PROB, gcfg, k, force=True)


def main():
    ap = parser(__doc__, C.GridStudyConfig)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--setup-only", action="store_true")
    mode.add_argument("--pair-index", help="int, or 'env' for the array index")
    mode.add_argument("--collect", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun a finished pair")
    args = ap.parse_args()

    if args.pair_index is not None:
        pc, gcfg = configs_from_manifest(EXP, C.GridStudyConfig)
        run_pair(Problem(pc, frozen=io.results_dir(EXP)), gcfg,
                 task_index(args.pair_index), args.force)
        return
    if args.collect:
        _, gcfg = configs_from_manifest(EXP, C.GridStudyConfig)
        collect(gcfg)
        return

    pc, gcfg = configs(args, C.GridStudyConfig)
    for f in glob.glob(os.path.join(io.results_dir(EXP), "pairs", "*.npz")):
        os.remove(f)                                  # a new setup = a new run
    prob = setup(pc, gcfg)
    n = len(pairs_of(gcfg))
    print(f"E2: {n} chains x {gcfg.n_iter} it, r={prob.r}, grid={pc.n}x{pc.n}, "
          f"truth={pc.truth_mode}, seeds={gcfg.seed_scheme}")
    if args.setup_only:
        print(f"set up; submit array tasks 0-{n - 1} with --pair-index env")
        return
    t0 = time.time()
    with Pool(min(gcfg.n_workers, os.cpu_count() or 1),
              initializer=_init, initargs=(pc,)) as pool:
        pool.map(_job, [(gcfg, k) for k in range(n)])
    collect(gcfg)
    man = io.read_manifest(EXP)
    man["minutes"] = (time.time() - t0) / 60
    io.save_json(os.path.join(io.results_dir(EXP), "manifest.json"), man)
    print(f"done in {man['minutes']:.1f} min")


if __name__ == "__main__":
    main()
