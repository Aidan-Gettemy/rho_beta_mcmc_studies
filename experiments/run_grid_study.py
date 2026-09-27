"""
E2 - multiple-MCMC identifiability study (the 5x5 (beta, rho) picture).

For every pair on the grid, run one pCN chain in theta-space with (beta, rho)
held fixed and record the posterior-mean fit.  Reproduces Figs. 2-5 of the
non-uniqueness report.

    python experiments/run_grid_study.py                    # standardized truth
    python experiments/run_grid_study.py --truth-mode raw   # July data exactly
    python experiments/run_grid_study.py --quick            # smoke test

Output: results/e2_grid/{grid.npz, truth.npz, manifest.json}
"""
import os
import time
from multiprocessing import Pool

import numpy as np

from _common import parser, configs, C
from brho import io
from brho.problem import Problem
from brho.samplers import pcn_fixed_pair

_PROB = None


def _init(pc):
    global _PROB
    _PROB = Problem(pc)


def _job(args):
    beta, rho, seed, gcfg = args
    t0 = time.time()
    out = pcn_fixed_pair(_PROB, beta, rho, gcfg, seed)
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = parser(__doc__, C.GridStudyConfig)
    args = ap.parse_args()
    pc, gcfg = configs(args, C.GridStudyConfig)
    prob = Problem(pc)
    out_dir = io.results_dir(io.EXP_GRID)
    prob.syn.save(os.path.join(out_dir, "truth.npz"))

    pairs = [(b, r) for b in gcfg.betas for r in gcfg.rhos]
    jobs = [(b, r, gcfg.seed0 + 17 * k, gcfg) for k, (b, r) in enumerate(pairs)]
    print(f"E2: {len(jobs)} chains x {gcfg.n_iter} it, r={prob.r}, "
          f"truth={pc.truth_mode}")
    t0 = time.time()
    with Pool(min(gcfg.n_workers, os.cpu_count() or 1),
              initializer=_init, initargs=(pc,)) as pool:
        res = pool.map(_job, jobs)

    stack = lambda k: np.array([q[k] for q in res])
    np.savez_compressed(
        os.path.join(out_dir, "grid.npz"),
        betas=np.array(gcfg.betas), rhos=np.array(gcfg.rhos),
        beta=stack("beta"), rho=stack("rho"), rms=stack("rms"),
        acc=stack("acc"), best_phi=stack("best_phi"),
        mean_phi=stack("mean_phi"), s_final=stack("s_final"),
        theta_norm2_mean=stack("theta_norm2_mean"),
        traces=stack("trace"), mean_pobs=stack("mean_pobs"),
        mean_G=stack("mean_G"), mean_theta=stack("mean_theta"),
        seconds=stack("seconds"), r=prob.r)
    io.write_manifest(io.EXP_GRID, C.to_dict(pc, gcfg),
                      dict(minutes=(time.time() - t0) / 60, r=prob.r,
                           truth=prob.syn.summary()))
    print(f"done in {(time.time() - t0) / 60:.1f} min; "
          f"RMS/sigma range {res and min(stack('rms')) / prob.syn.sigma_p:.2f}"
          f"-{max(stack('rms')) / prob.syn.sigma_p:.2f}")


if __name__ == "__main__":
    main()
