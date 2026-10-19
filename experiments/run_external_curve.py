"""
E3 with an external Darcy solver - offline beta(rho) curve for the 3D 38x38x45 problem (Step 1).

For realization i (multiscale, whitened field YP_i, brho/multiscale_mat.py) and each rho_j:
    K = K0 * exp(rho_j * YP_i)  -> binary file (x fastest) -> external solver -> pressure drop [Pa]
    Keff = |v| mu Lz / dP        (v, mu, Lz read from the solver's case file)
One task = one realization: the field is generated once and reused for all rho (common random
numbers).  Per-realization results are saved as they finish, so a stopped job resumes.  The solver
is not part of this repository: see external/README.md for how it is located and what it must do.

  # solver acceptance test (run first, on the machine that will run the study)
  python experiments/run_external_curve.py --selfcheck --cfg external/DATA.cfg \
      --ref-field external/data/Kref.dat --ref-pdrop external/data/Syn_2mm_pDrop.dat

  # one-time conversion of the global basis (workers then share it through mmap)
  python experiments/run_external_curve.py --cache-phi external/matrices/phi_matrix_full_<tag>.mat \
      external/matrices/phi_full_f32.npy

  # the study (one 64-core node: cluster/slurm/node_e3_external.sbatch)
  python experiments/run_external_curve.py --cfg external/DATA.cfg --matrix-dir external/matrices \
      --tag <tag> --whitening external/matrices/whitening_operator_SP1.npz \
      --phi-npy external/matrices/phi_full_f32.npy --workers 64 --n-real 2048 --n-rho 40 \
      --Kexp 1.036016e-13

  # fits only, from an existing run:   ... --fit --Kexp 1.036016e-13
  # pilot (timing, spread):            ... --n-real 64 --n-rho 6   with ALLOW_FEW_RHO=1

Output: results/e3_external/{curve_external.npz, keff_parts/, fields/YP_*.npy, manifest.json}
        tables/e3_external/{fits.json, fit_table.csv, beta_pchip_table.csv}
        figures/e3_external/fit_curves.png
curve_external.npz has the layout of results/e3_curve/curve_kle.npz (rhos, keff1 = keff / K0,
statistic), so brho.curve.BetaCurve.load reads it; keff and K0 are stored as well.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import _common  # noqa: F401  (first: pins threads before NumPy loads)
import numpy as np

from brho import io
from brho.curvefit import fit_all, summary_lines, write_fit
from brho.darcy_external import DarcyRunner, physics_from_cfg, selfcheck, solver_exe

EXP = io.EXP_CURVE_EXTERNAL
_FIELD = None


def get_field(args):
    """Per-process field source (loaded once per worker)."""
    global _FIELD
    if _FIELD is None:
        from brho.multiscale_mat import load_multiscale
        _FIELD = load_multiscale(args.matrix_dir, args.tag, args.whitening, n_sub=args.n_sub, phi_npy=args.phi_npy)
    return _FIELD


def worker(i, args, rhos, out):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    res = os.path.join(out, "keff_parts", f"keff_{i:05d}.npy")
    if os.path.exists(res):
        if np.load(res, mmap_mode="r").shape != (len(rhos),):
            raise SystemExit(f"{res} does not have {len(rhos)} values: it belongs to a run with a different rho "
                             "grid. Use a new --exp-tag (or the same --n-rho/--rho-min/--rho-max).")
        return i, 0.0
    t0 = time.time()
    YP = get_field(args).sample(np.random.default_rng([args.seed, i]))
    if i < args.n_save:
        os.makedirs(os.path.join(out, "fields"), exist_ok=True)
        np.save(os.path.join(out, "fields", f"YP_{i:05d}.npy"), YP.astype(np.float32))
    runner = DarcyRunner(args.exe, args.cfg, os.path.join(out, "work", f"w{os.getpid()}"),
                         launcher=args.launcher, rtol=args.rtol, extra=args.solver_extra)
    keff = np.array([runner.keff(args.K0 * np.exp(r * YP.astype(np.float64)), tag=f"r{i}") for r in rhos])
    np.save(res, keff)
    return i, time.time() - t0


def collect(args, rhos, out):
    keff = np.stack([np.load(os.path.join(out, "keff_parts", f"keff_{i:05d}.npy"))
                     for i in range(args.n_real)], axis=1)            # (n_rho, N)
    assert keff.shape[0] == len(rhos), (f"saved parts have {keff.shape[0]} rho values but the grid has "
                                         f"{len(rhos)}: pass the --n-rho/--rho-min/--rho-max of the run (see manifest.json)")
    # Same layout as BetaCurve.save.  Written directly: BetaCurve builds its interpolators on
    # construction and refuses NaN, but a run with failed solves must still be saved and inspectable.
    np.savez_compressed(os.path.join(out, "curve_external.npz"), rhos=rhos, keff1=keff / args.K0,
                        statistic="ratio_of_mean", keff=keff, K0=args.K0)
    return keff


def do_fit(args, rhos, keff, out):
    nbad = int(np.isnan(keff).sum())
    if nbad:
        print(f"{nbad} failed solves (NaN) in {int(np.isnan(keff).any(axis=1).sum())} of {len(rhos)} rho values: "
              "fits skipped (dropping failures would bias the curve, most at large rho)")
        return
    if args.Kexp is None:
        print("no --Kexp given: fits skipped (Kexp = |v| mu Lz / dP_observed[Pa])")
        return
    res, extra = fit_all(rhos, keff, args.K0, args.Kexp, args.degrees, weighted=not args.unweighted)
    write_fit(res, extra, rhos, io.tables_dir(EXP), io.figures_dir(EXP))
    print("\n".join(summary_lines(res, rhos)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", help="solver executable (default: $SOLVER_EXE, $SOLVER_DIR, external/solver.local)")
    ap.add_argument("--cfg", help="the solver's case file")
    ap.add_argument("--launcher", default="", help='e.g. "mpirun -np 1"; empty = run directly')
    ap.add_argument("--solver-extra", default="", help="extra solver flags, e.g. '-pc_type hypre'")
    ap.add_argument("--rtol", type=float, default=1e-8, help="linear solver tolerance")
    ap.add_argument("--K0", type=float, default=1.0e-13, help="base permeability of the test fields (cancels)")
    ap.add_argument("--Kexp", type=float, help="measured effective permeability [m^2] (enables the fits)")
    ap.add_argument("--degrees", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--unweighted", action="store_true", help="polynomial fits without 1/se^2 weights")
    ap.add_argument("--rho-min", type=float, default=0.05)
    ap.add_argument("--rho-max", type=float, default=5.0)
    ap.add_argument("--n-rho", type=int, default=40)
    ap.add_argument("--n-real", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--n-save", type=int, default=6, help="save YP for the first n realizations")
    ap.add_argument("--matrix-dir")
    ap.add_argument("--tag", help="name stem of the matrices, e.g. 38x38x45_38x38x45_l2.0x2.0x2.0_m500_IND")
    ap.add_argument("--n-sub", type=int, default=12, help="number of subdomain files (KLE_matrix_<tag>_1..n)")
    ap.add_argument("--whitening", help="whitening_operator_*.npz built for the same PHI")
    ap.add_argument("--phi-npy", help="float32 .npy cache of PHI (see --cache-phi)")
    ap.add_argument("--exp-tag", default="", help="suffix for the study folder (e3_external_<tag>)")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--ref-field")
    ap.add_argument("--ref-pdrop")
    ap.add_argument("--cache-phi", nargs=2, metavar=("PHI_MAT", "PHI_NPY"))
    ap.add_argument("--fit", action="store_true", help="fits only, from the saved parts")
    args = ap.parse_args()

    if args.cache_phi:
        from brho.multiscale_mat import cache_phi
        print("saved", cache_phi(*args.cache_phi))
        return
    if not args.cfg:
        ap.error("--cfg is required")
    global EXP
    if args.exp_tag:
        EXP = f"{io.EXP_CURVE_EXTERNAL}_{args.exp_tag}"
    out = io.results_dir(EXP)
    args.exe = solver_exe(args.exe)

    if args.selfcheck:
        runner = DarcyRunner(args.exe, args.cfg, os.path.join(out, "work_selfcheck"),
                             launcher=args.launcher, rtol=args.rtol, extra=args.solver_extra)
        sys.exit(0 if selfcheck(runner, args.K0, args.ref_field, args.ref_pdrop) else 1)

    if args.n_rho < 20 and os.environ.get("ALLOW_FEW_RHO") != "1":
        sys.exit("n-rho < 20 is too few for the curve fits (40 recommended). Set ALLOW_FEW_RHO=1 for pilots.")
    rhos = np.linspace(args.rho_min, args.rho_max, args.n_rho)

    if args.fit:
        do_fit(args, rhos, collect(args, rhos, out), out)
        return
    for need in ("matrix_dir", "tag", "whitening"):
        if getattr(args, need) is None:
            ap.error(f"--{need.replace('_', '-')} is required for a run")

    phys = physics_from_cfg(args.cfg)
    os.makedirs(os.path.join(out, "keff_parts"), exist_ok=True)
    io.write_manifest(EXP, dict(args=vars(args)), dict(rhos=rhos, physics=phys))
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, i, args, rhos, out) for i in range(args.n_real)]
        for fu in as_completed(futs):
            i, dt = fu.result()
            done += 1
            if done % 8 == 0 or done == args.n_real:
                print(f"{done}/{args.n_real} realizations  ({time.time() - t0:.0f} s elapsed, last {dt:.0f} s)",
                      flush=True)
    keff = collect(args, rhos, out)
    bad = int(np.isnan(keff).sum())
    print(f"saved curve_external.npz  keff shape {keff.shape}, failed solves: {bad}")
    if bad:
        print("WARNING: some solves failed; delete the NaN parts in keff_parts/ and rerun to retry only those.")
        return
    do_fit(args, rhos, keff, out)


if __name__ == "__main__":
    main()
