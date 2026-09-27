"""
Forward-solve timing by grid size, CSR-fill backend and linear solver.
Run it on the machine that will do the sampling (numba compiles on first
call; the warm-up solve is excluded from the timings).

    python benchmarks/bench_solver.py                  # 2D 16..200, 3D 10..30
    python benchmarks/bench_solver.py --sizes 100 --dims 2

Prints one line per (shape, backend, solver): ms for the CSR fill, for the
linear solve, and in total per forward solve, plus the speed-up relative to
the July configuration (NumPy fill + spsolve).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import fv  # noqa: E402


def timeit(fn, reps):
    fn()                                           # warm-up (numba compile)
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*")
    ap.add_argument("--dims", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--budget", type=float, default=1.0,
                    help="seconds per timing (sets repetitions)")
    a = ap.parse_args()
    shapes = []
    if 2 in a.dims:
        shapes += [(n, n) for n in (a.sizes or [16, 50, 100, 200])]
    if 3 in a.dims:
        shapes += [(n, n, n) for n in (a.sizes or [10, 20, 30])]
    backends = ["numpy"] + (["numba"] if fv.HAVE_NUMBA else [])
    print(f"numba installed: {fv.HAVE_NUMBA}")
    print(f"{'shape':>14} {'N':>7} {'backend':>7} {'solver':>8} "
          f"{'fill ms':>9} {'solve ms':>9} {'total ms':>9} {'speed-up':>8}")
    rng = np.random.default_rng(0)
    for shape in shapes:
        N = int(np.prod(shape))
        K = np.exp(1.5 * rng.standard_normal(N))
        base = None
        for backend in backends:
            for solver in ("spsolve", "splu", "cg"):
                ph = fv.Permeameter(shape, solver=solver, backend=backend)
                t1 = ph.solve(K) is not None and timeit(lambda: ph.csr_data(K), 1)
                reps = max(3, int(a.budget / max(t1, 1e-6)))
                t_fill = timeit(lambda: ph.csr_data(K), min(reps, 2000))
                t_tot = timeit(lambda: ph.solve(K), max(3, int(a.budget / max(ph_time(ph, K), 1e-6))))
                if base is None:
                    base = t_tot
                print(f"{str(shape):>14} {N:>7} {backend:>7} {solver:>8} "
                      f"{t_fill * 1e3:9.3f} {(t_tot - t_fill) * 1e3:9.2f} "
                      f"{t_tot * 1e3:9.2f} {base / t_tot:7.2f}x")


def ph_time(ph, K):
    t = time.perf_counter(); ph.solve(K)
    return time.perf_counter() - t


if __name__ == "__main__":
    main()
