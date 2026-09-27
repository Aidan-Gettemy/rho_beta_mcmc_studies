"""
Cross-check the Python Nystrom KLE against KLE_matlab.

Uses the configuration of the MATLAB example (buildKLE_matrix): squared
exponential, sigma2 = 1, ell = 0.022, 80 Gauss-Legendre nodes per direction
on [0,1]^2, 95% energy.  Prints the leading eigenvalues, the mode count and
the energy, and writes all kept eigenvalues to a CSV.

    python benchmarks/kle_compare_matlab.py [--ell 0.022 --nq 80 --full]

In MATLAB, from the KLE_matlab folder:

    Q = kle.quadrature(); B = kle.buildCovariance(); S = kle.solve();
    [xs, ys, ws] = Q.gauss_legendre_2d(80, [0 1], [0 1]);
    ko = kle.kernelOptions(1, struct('sigma2',1,'lx',0.022,'ly',0.022));
    C = B.matrix2d(xs, ys, ko.cov_fun, ko.kw);
    [lam, phi, M, energy] = S.solveKLE(ws, C, [], 0.95, false);
    writematrix(lam, 'lam_matlab.csv');

The eigenvalues are the same quantity in both codes (weighted problem
A = W^1/2 C W^1/2, total = sum of weights = 1) and should agree to
~1e-14 relative.  The mode count can differ by one or two: this code keeps
modes tied with the last kept one (e.g. the (i, j) / (j, i) pair on a
square), MATLAB's find(cumulative >= 0.95, 1) may cut between them.
Eigenfunctions agree up to sign and, inside tied pairs, rotation; compare
covariances (T T') rather than individual modes.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import fv                          # noqa: E402
from brho.kle import build_kle               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ell", type=float, default=0.022)
    ap.add_argument("--nq", type=int, default=80)
    ap.add_argument("--energy", type=float, default=0.95)
    ap.add_argument("--out", type=int, default=140, help="output cells per side")
    ap.add_argument("--full", action="store_true",
                    help="also run the full (non-separable) Nystrom solver")
    ap.add_argument("--csv", default="lam_python.csv")
    a = ap.parse_args()
    ph = fv.Permeameter((a.out, a.out))
    prm = dict(ell=a.ell, sigma2=1.0)
    t = time.perf_counter()
    k = build_kle(ph.points, ph.axes, "sqexp", prm, a.energy, "nystrom_sep", nq=a.nq)
    print(f"separable Nystrom: r={k.r}  energy={k.energy:.6f}  total={k.total:.15f}  "
          f"({time.perf_counter() - t:.2f} s)")
    print(f"quadrature check: {k.meta['nq_check']}")
    print("leading eigenvalues:")
    for i in range(min(20, k.r)):
        print(f"  {i + 1:3d}  {k.lam[i]:.15e}")
    np.savetxt(a.csv, k.lam, fmt="%.17e")
    print(f"wrote {k.r} eigenvalues to {a.csv}")
    if a.full:
        t = time.perf_counter()
        f = build_kle(ph.points, ph.axes, "sqexp", prm, a.energy, "nystrom", nq=a.nq)
        print(f"full Nystrom: r={f.r}  ({time.perf_counter() - t:.1f} s)  "
              f"max |lam_full - lam_sep| = {np.abs(f.lam - k.lam).max():.1e}")


if __name__ == "__main__":
    main()
