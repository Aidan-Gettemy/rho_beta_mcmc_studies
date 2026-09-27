"""
Solver checks for brho/fv.py (python tests/run_tests.py).

  reference   2D solver == the July solver of mcmc_study.py (frozen below)
  kernels     numba kernel == vectorized NumPy fill (the kernel runs as plain
              Python when numba is absent, so the logic is checked either way)
  3D          SPD matrix, homogeneous Keff exact, scaling lemma, 500 obs on 10^3
  solvers     splu (MMD_AT_PLUS_A) == spsolve
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import fv                                  # noqa: E402
from brho.physics2d import Darcy2D                   # noqa: E402


def july_solver(n, K):
    """mcmc_study.py solve_pressure, verbatim logic (reference only)."""
    N, h = n * n, 1.0 / n
    mid = lambda ix, iy: ix * n + iy
    ix1, iy1 = np.meshgrid(np.arange(n - 1), np.arange(n), indexing="ij")
    m1x, m2x = mid(ix1, iy1).ravel(), mid(ix1 + 1, iy1).ravel()
    ix2, iy2 = np.meshgrid(np.arange(n), np.arange(n - 1), indexing="ij")
    m1y, m2y = mid(ix2, iy2).ravel(), mid(ix2, iy2 + 1).ravel()
    mL, mR = mid(0, np.arange(n)), mid(n - 1, np.arange(n))
    rows = np.concatenate([m1x, m2x, m1x, m2x, m1y, m2y, m1y, m2y, mR])
    cols = np.concatenate([m1x, m2x, m2x, m1x, m1y, m2y, m2y, m1y, mR])
    kf = K.ravel()
    Tx = 2.0 * kf[m1x] * kf[m2x] / (kf[m1x] + kf[m2x])
    Ty = 2.0 * kf[m1y] * kf[m2y] / (kf[m1y] + kf[m2y])
    vals = np.concatenate([Tx, Tx, -Tx, -Tx, Ty, Ty, -Ty, -Ty, 2.0 * kf[mR]])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    b = np.zeros(N); b[mL] = h
    p = spla.spsolve(A, b)
    keff = 1.0 / (p[mL] + (h / 2) / kf[mL]).mean()
    return A, p, keff


def test_matches_july_solver():
    rng = np.random.default_rng(11)
    for n in (5, 16, 40):
        K = np.exp(1.5 * rng.standard_normal(n * n))
        A0, p0, k0 = july_solver(n, K)
        ph = Darcy2D(n)
        p = ph.solve(K)
        assert np.abs(ph.matrix(K).toarray() - A0.toarray()).max() < 1e-12
        assert np.abs(p - p0).max() / np.abs(p0).max() < 1e-12
        assert abs(ph.keff(K, p) / k0 - 1) < 1e-12


def test_kernel_equals_numpy_fill():
    rng = np.random.default_rng(12)
    for shape in [(7, 7), (16, 9), (5, 4, 3), (8, 8, 8)]:
        ph = fv.Permeameter(shape, backend="numpy")
        K = np.exp(1.5 * rng.standard_normal(ph.N))
        d_k = np.empty(ph.nnz)
        fv._fill_csr_numba(K, *ph._n3, *ph._f, d_k)   # compiled if numba present
        d_n = ph.csr_data(K)
        assert np.abs(d_k - d_n).max() <= 1e-13 * np.abs(d_n).max(), shape


def test_backends_agree_when_numba_present():
    if not fv.HAVE_NUMBA:
        print("      (numba not installed: compiled backend skipped)")
        return
    rng = np.random.default_rng(13)
    for shape in [(16, 16), (100, 100), (10, 10, 10)]:
        K = np.exp(rng.standard_normal(int(np.prod(shape))))
        a = fv.Permeameter(shape, backend="numba").solve(K)
        b = fv.Permeameter(shape, backend="numpy").solve(K)
        assert np.abs(a - b).max() / np.abs(b).max() < 1e-12, shape


def test_3d_properties():
    rng = np.random.default_rng(14)
    ph = fv.Permeameter((10, 10, 10))
    assert ph.obs_mask.sum() == 500
    K = np.exp(rng.standard_normal(ph.N))
    A = ph.matrix(K)
    assert abs(A - A.T).max() == 0
    Kh = np.full(ph.N, 2.5)
    assert abs(ph.keff(Kh, ph.solve(Kh)) / 2.5 - 1) < 1e-12
    sc = ph.scaling_check(K)
    assert sc["p_rel_err"] < 1e-12 and sc["keff_rel_err"] < 1e-12


def test_solvers_agree():
    rng = np.random.default_rng(15)
    for shape in [(30, 30), (8, 8, 8)]:
        K = np.exp(1.5 * rng.standard_normal(int(np.prod(shape))))
        ref = fv.Permeameter(shape, solver="spsolve").solve(K)
        for solver, tol in (("splu", 1e-12), ("cg", 1e-8)):
            p = fv.Permeameter(shape, solver=solver).solve(K)
            err = np.abs(p - ref).max() / np.abs(ref).max()
            assert err < tol, (shape, solver, err)
    assert fv.Permeameter((4, 4)).solver == "splu"
    assert fv.Permeameter((4, 4, 4)).solver == "cg"
