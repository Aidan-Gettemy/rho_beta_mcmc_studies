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


def test_kle_kron_equals_dense():
    """Separable KLE: same spectrum, same truncated covariance of G (the law
    of the prior field), Cholesky equal up to where the jitter enters."""
    from brho.kle import KLE
    for shape, ell in [((16, 16), 0.1), ((12, 9), 0.15), ((6, 5, 4), 0.3)]:
        ph = fv.Permeameter(shape)
        d = KLE(ph.points, ell, 0.95, axes=ph.axes, method="dense")
        k = KLE(ph.points, ell, 0.95, axes=ph.axes, method="kron")
        assert d.r == k.r
        assert np.abs(np.sort(d.lam_all)[::-1] - k.lam_all).max() < 1e-12
        Cd = (d.PHI * d.lam) @ d.PHI.T
        Ck = (k.PHI * k.lam) @ k.PHI.T
        assert np.abs(Cd - Ck).max() < 1e-12
        assert np.allclose(k.PHI.T @ k.PHI, np.eye(k.r), atol=1e-10)
        g_d = d.sample_full(np.random.default_rng(5))
        g_k = k.sample_full(np.random.default_rng(5))
        assert np.abs(g_d - g_k).max() < 1e-3


def test_kle_auto_keeps_july_dense():
    from brho.kle import KLE
    ph = fv.Permeameter((16, 16))
    assert KLE(ph.points, 0.1, axes=ph.axes).method == "dense"
    ph = fv.Permeameter((100, 100))
    k = KLE(ph.points, 0.1, axes=ph.axes)
    assert k.method == "kron" and k.C is None


def test_frozen_problem_round_trip():
    """A run's KLE basis and data are saved at setup and reloaded verbatim
    by chains and reports (eigh's basis inside degenerate eigenspaces is not
    reproducible across machines / thread counts)."""
    import tempfile
    from brho import config as C
    from brho.problem import Problem
    pc = C.ProblemConfig()
    a = Problem(pc)
    with tempfile.TemporaryDirectory() as d:
        a.freeze(d)
        b = Problem(pc, frozen=d)
        assert b.frozen and b.r == a.r
        assert np.array_equal(a.kle.PHI, b.kle.PHI)
        assert np.array_equal(a.kle.SQ, b.kle.SQ)
        assert np.array_equal(a.syn.d_p, b.syn.d_p)
        assert a.syn.keff_obs == b.syn.keff_obs
        th = np.random.default_rng(1).standard_normal(a.r)
        K = np.exp(a.kle.field(th))
        (pa, oa, ka), (pb, ob, kb) = a.potential(K), b.potential(K)
        assert pa == pb and ka == kb and np.array_equal(oa, ob)
        try:
            Problem(pc, frozen=os.path.join(d, "missing"))
            raise AssertionError("missing frozen dir must raise")
        except FileNotFoundError:
            pass


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
