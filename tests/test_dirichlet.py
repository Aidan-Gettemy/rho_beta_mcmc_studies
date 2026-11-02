"""
Checks for the Dirichlet-left option of brho/fv.py (python tests/run_tests.py).

    bc="dirichlet":  p = p_left (1) on x = 0,  p = p_right (0) on x = 1,
                     no flow on the lateral faces.

  constant K       p = 1 - x at the cell centers (exact), Keff = K
  K(y) only        p = 1 - x exactly, Keff = arithmetic mean of K(y)
  K(x) layered     Keff = harmonic mean exactly, p = the series solution
  reference        an explicit per-face assembly reproduces the matrix and p
  conservation     left flux = right flux, tiny residual, 0 <= p <= 1,
                   harmonic mean <= Keff <= arithmetic mean (Wiener bounds)
  scaling          p(cK) = p(K), Keff(cK) = c Keff(K)
  solvers          splu / spsolve / cg agree, numba == NumPy (if installed)
  convergence      Keff converges at second order on a smooth K(x, y)
  interface        argument checks, 32 x 32 has 512 checkerboard observations,
                   Darcy2D passes the option through, the influx default is
                   unchanged
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import fv                                  # noqa: E402
from brho.physics2d import Darcy2D                   # noqa: E402


def _solver_for(shape):
    """Direct solves keep the exactness checks tight (no CG tolerance)."""
    return "splu" if len(shape) == 2 else "spsolve"


def reference_dirichlet(shape, K, p_left, p_right):
    """Dense A, b from an explicit loop over cells and faces (2D only):
    harmonic inter-cell transmissibilities, half-cell links to the Dirichlet
    faces.  Written independently of fv.py."""
    nx, ny = shape
    hx, hy = 1.0 / nx, 1.0 / ny
    K = np.asarray(K, float).reshape(nx, ny)
    N = nx * ny
    A, b = np.zeros((N, N)), np.zeros(N)
    m = lambda ix, iy: ix * ny + iy
    for ix in range(nx):
        for iy in range(ny):
            i, k = m(ix, iy), K[ix, iy]
            if ix < nx - 1:
                j, k2 = m(ix + 1, iy), K[ix + 1, iy]
                T = (hy / hx) * 2.0 * k * k2 / (k + k2)
                A[i, i] += T; A[j, j] += T; A[i, j] -= T; A[j, i] -= T
            if iy < ny - 1:
                j, k2 = m(ix, iy + 1), K[ix, iy + 1]
                T = (hx / hy) * 2.0 * k * k2 / (k + k2)
                A[i, i] += T; A[j, j] += T; A[i, j] -= T; A[j, i] -= T
            if ix == 0:
                T = (hy / hx) * 2.0 * k
                A[i, i] += T; b[i] += T * p_left
            if ix == nx - 1:
                T = (hy / hx) * 2.0 * k
                A[i, i] += T; b[i] += T * p_right
    return A, b


def test_dirichlet_constant_K():
    for shape in [(32, 32), (7, 5), (6, 6, 6)]:
        ph = fv.Permeameter(shape, bc="dirichlet", solver=_solver_for(shape))
        for k0 in (1.0, 3.7, 1e-3, 250.0):
            K = np.full(ph.N, k0)
            p = ph.solve(K)
            assert np.abs(p - (1.0 - ph.points[:, 0])).max() < 1e-11, (shape, k0)
            assert abs(ph.keff(K, p) / k0 - 1) < 1e-11, (shape, k0)
            ql, qr = ph.boundary_flux(K, p)
            assert abs(ql / k0 - 1) < 1e-11 and abs(qr / k0 - 1) < 1e-11, (shape, k0)


def test_dirichlet_custom_pressures():
    ph = fv.Permeameter((16, 16), bc="dirichlet", p_left=3.0, p_right=1.0)
    K = np.full(ph.N, 2.0)
    p = ph.solve(K)
    assert np.abs(p - (3.0 - 2.0 * ph.points[:, 0])).max() < 1e-11
    assert abs(ph.keff(K, p) / 2.0 - 1) < 1e-11


def test_dirichlet_K_of_y_only():
    rng = np.random.default_rng(21)
    for shape in [(16, 16), (12, 9)]:
        nx, ny = shape
        ph = fv.Permeameter(shape, bc="dirichlet")
        ky = np.exp(1.5 * rng.standard_normal(ny))
        K = np.tile(ky, (nx, 1)).ravel()                 # cell m = ix*ny + iy
        p = ph.solve(K)
        assert np.abs(p - (1.0 - ph.points[:, 0])).max() < 1e-10, shape
        assert abs(ph.keff(K, p) / ky.mean() - 1) < 1e-10, shape


def test_dirichlet_K_of_x_layered():
    rng = np.random.default_rng(22)
    for shape in [(16, 16), (9, 12)]:
        nx, ny = shape
        ph = fv.Permeameter(shape, bc="dirichlet")
        kx = np.exp(1.2 * rng.standard_normal(nx))
        K = np.repeat(kx, ny)                            # cell m = ix*ny + iy
        p = ph.solve(K)
        h = 1.0 / nx
        keff_exact = 1.0 / (h * (1.0 / kx).sum())        # harmonic mean
        assert abs(ph.keff(K, p) / keff_exact - 1) < 1e-11, shape
        # series solution at the cell centers: p_i = 1 - Q (sum_{j<i} h/k_j + h/(2 k_i))
        q = keff_exact
        p_exact = 1.0 - q * (np.concatenate([[0.0], np.cumsum(h / kx)[:-1]]) + h / (2 * kx))
        assert np.abs(p.reshape(nx, ny) - p_exact[:, None]).max() < 1e-11, shape


def test_dirichlet_matches_independent_reference():
    rng = np.random.default_rng(23)
    for shape, (pl, pr) in [((6, 6), (1.0, 0.0)), ((7, 5), (1.0, 0.0)), ((5, 8), (2.5, -0.5))]:
        ph = fv.Permeameter(shape, bc="dirichlet", p_left=pl, p_right=pr)
        K = np.exp(1.5 * rng.standard_normal(ph.N))
        A0, b0 = reference_dirichlet(shape, K, pl, pr)
        A = ph.matrix(K).toarray()
        assert np.abs(A - A0).max() < 1e-12 * np.abs(A0).max(), shape
        assert np.abs(ph._rhs(K) - b0).max() < 1e-12 * np.abs(b0).max(), shape
        p0 = np.linalg.solve(A0, b0)
        p = ph.solve(K)
        assert np.abs(p - p0).max() / np.abs(p0).max() < 1e-11, shape


def test_dirichlet_conservation_and_bounds():
    rng = np.random.default_rng(24)
    ph = fv.Permeameter((32, 32), bc="dirichlet")
    for sd in (0.5, 1.5, 2.0):
        K = np.exp(sd * rng.standard_normal(ph.N))
        p = ph.solve(K)
        A, b = ph.matrix(K), ph._rhs(K)
        assert np.linalg.norm(A @ p - b) / np.linalg.norm(b) < 1e-10, sd
        ql, qr = ph.boundary_flux(K, p)
        assert abs(ql - qr) / abs(ql) < 1e-10, sd
        assert p.min() > -1e-12 and p.max() < 1.0 + 1e-12, sd          # maximum principle
        keff = ph.keff(K, p)
        assert 1.0 / np.mean(1.0 / K) * (1 - 1e-12) <= keff <= K.mean() * (1 + 1e-12), sd
    # the system is symmetric positive definite
    assert abs(A - A.T).max() == 0
    assert np.linalg.eigvalsh(ph.matrix(K).toarray()).min() > 0


def test_dirichlet_scaling():
    rng = np.random.default_rng(25)
    for shape in [(24, 24), (8, 8, 8)]:
        ph = fv.Permeameter(shape, bc="dirichlet", solver=_solver_for(shape))
        K = np.exp(rng.standard_normal(ph.N))
        sc = ph.scaling_check(K)
        assert sc["p_rel_err"] < 1e-11 and sc["keff_rel_err"] < 1e-11, (shape, sc)


def test_dirichlet_solvers_agree():
    rng = np.random.default_rng(26)
    for shape in [(30, 30), (8, 8, 8)]:
        K = np.exp(1.5 * rng.standard_normal(int(np.prod(shape))))
        ref = fv.Permeameter(shape, bc="dirichlet", solver="spsolve").solve(K)
        for solver, tol in (("splu", 1e-12), ("cg", 1e-8)):
            p = fv.Permeameter(shape, bc="dirichlet", solver=solver).solve(K)
            err = np.abs(p - ref).max() / np.abs(ref).max()
            assert err < tol, (shape, solver, err)
    assert fv.Permeameter((4, 4), bc="dirichlet").solver == "splu"
    assert fv.Permeameter((4, 4, 4), bc="dirichlet").solver == "cg"


def test_dirichlet_backends_agree():
    rng = np.random.default_rng(27)
    for shape in [(16, 16), (10, 10, 10)]:
        K = np.exp(rng.standard_normal(int(np.prod(shape))))
        b_np = fv.Permeameter(shape, bc="dirichlet", backend="numpy")
        d_np = b_np.csr_data(K)
        # the compiled kernel (plain Python when numba is absent) + the Dirichlet term
        d_k = np.empty(b_np.nnz)
        fv._fill_csr_numba(K, *b_np._n3, *b_np._f, d_k)
        d_k[b_np._dpos_L] += 2.0 * K[b_np.mL] * b_np._f[0]
        assert np.abs(d_k - d_np).max() <= 1e-13 * np.abs(d_np).max(), shape
    if not fv.HAVE_NUMBA:
        print("      (numba not installed: compiled backend skipped)")
        return
    for shape in [(16, 16), (100, 100), (10, 10, 10)]:
        K = np.exp(rng.standard_normal(int(np.prod(shape))))
        a = fv.Permeameter(shape, bc="dirichlet", backend="numba").solve(K)
        b = fv.Permeameter(shape, bc="dirichlet", backend="numpy").solve(K)
        assert np.abs(a - b).max() / np.abs(b).max() < 1e-12, shape


def test_dirichlet_second_order_convergence():
    """Keff of a smooth K(x, y) converges at second order under refinement
    (successive differences shrink by ~4)."""
    kfun = lambda x, y: np.exp(0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 0.3 * x)
    keff = {}
    for n in (16, 32, 64, 128):
        ph = fv.Permeameter((n, n), bc="dirichlet")
        K = kfun(ph.points[:, 0], ph.points[:, 1])
        keff[n] = ph.keff(K, ph.solve(K))
    d = [abs(keff[a] - keff[b]) for a, b in ((16, 32), (32, 64), (64, 128))]
    assert 3.5 < d[0] / d[1] < 4.5 and 3.7 < d[1] / d[2] < 4.3, d


def test_dirichlet_interface():
    # the experiment's grid: 32 x 32 cells, 512 checkerboard observations
    ph = fv.Permeameter((32, 32), bc="dirichlet")
    assert ph.N == 1024 and ph.obs_mask.sum() == 512
    assert len(ph.mL) == 32 and len(ph.mR) == 32
    assert (ph.bc, ph.p_left, ph.p_right) == ("dirichlet", 1.0, 0.0)
    p, pobs, keff = ph.forward(np.ones(ph.N))
    assert pobs.shape == (512,) and abs(keff - 1.0) < 1e-12
    # argument checks
    for kwargs in (dict(bc="neumann"), dict(bc="influx", p_left=1.0),
                   dict(bc="influx", p_right=0.0), dict(bc="dirichlet", p_left=2.0, p_right=2.0)):
        try:
            fv.Permeameter((4, 4), **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {kwargs}")
    # Darcy2D passes the option through
    rng = np.random.default_rng(28)
    K = np.exp(rng.standard_normal(16 * 16))
    a = Darcy2D(16, bc="dirichlet").solve(K)
    b = fv.Permeameter((16, 16), bc="dirichlet").solve(K)
    assert np.abs(a - b).max() == 0


def test_influx_default_unchanged():
    rng = np.random.default_rng(29)
    ph = fv.Permeameter((16, 16))
    assert (ph.bc, ph.p_left, ph.p_right) == ("influx", None, None)
    K = np.exp(rng.standard_normal(ph.N))
    assert ph._rhs(K) is ph._b                           # constant influx right-hand side
    p_default = ph.solve(K)
    p_explicit = fv.Permeameter((16, 16), bc="influx").solve(K)
    assert np.abs(p_default - p_explicit).max() == 0
    assert np.abs(Darcy2D(16).solve(K) - p_default).max() == 0
    try:
        ph.boundary_flux(K, p_default)
    except ValueError:
        pass
    else:
        raise AssertionError("boundary_flux must be Dirichlet-only")
    sc = ph.scaling_check(K)
    assert sc["p_rel_err"] < 1e-12 and sc["keff_rel_err"] < 1e-12
