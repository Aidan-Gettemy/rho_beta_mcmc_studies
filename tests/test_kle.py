"""
KLE checks (python tests/run_tests.py).

  quadrature  Gauss-Legendre exactness, weights = volume (MATLAB selftest 1-2)
  identity    midpoint-rule Nystrom on the cell centers == discrete dense KLE
  separable   nystrom_sep == nystrom (full) on the same Gauss-Legendre rule,
              2D and 3D: same spectrum, same truncated covariance
  kron        discrete kron == discrete dense
  Nystrom     extension reproduces the nodal eigenfunctions (MATLAB selftest 6);
              weighted orthonormality at the nodes (selftest 5)
  converge    GL eigenvalues converge in nq; nq_check reports it
  July        auto keeps the dense July KLE; legacy frozen basis still loads
  full-rank   Kronecker Cholesky draws == dense Cholesky draws
  frozen      save/load round trip, Problem(frozen=...), build-once store
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import fv, config as C                               # noqa: E402
from brho import kernels as KN                                 # noqa: E402
from brho.kle import (KLE, build_kle, FullRankGaussian,        # noqa: E402
                      _nystrom_full)
from brho.quadrature import gauss_legendre_1d, tensor_rule    # noqa: E402

P = dict(ell=0.1, sigma2=1.0)
cov = lambda T: T @ T.T


def grid(shape):
    ph = fv.Permeameter(shape)
    return ph.points, ph.axes


def test_quadrature():
    x, w = gauss_legendre_1d(5)
    assert abs((w * x ** 6).sum() - 1 / 7) < 1e-14
    _, w2 = tensor_rule("gauss_legendre", 8, 2, 0.0, 1.0)
    _, w3 = tensor_rule("gauss_legendre", 6, 3, 0.0, 2.0)
    assert abs(w2.sum() - 1) < 1e-13 and abs(w3.sum() - 8) < 1e-12


def test_midpoint_nystrom_is_discrete_kle():
    pts, axes = grid((16, 16))
    d = build_kle(pts, axes, "sqexp", P, 0.95, "dense")
    m = build_kle(pts, axes, "sqexp", P, 0.95, "nystrom", nq=16, quadrature="midpoint")
    assert np.abs(m.lam[:d.r] * 256 - d.lam).max() < 1e-12
    assert np.abs(cov(m.T[:, :d.r]) - cov(d.T)).max() < 1e-12


def test_separable_equals_full_nystrom():
    for shape, prm, nq in [((16, 16), P, 20), ((12, 12), dict(ell=0.15), 16),
                           ((8, 8, 8), dict(ell=0.25), 8)]:
        pts, axes = grid(shape)
        f = build_kle(pts, axes, "sqexp", prm, 0.95, "nystrom", nq=nq)
        s = build_kle(pts, axes, "sqexp", prm, 0.95, "nystrom_sep", nq=nq)
        assert f.r == s.r, (shape, f.r, s.r)
        assert np.abs(f.lam - s.lam).max() < 1e-13
        assert np.abs(cov(f.T) - cov(s.T)).max() < 1e-11
        assert abs(f.total - s.total) < 1e-12


def test_kron_equals_dense():
    for shape, ell in [((16, 16), 0.1), ((12, 9), 0.15), ((6, 5, 4), 0.3)]:
        pts, axes = grid(shape)
        d = build_kle(pts, axes, "sqexp", dict(ell=ell), 0.95, "dense")
        k = build_kle(pts, axes, "sqexp", dict(ell=ell), 0.95, "kron")
        assert d.r == k.r
        assert np.abs(np.sort(d.lam_all)[::-1] - k.lam_all).max() < 1e-12
        assert np.abs(cov(d.T) - cov(k.T)).max() < 1e-12


def test_nystrom_extension_self_consistent():
    """Extending to the quadrature nodes returns the nodal eigenfunctions,
    which are orthonormal in the weighted inner product."""
    Xq, wq = tensor_rule("gauss_legendre", 12, 2)
    k = _nystrom_full(Xq, "sqexp", dict(ell=0.25), 0.95, 12, "gauss_legendre")
    phi = k.T / np.sqrt(k.lam)                         # phi_k at the nodes
    G = phi.T @ (wq[:, None] * phi)
    assert np.abs(G - np.eye(k.r)).max() < 1e-8
    K = KN.sqexp_cov(Xq, Xq, dict(ell=0.25))
    lhs = K @ (wq[:, None] * phi)                      # int C phi = lam phi
    assert np.abs(lhs - phi * k.lam).max() < 1e-8


def test_quadrature_convergence_and_check():
    pts, axes = grid((16, 16))
    ref = build_kle(pts, axes, "sqexp", P, 0.95, "nystrom_sep", nq=90)
    g = build_kle(pts, axes, "sqexp", P, 0.95, "nystrom_sep", nq=30)
    assert g.r == ref.r and np.abs(g.lam / ref.lam - 1).max() < 1e-12
    assert g.meta["nq_check"]["converged"]
    coarse = build_kle(pts, axes, "sqexp", P, 0.95, "nystrom_sep", nq=10)
    assert not coarse.meta["nq_check"]["converged"]
    # truncated variance tracks the retained energy
    assert abs(g.pointwise_variance().mean() - g.energy) < 0.01


def test_tied_modes_kept_together():
    pts, axes = grid((16, 16))
    g = build_kle(pts, axes, "sqexp", P, 0.95, "nystrom_sep", nq=30)
    assert abs(g.lam_all[g.r] - g.lam_all[g.r - 1]) > 1e-9 * g.lam_all[g.r - 1]


def test_auto_and_legacy():
    pts, axes = grid((16, 16))
    assert build_kle(pts, axes, "sqexp", P, 0.95).method == "dense"
    pts, axes = grid((80, 80))
    assert build_kle(pts, axes, "sqexp", P, 0.95).method == "nystrom_sep"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    legacy = os.path.join(here, "runs/july_raw/results/e2_grid/kle.npz")
    if os.path.exists(legacy):
        k = KLE.load(legacy)
        assert k.r == 56 and k.T.shape == (256, 56)


def test_full_rank_kron_equals_dense():
    pts, axes = grid((12, 9))
    d = FullRankGaussian(pts, axes, "sqexp", dict(ell=0.15), "dense")
    k = FullRankGaussian(pts, axes, "sqexp", dict(ell=0.15), "kron")
    gd = d.sample(np.random.default_rng(5), 3)
    gk = k.sample(np.random.default_rng(5), 3)
    assert np.abs(gd - gk).max() < 1e-3                # jitter enters differently


def test_frozen_and_store():
    from brho.problem import Problem
    pc = C.ProblemConfig(kle_method="nystrom_sep", kle_nq=30)
    with tempfile.TemporaryDirectory() as d:
        logs = []
        a = Problem(pc, store=d, log=logs.append)
        b = Problem(pc, store=d, log=logs.append)            # loaded, not rebuilt
        assert sum("building" in m for m in logs) == 2       # kle + truth once
        assert sum("loaded" in m for m in logs) == 2
        assert np.array_equal(a.kle.T, b.kle.T)
        assert np.array_equal(a.syn.d_p, b.syn.d_p)
        run = os.path.join(d, "run")
        os.makedirs(run)
        a.freeze(run)
        c = Problem(pc, frozen=run)
        assert c.frozen and np.array_equal(c.kle.T, a.kle.T)
        assert c.kle.meta["nq"] == 30
        th = np.random.default_rng(1).standard_normal(a.r)
        K = np.exp(a.kle.field(th))
        (pa, oa, ka), (pb, ob, kb) = a.potential(K), c.potential(K)
        assert pa == pb and ka == kb and np.array_equal(oa, ob)
        try:
            Problem(pc, frozen=os.path.join(d, "missing"))
            raise AssertionError("missing frozen dir must raise")
        except FileNotFoundError:
            pass
