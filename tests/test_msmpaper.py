"""Checks for the paper-MSM study (brho/msmpaper.py, docs/paper_msm_experiment.md)."""
import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import msmpaper as M  # noqa: E402


@functools.lru_cache(maxsize=None)
def _S():
    return M.PaperSetup()


@functools.lru_cache(maxsize=None)
def _D():
    return M.PaperData(_S())


def test_local_kle_structure():
    S = _S()
    assert S.p == 20 and S.E.shape == (256, 20)
    assert [len(c) for c in S.sub_cells] == [64] * 4
    # local modes live on their own subdomain only and are the same function in all four blocks
    for i, cells in enumerate(S.sub_cells):
        other = np.setdiff1d(np.arange(256), cells)
        assert np.all(S.E[np.ix_(other, np.arange(i * 5, (i + 1) * 5))] == 0)
        assert np.allclose(S.E[cells, i * 5:(i + 1) * 5], S.E[S.sub_cells[0], 0:5])
    assert S.tie_cut                                  # the cut at 5 splits the tied pair (5, 6): documented
    assert np.allclose(S.mu[1], S.mu[2]) and S.mu[3] > S.mu[4] > 2.0


def test_local_basis_independent_of_eigensolver():
    """The kept direction of the tied pair, and the basis of the pair (2, 3), must not depend on LAPACK."""
    ref = M.PaperSetup().E.copy()
    real = np.linalg.eigh
    rng = np.random.default_rng(3)

    def scrambled(a, *args, **kw):
        lam, U = real(a, *args, **kw)
        U = U.copy()
        i = 0
        while i < len(lam):
            j = i + 1
            while j < len(lam) and abs(lam[j] - lam[i]) <= 1e-9 * abs(lam[i]):
                j += 1
            Q, _ = np.linalg.qr(rng.standard_normal((j - i, j - i)))
            U[:, i:j] = U[:, i:j] @ Q
            i = j
        return lam, U

    np.linalg.eigh = scrambled
    try:
        E2 = M.PaperSetup().E
    finally:
        np.linalg.eigh = real
    assert np.max(np.abs(E2 - ref)) < 1e-8


def test_smoothing_operator():
    S = _S()
    assert len(S.band_idx) == 112                       # 4 of 16 columns and rows next to the interfaces
    assert np.allclose(S.A.sum(1), 1.0)                 # plain averages (mean preserving)
    n_nb = np.round(1.0 / S.A_band.max(1))
    assert n_nb.min() == 6 and n_nb.max() == 9
    # cells outside the band are untouched, a constant field is unchanged
    Y = np.random.default_rng(0).standard_normal(256)
    out = S.smooth_all(Y)
    assert np.array_equal(out[~S.band], Y[~S.band])
    assert np.allclose(S.smooth_all(np.ones(256)), 1.0)
    # linearity (all-at-once smoothing is a linear map of c)
    c1, c2 = np.random.default_rng(1).standard_normal((2, 20))
    f = lambda c: S.field("smooth", c)
    assert np.allclose(f(2 * c1 + c2), 2 * f(c1) + f(c2))


def test_smoothing_history_update():
    S = _S()
    rng = np.random.default_rng(2)
    eta = S.smooth_all(S.E @ rng.standard_normal(20))
    c = rng.standard_normal(20)
    i = 2
    Yi = np.zeros(256)
    sel = S.coef_sub == i
    Yi[S.sub_cells[i]] = S.E[np.ix_(S.sub_cells[i], np.flatnonzero(sel))] @ c[sel]
    new = S.smooth_subdomain(eta, i, Yi)
    other = np.setdiff1d(np.arange(256), S.sub_cells[i])
    assert np.array_equal(new[other], eta[other])               # the other subdomains keep their stored values
    inner = np.setdiff1d(S.sub_cells[i], S.band_sub[i])
    assert np.allclose(new[inner], Yi[inner])                   # non-band cells: the new raw field
    V = eta.copy(); V[S.sub_cells[i]] = Yi[S.sub_cells[i]]
    assert np.allclose(new[S.band_sub[i]], S.A[S.band_sub[i]] @ V)    # band: averages of V (stale neighbours)
    # with the neighbours' stored values replaced by raw values the update equals the simultaneous smoothing there
    Yraw = S.E @ c
    assert np.allclose(S.smooth_subdomain(Yraw, i, Yi)[S.band_sub[i]], S.smooth_all(Yraw)[S.band_sub[i]])


def test_projected_field_is_in_global_span():
    S = _S()
    c = np.random.default_rng(4).standard_normal(20)
    eta = S.field("proj", c)
    assert np.allclose(S.T @ S.coords(eta), eta, atol=1e-9)
    # the projection is the L2 projection of the raw field
    assert np.allclose(S.coords(S.E @ c), S.Bproj @ c)


def test_groups_and_chain_determinism():
    S, D = _S(), _D()
    assert [len(g) for g in M.make_groups("all", S, 20)] == [20]
    assert [len(g) for g in M.make_groups("coord", S, 20)] == [1] * 20
    assert [len(g) for g in M.make_groups("subdomain", S, 20)] == [5] * 4
    for exp in M.EXPS:
        a = M.run_chain(S, D, exp, 7, 3, n_iter=400, n_adapt=200, field_every=100)
        b = M.run_chain(S, D, exp, 7, 3, n_iter=400, n_adapt=200, field_every=100)
        assert np.array_equal(a["c"], b["c"]) and np.array_equal(a["acc"], b["acc"])
        assert a["eta_thin"].shape == (4, 256) and np.isfinite(a["relerr"]).all()


def test_e0_is_gridstudy_global_pcn():
    """The control engine evaluates the same likelihood as the grid study's arm 1 at 16 x 16."""
    from brho import gridstudy as G
    S, D = _S(), _D()
    g = G.GridData(S.G, "check")
    th = np.random.default_rng(5).standard_normal(20)
    p0, r0 = g.potential(th)
    p1, r1 = D.potential(S.field("global", th))
    assert abs(p0 - p1) < 1e-9 * max(1, abs(p0)) and abs(r0 - r1) < 1e-12
