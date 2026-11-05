"""Checks for the grid-refinement study (brho/gridstudy.py, docs/grid_arms_experiment.md)."""
import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import gridstudy as G                       # noqa: E402


@functools.lru_cache(maxsize=None)
def _S(n):
    return G.GridSetup(n)


@functools.lru_cache(maxsize=None)
def _D(n, arm):
    return G.GridData(_S(n), arm)


def test_kle_same_modes_on_every_grid():
    # the Nystrom builder uses a grid-independent quadrature: identical eigenvalues, and the field of one
    # theta on a fine grid agrees with the coarse one up to point-versus-cell-average sampling
    th = G.theta_truth()
    f = {n: _S(n).kle.field(th) for n in G.GRIDS}
    for n in G.GRIDS:
        assert _S(n).kle.T.shape == (n * n, 20)
        assert np.allclose(_S(n).kle.lam, _S(16).kle.lam, rtol=1e-12)
    def block(v, n, m):
        return v.reshape(m, n // m, m, n // m).mean((1, 3)).ravel()
    assert np.linalg.norm(block(f[64], 64, 16) - f[16]) / np.linalg.norm(f[16]) < 0.01
    assert np.linalg.norm(block(f[32], 32, 16) - f[16]) / np.linalg.norm(f[16]) < 0.01
    assert 0.96 < _S(16).kle.energy < 0.98            # 20 modes at ell = 0.2 hold about 97.1% of the variance


def test_truncation_does_not_split_a_degenerate_pair():
    lam = _S(16).kle.lam
    assert lam[19] > lam[20] * (1 + 1e-6) if len(lam) > 20 else True
    S = _S(16)
    class K:                                              # a KLE whose modes 20 and 21 are tied is refused
        lam = np.array([1.0, .5, .25, .25]); T = np.ones((4, 4)); total = 2.0
    try:
        G.TruncKLE(K, 3)
    except ValueError:
        return
    raise AssertionError("tied cut was accepted")


def test_observation_sets():
    assert [len(G.obs_index(_S(n), "check")) for n in G.GRIDS] == [128, 512, 2048]
    assert [len(G.obs_index(_S(n), "abs")) for n in G.GRIDS] == [128, 512, 2048]
    assert [len(G.obs_index(_S(n), "fixed")) for n in G.GRIDS] == [128, 128, 128]
    assert np.array_equal(G.obs_index(_S(16), "fixed"), G.obs_index(_S(16), "check"))   # arms coincide at 16
    for n in (16, 32, 64):                                # centre cells are within half a fine-cell diagonal
        idx = G.centre_cell_indices(n)
        x, y = (idx // n + .5) / n, (idx % n + .5) / n
        cx, cy = (np.floor(x * 16) + .5) / 16, (np.floor(y * 16) + .5) / 16
        assert np.hypot(x - cx, y - cy).max() <= np.sqrt(2) / 2 / n + 1e-12
        assert len(set(idx.tolist())) == 128


def test_sigma_abs_is_eps_times_16_data_norm():
    D16 = _D(16, "check")
    s = G.sigma_abs_reference()
    assert abs(s - 0.003 * np.linalg.norm(D16.obs_clean)) < 1e-12
    assert abs(s - 0.01670280300644679) < 1e-9, f"sigma_abs_reference() = {s!r}, expected 0.01670280300644679 (sandbox value)"
    for n in G.GRIDS:
        D = _D(n, "abs")
        # tolerance, not ==: each GridData recomputes sigma_abs from a fresh solve and threaded
        # linear algebra may differ in the last bit between calls
        assert abs(D.sigma_abs - s) < 1e-12 and D.sigma_p == D.sigma_abs


def test_same_theta_and_shared_noise():
    th = G.theta_truth()
    for n in G.GRIDS:
        for arm in G.ARMS:
            assert np.array_equal(_D(n, arm).theta_true, th)
    # fixed-count arm: the same 128 noise draws on every grid; check and abs share them at equal obs sets
    xi = lambda D: (D.d - D.obs_clean) / D.sigma_p
    assert np.allclose(xi(_D(16, "check")), xi(_D(16, "fixed")))
    assert np.allclose(xi(_D(32, "fixed")), xi(_D(64, "fixed")))
    for n in G.GRIDS:
        assert np.allclose(xi(_D(n, "check")), xi(_D(n, "abs")))


def test_likelihoods_at_truth():
    for n in G.GRIDS:
        for arm in ("check", "fixed"):
            D = _D(n, arm)                                # relative-error arms: Phi at the truth about 0.5
            assert 0.2 < D.phi_truth < 1.2 and 0.002 < D.relerr_truth < 0.0045
            assert abs(D.potential(D.theta_true)[0] - D.phi_truth) < 1e-8 * max(1, D.phi_truth)
        D = _D(n, "abs")                                  # proper Gaussian: Phi at the truth about n_obs / 2
        assert abs(D.phi_truth / D.phi_expected - 1) < 0.35
    # at 16 x 16 the abs likelihood is the relative-error one (same weight up to the noisy ||d||)
    assert abs(_D(16, "abs").w / _D(16, "check").w - 1) < 0.01


def test_chain_runs_and_adapts():
    D = _D(16, "check")
    o = G.run_chain(_S(16), D, seed0=7, chain=0, n_iter=1500, n_adapt=1000, save_every=50)
    assert o["theta"].shape == (30, 20) and np.isfinite(o["relerr"]).all()
    assert 0.05 < o["acc"][1000:].mean() < 0.7
    o2 = G.run_chain(_S(16), D, seed0=7, chain=0, n_iter=1500, n_adapt=1000, save_every=50)
    assert np.array_equal(o["theta"], o2["theta"])        # deterministic given (seed0, chain)
