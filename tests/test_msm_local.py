"""
Checks for the local-KLE (MSM) experiment code: the extended MSInnovation
(brho/multiscale.py: explicit partition, fixed local dimension, local
coefficient maps), brho/msm_local.py (setup, rank report, moves, prior chains),
and the recorded outcome of the Sigma_ms rank test
(docs/msm_local_kle_experiment.md, Section 6).
"""
import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import multiscale as MS                      # noqa: E402
from brho.blockpcn import boxes                        # noqa: E402
from brho import msm_local as ML                       # noqa: E402


@functools.lru_cache(maxsize=None)
def _setup(primal_local_dim=4):
    return ML.MSMSetup(primal_local_dim=primal_local_dim)


def test_dual_partition_matches_blockpcn_boxes():
    S = _setup()
    idx = MS.dual_block_index(S.points, 4)
    bx = boxes(2, 4, True)
    assert len(bx) == 25 and idx.max() == 24
    for k, (lo, hi) in enumerate(bx):
        cells = np.flatnonzero(idx == k)
        assert len(cells) > 0
        P = S.points[cells]
        assert np.all(P >= lo) and np.all(P <= hi), k
    # 4 quarter-size corners, 12 half-size edges, 9 interior blocks of 8 x 8 cells
    sizes = np.bincount(idx)
    assert sorted(sizes) == [16] * 4 + [32] * 12 + [64] * 9


def test_explicit_partition_equals_default_path():
    S = _setup()
    kp = dict(ell=0.1, sigma2=1.0)
    a = MS.MSInnovation(S.kle, S.points, "sqexp", kp, blocks=4, local_energy=0.95)
    b = MS.MSInnovation(S.kle, S.points, "sqexp", kp, blocks=4, local_energy=0.95,
                        block_idx=MS.block_index(S.points, 4))
    assert np.array_equal(a.B, b.B) and a.r_k == b.r_k
    try:
        MS.MSInnovation(S.kle, S.points, "sqexp", kp, block_idx=np.zeros(5, int))
    except ValueError:
        pass
    else:
        raise AssertionError("block_idx of the wrong length must raise")


def test_fixed_local_dim_and_tie_flag():
    S = _setup()
    assert S.primal.r_k == [4] * 16 and S.primal.p == 64
    assert sum(S.primal.tie_cut) == 0              # the 4th local mode is not tied with the 5th
    kp = dict(ell=0.1, sigma2=1.0)
    m5 = MS.MSInnovation(S.kle, S.points, "sqexp", kp, blocks=4, local_dim=5)
    assert m5.r_k == [5] * 16 and all(m5.tie_cut)  # 5 cuts the tied pair (modes 5, 6)
    assert S.dual.p == 114 and sum(S.dual.tie_cut) == 0


def test_local_maps_are_consistent():
    S = _setup()
    rng = np.random.default_rng(41)
    for ms in (S.primal, S.dual):
        c = rng.standard_normal(ms.p)
        f = ms.local_field(c)
        assert f.shape == (S.phys.N,)
        # orthonormal local modes: coefficients of the piecewise field are c itself
        assert np.abs(ms.local_coeffs(f) - c).max() < 1e-12
        # B is "build the piecewise field, project onto the global KLE"
        assert np.abs(ms.B @ c - S.kle.project(f)).max() < 1e-12
        # Pi is "field of theta, then local coefficients"
        th = rng.standard_normal(S.r)
        assert np.abs(ms.Pi @ th - ms.local_coeffs(S.kle.field(th))).max() < 1e-12
        # batched shapes
        C = rng.standard_normal((3, ms.p))
        assert ms.local_field(C).shape == (3, S.phys.N)
        assert np.abs(ms.local_field(C)[1] - ms.local_field(C[1])).max() < 1e-14


def test_rank_test_outcome_primal_fails_dual_passes():
    """The recorded result of the Sigma_ms rank test (LOG.md 2026-11-02): with 4
    local modes per block the primal Sigma_ms has rank 58 of 64 (a 12-orders-
    of-magnitude gap in the singular values of B), so the whitened primal
    sampler is not available; the dual grid is full rank."""
    S = _setup()
    rp, rd = ML.rank_report(S.primal), ML.rank_report(S.dual)
    assert rp["p"] == 64 and rp["r_plus"] == 58 and rp["matrix_rank_B"] == 58 and not rp["passed"]
    sv = rp["singular_values_B"]
    assert sv[57] > 0.3 and sv[58] < 1e-10
    assert rd["p"] == 114 and rd["r_plus"] == 64 and rd["passed"]
    assert rd["cond_Sigma"] < 20


def test_full_rank_with_six_local_modes_and_whitening():
    S6 = _setup(6)                                   # information only, not the agreed setup
    rep = ML.rank_report(S6.primal)
    assert rep["p"] == 96 and rep["passed"] and rep["r_plus"] == 64
    P = S6.primal
    assert P.operator_check < 1e-12
    assert np.abs(P.WB @ P.WB.T - np.eye(64)).max() < 1e-12      # theta = W B c has law N(0, I)


def test_whitened_map_refuses_rank_deficient_sigma_and_3a_3b_laws():
    S = _setup()
    P = S.primal
    c = np.zeros((2, P.p))
    try:
        ML.theta_from_c(P, c, whiten=True)
    except ValueError as e:
        assert "rank-deficient" in str(e)
    else:
        raise AssertionError("exact whitened map must refuse a rank-deficient Sigma_ms")
    assert ML.theta_from_c(P, c).shape == (2, 64)                 # raw map is fine
    # exact laws: 3a  cov = V+ V+^T  (identity on the range, zero on the null space);
    #             3b  cov = I
    Wb = P.WB
    cov_a = Wb @ Wb.T
    assert np.abs(cov_a - P.Vp @ P.Vp.T).max() < 1e-10
    cov_b = cov_a + P.V0 @ P.V0.T
    assert np.abs(cov_b - np.eye(64)).max() < 1e-10
    assert P.r_plus == 58 and P.V0.shape[1] == 6


def test_range_coords_shape_and_roundtrip():
    S = _setup()
    P = S.primal
    rng = np.random.default_rng(44)
    th = ML.theta_from_c(P, rng.standard_normal((5, P.p)))
    z = ML.range_coords(P, th)
    assert z.shape == (5, 58)
    assert np.abs(z @ P.Vp.T - th).max() < 1e-10                  # theta lies in the range


def test_moves_deterministic_limits():
    S = _setup()
    rng = np.random.default_rng(42)
    c = rng.standard_normal((4, S.primal.p))
    assert np.array_equal(ML.step_c(c, 0.0, rng), c)
    th = rng.standard_normal((4, S.r))
    for ms in (S.primal, S.dual):
        # s = 0: the experiment-4 move is the lossy round trip theta -> c -> theta
        assert np.abs(ML.theta_move_via_grid(ms, th, 0.0, rng) - (th @ ms.Pi.T) @ ms.B.T).max() < 1e-12


def test_prior_chain_shapes_and_switch():
    S, S6 = _setup(), _setup(6)
    rng = np.random.default_rng(43)
    m, n, thin, burn = 3, 40, 5, 10
    for kind in ("global", "primal_raw", "mixture"):
        out = ML.prior_chain(kind, S, m, n, thin, burn, 0.5, rng)
        assert out["theta"].shape == (m, 6, 64), kind
        assert np.all(np.isfinite(out["theta"])), kind
    out = ML.prior_chain("primal_white", S6, m, n, thin, burn, 0.5, rng)
    assert out["theta"].shape == (m, 6, 64) and out["c"].shape == (m, 6, 96)
    out = ML.prior_chain("sequence", S, m, n, thin, burn, 0.5, rng)
    assert out["theta1"].shape == out["theta2"].shape == (m, 2, 64)
    assert out["c1"].shape == (m, 2, 64) and out["c2"].shape == (m, 2, 114)
    assert 0.0 < out["switch_field_change"] < 1.0
    try:
        ML.prior_chain("primal_white", S, m, n, thin, burn, 0.5, rng)
    except ValueError:
        pass
    else:
        raise AssertionError("exact whitened primal chain must refuse the rank-deficient 4-mode grid")
    out = ML.prior_chain("primal_pinv", S, m, n, thin, burn, 0.5, rng)
    assert out["theta"].shape == (m, 6, 64) and "zeta" not in out
    out = ML.prior_chain("primal_aug", S, m, n, thin, burn, 0.5, rng)
    assert out["theta"].shape == (m, 6, 64) and out["zeta"].shape == (m, 6, 6)
