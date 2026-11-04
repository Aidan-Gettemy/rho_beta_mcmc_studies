"""Checks for the MSM chain sampler (brho/msm_sampler.py) and the field-norm lift."""
import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import msm_local as ML                      # noqa: E402
from brho import msm_sampler as MS                    # noqa: E402


@functools.lru_cache(maxsize=None)
def _setup():
    S = ML.MSMSetup()
    return S, MS.Data(S)


def test_field_lift_dual_exact_primal_projector():
    S, _ = _setup()
    T = S.kle.T
    Ld = ML.field_lift(S.dual, T)
    assert np.abs(S.dual.B @ Ld - np.eye(64)).max() < 1e-9              # lossless
    Lp = ML.field_lift(S.primal, T)
    Pm = S.primal.B @ Lp                                                  # theta -> best representable theta
    assert np.abs(Pm @ Pm - Pm).max() < 1e-9                              # projector
    lam = (T ** 2).sum(0)
    assert np.abs(lam[:, None] * Pm - (lam[:, None] * Pm).T).max() < 1e-9  # orthogonal in the field norm
    assert np.linalg.matrix_rank(Pm, tol=1e-8) == 58


def test_new_kinds_run_and_shapes():
    S, data = _setup()
    for kind, kw in (("alt", dict(alt_L=5)), ("alt", dict(alt_offset=False)), ("mixture_inn", {}),
                     ("sequence_exact", dict(k_phase=100))):
        o = MS.run_chain(kind, S, data, 3, 0, 200, 100, save_every=50, **kw)
        assert o["theta"].shape == (4, 64) and o["relerr"].shape == (200,), kind
        assert np.all(np.isfinite(o["relerr"])), kind
        if kind != "sequence_exact":
            assert o["c_primal"].shape == (4, 64) and o["c_dual"].shape == (4, 114)
        else:
            assert o["c1"].shape == (2, 64) and o["c2"].shape == (2, 114)
            assert abs(float(o["phi0"]) - float(o["phi1"])) < 1e-6 * max(1.0, abs(float(o["phi0"])))   # exact switch


def test_alt_schedule_follows_L():
    S, data = _setup()
    o = MS.run_chain("alt", S, data, 3, 0, 120, 60, save_every=60, alt_L=10)
    assert np.array_equal(o["move"], (np.arange(120) // 10) % 2)
