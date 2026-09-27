"""E1 naive sampler checks (python tests/run_tests.py).

  resume      a chain interrupted at a checkpoint and resumed is identical to
              an uninterrupted one (RNG state, adaptation, histories)
  prior       with the data term removed, the global pCN chain samples the
              prior of u = (theta, xi_b, xi_r): N(0, I)
  reconstruct saved (theta, beta, rho) rebuild the exact log K and the
              predicted observations stored by the chain
"""
import os
import sys
import tempfile
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import config as C                          # noqa: E402
from brho.problem import Problem                      # noqa: E402
from brho.naive import run_chain, reconstruct_fields, NaiveModel  # noqa: E402

_P = {}


def prob():
    if "p" not in _P:
        _P["p"] = Problem(C.ProblemConfig())
    return _P["p"]


def cfg(**kw):
    base = C.build(C.NaiveConfig, quick=True)
    return replace(base, log_every=10 ** 9, **kw)


def test_resume_is_exact():
    p = prob()
    c_full = cfg(n_iter=1000, n_burn=400, checkpoint_every=200)
    c_half = replace(c_full, n_iter=600)
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.npz"), os.path.join(d, "b.npz")
        run_chain(p, c_full, 3, a, log=lambda m: None)
        run_chain(p, c_half, 3, b, log=lambda m: None)        # "interrupted" at 600
        run_chain(p, c_full, 3, b, log=lambda m: None)        # resumed to 1000
        za, zb = np.load(a), np.load(b)
        for k in ("acc", "phi", "relerr", "logbeta", "rho", "kept_theta", "u"):
            assert np.array_equal(za[k], zb[k]), k


def test_global_pcn_samples_prior_without_data():
    """Data term off: acceptance 1, u ~ N(0, I) (checks the move itself)."""
    p = prob()
    c = cfg(n_iter=4000, n_burn=0, thin=1, s0=0.5)
    m = NaiveModel(p, c)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(p.r + 2)
    us = []
    for it in range(c.n_iter):
        u = np.sqrt(1 - c.s0 ** 2) * u + c.s0 * rng.standard_normal(p.r + 2)
        us.append(u.copy())
    us = np.array(us[500:])
    assert abs(us.mean()) < 0.1 and abs(us.std() - 1) < 0.05
    th, lb, rho = m.unpack(np.zeros(p.r + 2))
    assert lb == c.mu_logbeta and abs(rho - np.exp(c.mu_logrho)) < 1e-15


def test_reconstruction_matches_saved_observations():
    p = prob()
    c = cfg(n_iter=300, n_burn=100, thin=50, checkpoint_every=300)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "c.npz")
        run_chain(p, c, 0, f, log=lambda m: None)
        z = np.load(f)
        lk, pf = reconstruct_fields(p, z["kept_theta"], z["kept_logbeta"], z["kept_rho"])
        assert pf.shape == (len(z["kept_it"]), p.phys.N)
        pobs = pf[:, p.phys.obs_mask]
        assert np.abs(pobs - z["kept_pobs"]).max() < 1e-5 * np.abs(pobs).max()
        assert set(np.unique(z["acc"])) <= {0.0, 1.0}
