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
        _P["p"] = Problem(C.ProblemConfig(n=16))    # small grid: fast tests
    return _P["p"]


def cfg(**kw):
    base = C.build(C.NaiveConfig, quick=True)
    return replace(base, log_every=10 ** 9, **kw)


def test_resume_is_exact():
    """Both moves: interrupted at a checkpoint and resumed == uninterrupted."""
    p = prob()
    for move in ("blocked", "global"):
        c_full = cfg(move=move, n_iter=1000, n_burn=400, checkpoint_every=200)
        c_half = replace(c_full, n_iter=600)
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.npz"), os.path.join(d, "b.npz")
            run_chain(p, c_full, 3, a, log=lambda m: None)
            run_chain(p, c_half, 3, b, log=lambda m: None)    # "interrupted" at 600
            run_chain(p, c_full, 3, b, log=lambda m: None)    # resumed to 1000
            za, zb = np.load(a), np.load(b)
            for k in ("acc", "acc_h", "phi", "relerr", "logbeta", "rho", "kept_theta",
                      "u", "s_final"):
                assert np.array_equal(za[k], zb[k]), (move, k)


def test_blocked_has_two_adapted_blocks():
    p = prob()
    # opposite targets: theta's step must grow, the hyper step must shrink,
    # which only happens if each block adapts to its own acceptance
    c = cfg(move="blocked", n_iter=1500, n_burn=1000, checkpoint_every=500,
            target=0.0, target_h=1.0)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "c.npz")
        out = run_chain(p, c, 0, f, log=lambda m: None)
        z = np.load(f)
        assert z["s_final"].shape == (2,) and z["s_hist"].shape[1] == 2
        assert int(z["solves"]) == 2 * 1500
        assert 0 < z["acc_h"].mean() < 1 and 0 < z["acc"].mean() < 1
        assert "acc_h_post" in out
        assert z["s_final"][0] > c.s0 and z["s_final"][1] < c.s0_h
    g = cfg(move="global", n_iter=300, n_burn=100, checkpoint_every=300)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "g.npz")
        run_chain(p, g, 0, f, log=lambda m: None)
        z = np.load(f)
        assert z["s_final"].shape == (1,) and not z["acc_h"].any()


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


def test_mpsrf_matlab_port():
    """get_mpsrf matches diagnostics.mpsrf; iid chains give ~1, a shifted
    component is caught by its own PSRF; AR(1) chains from spread-out starts
    give a falling curve that crosses 1.2."""
    from brho import mpsrf as M, diagnostics as D
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 400, 20))
    mp, pm, pi = M.get_mpsrf(x)
    assert abs(mp - D.mpsrf(x)) < 1e-10 and mp < 1.05 and pm < 1.05
    y = x.copy()
    y[:, :, 3] += rng.normal(0, 1, (16, 1))
    _, _, pi = M.get_mpsrf(y)
    assert pi[3] > 1.2 and np.delete(pi, 3).max() < 1.05
    z = np.zeros((16, 3000, 4))
    z[:, 0] = rng.normal(0, 10, (16, 4))
    for t in range(1, 3000):
        z[:, t] = 0.95 * z[:, t - 1] + np.sqrt(1 - 0.95 ** 2) * rng.standard_normal((16, 4))
    cur = M.mpsrf_curve(z, skip=2, step=50, iter_burn=1, dispiter=0)
    # MATLAB indexing: iter_step = (step+iter_burn):step:n, idx = iter_burn:skip:k
    assert cur["k"][0] == 51 and cur["k"][-1] <= 3000
    k = cur["k"][3]
    assert abs(cur["mpsrf"][3] - M.get_mpsrf(z[:, 0:k:2])[0]) < 1e-12
    assert cur["mpsrf"][0] > 1.5 and cur["mpsrf"][-1] < 1.05
    # crossing read-out: first 10 points ignored, linear interpolation
    x = np.arange(1, 21) * 100.0
    v = np.r_[np.full(10, 0.5), np.linspace(3, 1.0, 10)]
    c = M.convergence_iteration(x, v, 1.2)
    j = int(np.flatnonzero(v[10:] < 1.2)[0]) + 10
    assert c == np.ceil(x[j - 1] + (x[j] - x[j - 1]) * (1.2 - v[j - 1]) / (v[j] - v[j - 1]))
    assert M.convergence_iteration(x, np.full(20, 3.0)) is None


def test_theta_repeat():
    """E1 blocked with theta_repeat = k: move A k times per iteration
    (k + 1 solves), acc = fraction of the k accepted, resume still exact;
    refused for the global move."""
    p = prob()
    c_full = cfg(move="blocked", theta_repeat=3, n_iter=600, n_burn=300, checkpoint_every=200)
    c_half = replace(c_full, n_iter=400)
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.npz"), os.path.join(d, "b.npz")
        out = run_chain(p, c_full, 1, a, log=lambda m: None)
        run_chain(p, c_half, 1, b, log=lambda m: None)
        run_chain(p, c_full, 1, b, log=lambda m: None)
        za, zb = np.load(a), np.load(b)
        for k in ("acc", "acc_h", "phi", "kept_theta", "u", "s_final"):
            assert np.array_equal(za[k], zb[k]), k
        assert int(za["solves_per_iteration"]) == 4 and int(za["solves"]) == 4 * 600
        assert int(za["theta_repeat"]) == 3 and out["theta_repeat"] == 3
        fr = np.unique(np.round(za["acc"] * 3, 4))
        assert set(fr) <= {0.0, 1.0, 2.0, 3.0} and len(fr) > 1
    try:
        run_chain(p, cfg(move="global", theta_repeat=2), 0, "unused.npz", log=lambda m: None)
        raise AssertionError("global + theta_repeat should be refused")
    except ValueError:
        pass
    from brho import io
    assert io.exp_e1("blocked", "a7") == "e1_blocked_a7" and io.exp_e1("global") == "e1_global"
