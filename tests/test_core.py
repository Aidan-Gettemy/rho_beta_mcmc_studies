"""
Core correctness checks (no pytest needed: python tests/run_tests.py).

  physics   scaling lemma p(cK) = p(K)/c and Keff(cK) = c Keff(K);
            homogeneous plug has Keff = K exactly
  kle       r = 56 at 95% for n=16, ell=0.1; orthonormal modes
  truth     raw mode reproduces the July numbers (sigma_p, K*, level, Gbar)
  curve     beta_bar(0) = K*; ratio_of_mean <= mean_of_ratio (Jensen)
  sampler   likelihood-off chain reproduces the prior (ridge Jacobian check)
"""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import config as C                       # noqa: E402
from brho.problem import Problem                   # noqa: E402
from brho.curve import BetaCurve, sweep            # noqa: E402
from brho.samplers import ThreeStepSampler, StepAdapter  # noqa: E402

_P = {}


def prob(mode="standardized"):
    if mode not in _P:
        _P[mode] = Problem(C.ProblemConfig(truth_mode=mode))
    return _P[mode]


def test_scaling_lemma():
    p = prob()
    sc = p.phys.scaling_check(p.syn.K_true, c=5.3)
    assert sc["p_rel_err"] < 1e-12 and sc["keff_rel_err"] < 1e-12, sc


def test_homogeneous_keff():
    p = prob()
    K = np.full(p.phys.N, 3.7)
    assert abs(p.phys.keff(K, p.phys.solve(K)) / 3.7 - 1) < 1e-12


def test_kle():
    k = prob().kle
    assert k.r == 56, k.r
    assert k.method == "dense"
    assert np.allclose(k.T.T @ k.T, np.diag(k.lam), atol=1e-8)   # orthogonal modes
    th = np.random.default_rng(0).standard_normal(k.r)
    assert np.allclose(k.spatial_mean(th), k.field(th).mean())


def test_truth_raw_reproduces_july():
    s = prob("raw").syn
    assert abs(s.sigma_p - 3.61e-3) < 1e-5, s.sigma_p
    assert abs(s.Gbar - 0.286) < 1e-3, s.Gbar
    assert abs(s.level_true - 1.26) < 5e-3, s.level_true
    assert abs(s.keff_obs - 3.19) < 5e-3, s.keff_obs


def test_truth_standardized():
    s = prob().syn
    assert abs(s.Gbar) < 1e-12 and abs(s.G_std - 1) < 1e-12
    assert abs(s.level_true - np.log(2.3)) < 1e-12


def test_curve_anchor_and_order():
    p = prob()
    cfg = C.build(C.CurveConfig, quick=True)
    rhos, k1, _ = sweep(p, cfg)
    Ks = p.syn.keff_obs
    c_rm = BetaCurve(rhos, k1, "ratio_of_mean")
    c_mr = BetaCurve(rhos, k1, "mean_of_ratio")
    assert abs(c_rm.beta_bar(0.0, Ks) - Ks) < 1e-12
    assert np.all(c_rm.beta_bar(rhos, Ks) <= c_mr.beta_bar(rhos, Ks) + 1e-12)
    assert np.isnan(c_rm.beta_bar(cfg.rho_max + 0.1, Ks))


def test_sampler_prior_invariance():
    p = prob()
    rhos = np.linspace(0.05, 5, 6)
    curve = BetaCurve(rhos, np.ones((6, 4)))        # flat f: cheap, exact
    cfg = replace(C.SamplerConfig(), prior_test=True, s_init=(0.5, 0.5, 0.3))
    s = ThreeStepSampler(p, curve, cfg)
    rng = np.random.default_rng(1)
    st = s.init_state(rng)
    ad = StepAdapter(cfg.s_init, targets=cfg.targets, every=10 ** 9)
    th, w = [], []
    for it in range(30000):
        s.step(st, ad, rng)
        if it > 1000 and it % 5 == 0:
            th.append(st["theta"].copy()); w.append(st["w"])
    th, w = np.array(th), np.array(w)
    frob = np.linalg.norm(np.cov(th.T) - np.eye(p.r)) / np.sqrt(p.r)
    assert frob < 0.15, frob
    assert abs(np.mean((th ** 2).sum(1)) / p.r - 1) < 0.05
    assert abs(np.mean(w) + 0.041) < 0.08 and abs(np.std(w) - 0.955) < 0.08
