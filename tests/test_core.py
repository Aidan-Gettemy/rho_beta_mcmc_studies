"""
Core correctness checks (no pytest needed: python tests/run_tests.py).

  physics   scaling lemma p(cK) = p(K)/c and Keff(cK) = c Keff(K);
            homogeneous plug has Keff = K exactly
  kle       r = 56 at 95% for n=16, ell=0.1; orthonormal modes
  default   32x32 grid: 512 chessboard observations, >= 3 cells per ell
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
        _P[mode] = Problem(C.ProblemConfig(truth_mode=mode, n=16))   # July grid
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


def test_default_grid_32():
    pc = C.ProblemConfig()
    assert pc.n == 32 and pc.n * pc.ell >= 3
    logs = []
    p = Problem(pc, log=logs.append)
    assert p.syn.n_obs == 512 and p.phys.N == 1024
    assert not any("WARNING" in m for m in logs)
    assert p.kle.method == "dense" and 50 <= p.r <= 65
    logs = []
    Problem(replace(pc, n=16), log=logs.append)          # 1.6 cells per ell
    assert any("WARNING" in m for m in logs)


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


def test_task_indices_from_launchers():
    """Array index, srun rank and mpirun rank all map to the right tasks."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments"))
    from _common import task_indices
    keep = {k: os.environ.pop(k) for k in list(os.environ)
            if k.startswith(("SLURM_", "OMPI_", "PMI", "PBS_"))}
    try:
        assert task_indices("7", 25) == [7]
        os.environ.update(SLURM_ARRAY_TASK_ID="3", SLURM_PROCID="0", SLURM_NTASKS="1")
        assert task_indices("env", 25) == [3]              # array wins
        del os.environ["SLURM_ARRAY_TASK_ID"]
        os.environ.update(SLURM_PROCID="2", SLURM_NTASKS="4")
        assert task_indices("env", 25) == [2, 6, 10, 14, 18, 22]
        os.environ.update(OMPI_COMM_WORLD_RANK="30", OMPI_COMM_WORLD_SIZE="64")
        assert task_indices("env", 25) == []               # idle rank
        assert task_indices("env", 128) == [30, 94]
        all_tasks = sorted(k for r in range(64) for k in range(r, 200, 64))
        assert all_tasks == list(range(200))               # full cover, no overlap
    finally:
        for k in list(os.environ):
            if k.startswith(("SLURM_", "OMPI_", "PMI", "PBS_")):
                del os.environ[k]
        os.environ.update(keep)
