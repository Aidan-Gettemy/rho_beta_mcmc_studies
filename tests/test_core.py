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


def test_relative_likelihood():
    """sigma_rel = eps / sqrt(n_eff); Phi = relerr^2 / (2 sigma_rel^2);
    the floor at the truth is nearly grid-independent; absolute is exact July."""
    assert abs(C.sigma_rel_from(0.02, 128) - 0.02 / np.sqrt(128)) < 1e-15
    assert abs(C.eps_from(C.sigma_rel_from(0.02, 128), 128) - 0.02) < 1e-15
    assert abs(C.sigma_rel_from(0.02, 128) - C.sigma_rel_from(0.04, 512)) < 1e-15
    floors = {}
    for n in (16, 32):
        rel = Problem(C.ProblemConfig(n=n), log=lambda m: None)
        ab = Problem(C.ProblemConfig(n=n, likelihood="absolute"), log=lambda m: None)
        d = rel.syn.d_p
        K = np.exp(rel.syn.logK_true + 0.05)            # some other field
        phi_r, pobs, _ = rel.potential(K)
        relerr = np.linalg.norm(d - pobs) / np.linalg.norm(d)
        assert abs(phi_r - relerr ** 2 / (2 * rel.sigma_rel ** 2)) < 1e-9 * phi_r
        phi_a, _, _ = ab.potential(K)
        assert abs(phi_a - 0.5 * np.sum(((d - pobs) / ab.syn.sigma_p) ** 2)) < 1e-9 * phi_a
        assert ab.noise_floor() == ab.syn.n_obs / 2
        floors[n] = rel.noise_floor()
    assert abs(floors[32] / floors[16] - 1) < 0.25       # absolute: x4 (128 -> 512)
    # sigma_rel given directly wins over the pair
    p = Problem(C.ProblemConfig(n=16, sigma_rel=1e-3, rel_eps=0.5), log=lambda m: None)
    assert p.sigma_rel == 1e-3


def test_old_manifests_keep_absolute_likelihood():
    old = {k: v for k, v in C.to_dict(C.ProblemConfig(n=16))["ProblemConfig"].items()
           if k not in ("likelihood", "sigma_rel", "rel_eps", "rel_n_eff")}
    assert C.from_dict(C.ProblemConfig, old).likelihood == "absolute"
    new = C.to_dict(C.ProblemConfig())["ProblemConfig"]
    assert C.from_dict(C.ProblemConfig, new).likelihood == "relative"


def test_truth_kle_mode_is_a_draw_in_the_basis():
    """truth_mode "kle": the true G lies exactly in the sampler's KLE span
    (no truncation floor), is not standardized, and carries theta_true."""
    p = prob("kle")
    syn, kle = p.syn, p.kle
    assert syn.theta_true is not None and len(syn.theta_true) == p.r
    assert np.allclose(kle.field(syn.theta_true), syn.G_true)
    th = kle.project(syn.G_true)
    assert np.linalg.norm(kle.field(th) - syn.G_true) < 1e-8 * np.linalg.norm(syn.G_true)
    assert np.allclose(syn.logK_true, np.log(2.3) + 1.5 * syn.G_true)
    assert "theta_norm2_over_r" in syn.summary()
    # "kle" = the standardized truth projected: nearly the same field
    q = prob()
    assert abs(syn.G_true.mean()) < 0.05
    assert np.linalg.norm(syn.G_true - q.syn.G_true) < 0.5 * np.linalg.norm(q.syn.G_true)


def _soft_curve():
    """A curve whose realization scatter grows with rho (exercises s(rho))."""
    rng = np.random.default_rng(3)
    rhos = np.linspace(0.05, 5, 12)
    z = rng.standard_normal(64)
    keff1 = np.exp(-0.4 * rhos[:, None] ** 2 + 0.3 * rhos[:, None] * z[None, :])
    return BetaCurve(rhos, keff1)


def test_soft_sampler_prior_invariance():
    """E5 with Phi = 0 must sample the prior exactly: theta ~ N(0, I),
    w ~ N(0,1) truncated at rho_max, xi_b = (log beta - m(rho))/s(rho) ~ N(0,1),
    independent.  Checks the ridge Jacobian with a rho-dependent s(rho) and
    the level-swap move (moves C and D carry all the non-pCN terms)."""
    from brho.samplers import SoftCurveSampler
    p = prob()
    # small B step and a ridge step ~ 1/sqrt(r): C does much of the w-moving,
    # so an error in its ratio shows up in the law of (w, xi_b)
    cfg = replace(C.SoftSamplerConfig(), prior_test=True,
                  s_init=(0.5, 0.3, 0.12, 0.6))
    s = SoftCurveSampler(p, _soft_curve(), cfg)
    rng = np.random.default_rng(2)
    st = s.init_state(rng)
    ad = StepAdapter(cfg.s_init, targets=cfg.targets, every=10 ** 9)
    th, w, xi = [], [], []
    for it in range(60000):
        s.step(st, ad, rng)
        if it > 2000 and it % 5 == 0:
            th.append(st["theta"].copy()); w.append(st["w"])
            xi.append(s.extra_traces(st)["xi_b"])
    th, w, xi = np.array(th), np.array(w), np.array(xi)
    frob = np.linalg.norm(np.cov(th.T) - np.eye(p.r)) / np.sqrt(p.r)
    assert frob < 0.15, frob
    assert abs(np.mean((th ** 2).sum(1)) / p.r - 1) < 0.05
    assert abs(np.mean(w) + 0.041) < 0.08 and abs(np.std(w) - 0.955) < 0.04
    # tolerance 0.05 on sd(xi_b): dropping the curve-prior ratio from move C
    # gives 1.07, from move D 1.2 (checked by mutation)
    assert abs(np.mean(xi)) < 0.06 and abs(np.std(xi) - 1) < 0.05, (xi.mean(), xi.std())
    assert abs(np.corrcoef(w, xi)[0, 1]) < 0.08
    acc = ad.acc  # not reset (every=1e9): acceptance counts of the 4 moves
    assert all(a > 0 for a in acc)


def test_soft_sampler_moves_c_and_d_preserve_logK():
    """C leaves log K exactly unchanged; D changes it only by the part of the
    constant field outside the KLE span."""
    from brho.samplers import SoftCurveSampler
    p = prob()
    s = SoftCurveSampler(p, _soft_curve(), C.SoftSamplerConfig())
    rng = np.random.default_rng(0)
    th, w, lb = rng.standard_normal(p.r), 0.3, 0.7
    rho = s.rho(w)
    logK = lb + rho * p.kle.field(th)
    eta = 0.2
    logK_c = lb + s.rho(w + eta / s.cfg.s_rho) * p.kle.field(np.exp(-eta) * th)
    assert np.allclose(logK_c, logK)
    d = 0.3
    logK_d = lb + d + rho * p.kle.field(th - d / rho * s.c)
    assert np.sqrt(np.mean((logK_d - logK) ** 2)) <= d * (s.c_resid + 1e-12) * 1.0001
    assert s.c_resid < 0.1
