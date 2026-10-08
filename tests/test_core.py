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


def test_sampler_ab_ablation():
    """moves="AB": move C is never proposed, A+B alone still keep the prior."""
    p = prob()
    rhos = np.linspace(0.05, 5, 6)
    curve = BetaCurve(rhos, np.ones((6, 4)))
    cfg = replace(C.SamplerConfig(), prior_test=True, s_init=(0.5, 0.5, 0.3), moves="AB")
    s = ThreeStepSampler(p, curve, cfg)
    rng = np.random.default_rng(2)
    st = s.init_state(rng)
    ad = StepAdapter(cfg.s_init, targets=cfg.targets, every=10 ** 9)
    w = []
    for it in range(20000):
        s.step(st, ad, rng)
        if it > 1000 and it % 5 == 0:
            w.append(st["w"])
    assert ad.n[2] == 0 and ad.n[0] == 20000 and ad.n[1] == 20000
    assert abs(np.mean(w)) < 0.1 and abs(np.std(w) - 1) < 0.1
    try:
        ThreeStepSampler(p, curve, replace(cfg, moves="AC"))
        raise AssertionError("moves='AC' should be rejected")
    except ValueError:
        pass
    from brho import io as _io
    assert _io.exp_sampler("ab") == "e4_sampler_ab" and _io.exp_sampler() == "e4_sampler"


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


def test_spectrum_check_recovers_anisotropy_and_slow_directions():
    """reports/report_spectrum.spectrum on synthetic chains: a rotated
    Gaussian with sds from 0.01 to 1, AR(1) chains that are slow in the broad
    directions (as pCN is), started over-dispersed.  It must recover the sds,
    find R growing with sd, and flag the slow direction aligned with c."""
    import importlib
    rep = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reports")
    saved = {k: sys.modules.pop(k) for k in ("_common",) if k in sys.modules}
    sys.path.insert(0, rep)
    try:                               # reports/ has its own _common: isolate it
        spectrum = importlib.import_module("report_spectrum").spectrum
    finally:
        sys.path.remove(rep)
        sys.modules.pop("_common", None)
        sys.modules.pop("report_spectrum", None)
        sys.modules.pop("report_mpsrf", None)
        sys.modules.update(saved)
    rng = np.random.default_rng(1)
    r, m, n = 12, 16, 400
    sd = np.logspace(0, -2, r)                              # broadest first
    Q, _ = np.linalg.qr(rng.standard_normal((r, r)))
    phi = 1 - 0.002 / sd                                    # slow when broad
    z = np.empty((m, n, r))
    z[:, 0] = 3 * sd * rng.standard_normal((m, r))          # over-dispersed start
    for t in range(1, n):
        z[:, t] = phi * z[:, t - 1] + sd * np.sqrt(1 - phi ** 2) * rng.standard_normal((m, r))
    res = spectrum(z @ Q.T, c=3.0 * Q[:, 0], step=0.01)
    s = res["summary"]
    assert np.allclose(res["sd"][r // 2:], sd[r // 2:], rtol=0.2)   # informed dirs
    assert s["sd_ratio"] > 20
    assert s["spearman_R_vs_sd"] > 0.6
    assert s["c_direction"] == 0 and s["c_direction_cos"] > 0.9
    assert s["R_argmax"] <= 2 and abs(s["step_over_sd_min"] - 1) < 0.3
    # conditional part: theta = u * (lb / 0.3) + tiny noise along u, so the
    # marginal width along u is ~1 but the width given lb is 0.02
    u = Q[:, 0]
    lb = 0.3 * rng.standard_normal((m, n, 1))
    th2 = rng.standard_normal((m, n, r)) * 0.5
    th2 -= (th2 @ u)[..., None] * u
    th2 += (lb / 0.3 + 0.02 * rng.standard_normal((m, n, 1))) * u
    hyp = np.concatenate([lb, rng.standard_normal((m, n, 1))], axis=2)
    s2 = spectrum(th2, step=0.01, hyper=hyp, u_level=u)["summary"]
    assert 0.9 < s2["sd_marg_along_level"] < 1.1
    assert 0.015 < s2["sd_cond_along_level"] < 0.025
    assert s2["cos_level_tightest_cond"] > 0.99


def test_soft_sampler_extension_is_exact():
    """E5 run to n_total, then continued to a larger n_total (what
    experiments/extend_run.py does) == one uninterrupted run."""
    import tempfile
    from brho.samplers import SoftCurveSampler, run_chunk
    p = prob()
    full = replace(C.SoftSamplerConfig(), n_total=1200, n_burn=300, thin=20, chunk=500)
    short = replace(full, n_total=700)
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.npz"), os.path.join(d, "b.npz")
        for cfg_, path in ((full, a), (short, b), (full, b)):
            s = SoftCurveSampler(p, _soft_curve(), cfg_)
            it = 0
            while it < cfg_.n_total:
                it = run_chunk(s, 2, path)
        za, zb = np.load(a), np.load(b)
        assert int(za["it"]) == int(zb["it"]) == 1200
        for k in za.files:
            if k != "rng_state":
                assert np.array_equal(za[k], zb[k]), k
        assert za["rng_state"] == zb["rng_state"]


def test_chain_files_skips_checkpoints_being_written():
    """A report run while a job saves a checkpoint must not trip over the
    temporary file (chain45.npz.tmp.npz), and chains sort numerically."""
    import tempfile
    from brho import io
    with tempfile.TemporaryDirectory() as d:
        for name in ("chain10.npz", "chain2.npz", "chain45.npz.tmp.npz", "chain45.npz",
                     "chainX.npz", "chains_summary.json"):
            open(os.path.join(d, name), "w").close()
        got = [os.path.basename(p) for p in io.chain_files(d)]
        assert got == ["chain2.npz", "chain10.npz", "chain45.npz"], got


def test_multiscale_whitening_is_exact():
    """brho/multiscale.py: W Sigma_ms W^T + V0 V0^T = I to machine precision;
    whitened draws have covariance I, raw draws do not (Pereira, Sec. 2)."""
    from brho.multiscale import from_problem
    p = prob()
    ms = from_problem(p, blocks=2, local_energy=0.95)
    assert ms.operator_check < 1e-12
    rng = np.random.default_rng(4)
    X = np.array([ms.draw("ms_white", rng) for _ in range(20000)])
    R = np.array([ms.draw("ms_raw", rng) for _ in range(20000)])
    ref = np.linalg.norm(np.cov(rng.standard_normal((20000, p.r)).T) - np.eye(p.r)) / np.sqrt(p.r)
    err_w = np.linalg.norm(np.cov(X.T) - np.eye(p.r)) / np.sqrt(p.r)
    err_r = np.linalg.norm(np.cov(R.T) - np.eye(p.r)) / np.sqrt(p.r)
    assert err_w < 2 * ref + 0.01, (err_w, ref)
    assert err_r > 5 * ref, (err_r, ref)
    assert np.allclose(ms.draw("global", np.random.default_rng(0)),
                       np.random.default_rng(0).standard_normal(p.r))


def test_soft_sampler_multiscale_prior_test():
    """E7: the full E5 sampler (moves A-D) with Phi = 0 and the multiscale
    theta innovation: whitened reproduces theta ~ N(0, I); raw does not."""
    from brho.samplers import SoftCurveSampler
    p = prob()
    frob = {}
    for innov in ("ms_white", "ms_raw"):
        cfg = replace(C.SoftSamplerConfig(), prior_test=True, theta_innovation=innov,
                      s_init=(0.5, 0.3, 0.12, 0.6))
        s = SoftCurveSampler(p, _soft_curve(), cfg)
        rng = np.random.default_rng(2)
        st = s.init_state(rng)
        ad = StepAdapter(cfg.s_init, targets=cfg.targets, every=10 ** 9)
        th = []
        for it in range(30000):
            s.step(st, ad, rng)
            if it > 2000 and it % 5 == 0:
                th.append(st["theta"].copy())
        th = np.array(th)
        frob[innov] = np.linalg.norm(np.cov(th.T) - np.eye(p.r)) / np.sqrt(p.r)
    assert frob["ms_white"] < 0.2, frob
    assert frob["ms_raw"] > 0.3, frob


def test_soft_sampler_global_innovation_unchanged():
    """The default ("global") theta move must be bit-identical to the code
    that produced the E5 runs, so those runs extend exactly."""
    from brho.samplers import SoftCurveSampler, pcn
    p = prob()
    cfg = replace(C.SoftSamplerConfig(), prior_test=True)
    s = SoftCurveSampler(p, _soft_curve(), cfg)
    assert s.ms is None
    th = np.random.default_rng(1).standard_normal(p.r)
    a = pcn(th, 0.3, np.random.default_rng(7))
    b = np.sqrt(1 - 0.09) * th + 0.3 * np.random.default_rng(7).standard_normal(p.r)
    assert np.array_equal(a, b)


def test_logK_summaries_match_the_full_matrix():
    """The chunked pointwise summaries equal those of the full sample x cell
    matrix (reports use them so long runs fit in memory)."""
    from brho import diagnostics as D
    p = prob()
    rng = np.random.default_rng(5)
    n = 900
    th = rng.standard_normal((n, p.r))
    lb, rho = rng.normal(0.8, 0.3, n), np.exp(rng.normal(0.3, 0.2, n))
    full = lb[:, None] + rho[:, None] * p.kle.field(th)
    truth = full[0] + 0.1
    fs = D.logK_summaries(lb, rho, th, p.kle, truth, cells_per_chunk=37)
    assert np.allclose(fs["mean"], full.mean(0)) and np.allclose(fs["sd"], full.std(0))
    lo, hi = np.percentile(full, [5, 95], axis=0)
    assert np.allclose(fs["lo"], lo) and np.allclose(fs["hi"], hi)
    assert fs["coverage"] == D.pointwise_coverage(full, truth)
    assert np.allclose(lb + rho * p.kle.spatial_mean(th), full.mean(1))
def test_pointwise_moments_exact():
    """Pointwise mean/variance of G = T theta from the moments of theta equal
    the brute-force values over the sample fields."""
    from brho import diagnostics as D
    p = prob()
    th = np.random.default_rng(6).standard_normal((3, 400, p.r)) * 0.7 + 0.1
    pm, pv = D.pointwise_moments(th, p.kle.T)
    G = p.kle.field(th.reshape(-1, p.r))
    assert np.allclose(pm, G.mean(0)) and np.allclose(pv, G.var(0))


_P3 = {}


def prob3():
    """A small 3D permeameter: 8^3 cells, ell = 0.4 (3.2 cells per ell, the
    study's ratio), truth in the KLE (as in the 3D study)."""
    if "p" not in _P3:
        _P3["p"] = Problem(C.ProblemConfig(dim=3, n=8, ell=0.4, truth_mode="kle",
                                           kle_method="nystrom_sep", kle_nq=24,
                                           sigma_rel=2e-3))
    return _P3["p"]


def test_3d_problem_builds_and_solves():
    """--dim 3: unit cube, unit influx at x = 0, p = 0 at x = 1, no flow on
    the side faces; chessboard sensors (half the cells); truth in the KLE.
    The flux boundary condition makes the pressures scale as 1/K (Lemma 2)."""
    p = prob3()
    assert p.phys.dim == 3 and p.phys.N == 512
    assert p.syn.n_obs == 256 and p.kle.T.shape[0] == 512
    assert abs(p.kle.energy - 0.95) < 0.03
    K = p.syn.K_true
    p1 = p.phys.solve(K)
    assert np.allclose(p.phys.solve(3.0 * K), p1 / 3.0, rtol=1e-8, atol=1e-12)
    th = p.kle.project(p.syn.G_true)
    assert np.allclose(p.kle.field(th), p.syn.G_true, atol=1e-8)   # truth lies in the KLE
    phi, pobs, keff = p.potential(K)
    assert np.isfinite(phi) and pobs.shape == (256,) and keff > 0


def test_2d_store_keys_unchanged_by_dim():
    """Adding ProblemConfig.dim must not change the keys of existing 2D
    stores (runs/cluster_kle, the July runs)."""
    from brho.problem import kle_key, truth_key
    pc = C.ProblemConfig(truth_mode="kle")
    assert kle_key(pc, 1024)["grid"] == [32, 32]
    assert truth_key(pc)["grid"] == [32, 32]
    assert kle_key(C.ProblemConfig(dim=3, n=16), 4096)["grid"] == [16, 16, 16]


def test_3d_soft_sampler_and_multiscale():
    """E5 steps run in 3D with the likelihood on; the 2x2x2 multiscale
    whitening is exact; the E5 prior test with ms_white keeps theta ~ N(0, I)."""
    from brho.samplers import SoftCurveSampler
    from brho.multiscale import from_problem
    p = prob3()
    ms = from_problem(p, blocks=2)
    assert ms.dim == 3 and len(ms.r_k) == 8 and ms.operator_check < 1e-12
    cfg = replace(C.SoftSamplerConfig(), theta_innovation="ms_white")
    s = SoftCurveSampler(p, _soft_curve(), cfg)
    rng = np.random.default_rng(3)
    st = s.init_state(rng)
    ad = StepAdapter(cfg.s_init, targets=cfg.targets, every=10 ** 9)
    for _ in range(20):
        s.step(st, ad, rng)
    assert np.isfinite(st["phi"])
    cfg0 = replace(cfg, prior_test=True, s_init=(0.5, 0.3, 0.12, 0.6))
    s0 = SoftCurveSampler(p, _soft_curve(), cfg0)
    st = s0.init_state(rng)
    ad = StepAdapter(cfg0.s_init, targets=cfg0.targets, every=10 ** 9)   # its own steps
    th = []
    for it in range(12000):
        s0.step(st, ad, rng)
        if it > 1000 and it % 5 == 0:
            th.append(st["theta"].copy())
    th = np.array(th)
    frob = np.linalg.norm(np.cov(th.T) - np.eye(p.r)) / np.sqrt(p.r)
    ref = np.linalg.norm(np.cov(rng.standard_normal(th.shape).T) - np.eye(p.r)) / np.sqrt(p.r)
    assert frob < 2.5 * ref, (frob, ref)


def test_3d_plots_show_the_mid_plane():
    from brho import plotting as P
    import matplotlib.pyplot as plt
    p = prob3()
    img, note = P.plane(p.syn.logK_true, 8)
    assert img.shape == (8, 8) and "z=" in note
    assert np.allclose(img, p.syn.logK_true.reshape(8, 8, 8)[:, :, 4])
    assert P.line(p.syn.logK_true, 8, 3).shape == (8,)
    fig, ax = plt.subplots()
    P.field(ax, p.syn.p_true, 8, "p", obs_mask=p.syn.obs_mask)
    assert "z=" in ax.get_title()
    plt.close(fig)


# ------------------------------------------------------------ E8: block pCN
def test_block_subspaces_geometry():
    """Local subspaces: orthonormal, every direction keeps >= energy of its
    field inside its block, independent of the eigensolver basis (only the
    projector matters); coarse + primal locals span R^r; 2x2 -> 3x3 dual."""
    from brho.blockpcn import from_problem, local_subspace
    p = prob()
    bp = from_problem(p, blocks=2, energy=0.9)
    s = bp.summary()
    assert s["spans"] and len(s["primal_dims"]) == 4 and len(s["dual_dims"]) == 9
    assert bp.n_local == 13 and bp.solves("sweep") == 14 and bp.solves("random") == 2
    T = p.kle.T
    for mv in bp.moves[1:]:
        V = mv["V"]
        assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-10)
        F = T @ V
        inside = (F[mv["mask"]] ** 2).sum(0) / (F ** 2).sum(0)
        assert inside.min() >= 0.9 - 1e-9, (mv["name"], inside.min())
    V0 = bp.moves[0]["V"]
    for mv in bp.moves[1:5]:                       # coarse is orthogonal to primal
        assert np.abs(V0.T @ mv["V"]).max() < 1e-10
    # symmetric 2x2 blocks have equal local dimensions
    assert len(set(s["primal_dims"])) == 1
    # a rotated copy of the Gram matrix gives the same projector
    m = bp.moves[1]["mask"]
    Q1, _ = local_subspace(T, m, 0.9)
    R = np.linalg.qr(np.random.default_rng(0).standard_normal((p.r, p.r)))[0]
    Q2, _ = local_subspace(T @ R, m, 0.9)
    assert np.allclose(Q1 @ Q1.T, R @ Q2 @ Q2.T @ R.T, atol=1e-8)


def test_block_moves_preserve_the_prior_exactly():
    """Started at exact prior draws, any number of block sweeps (sweep or
    random schedule) leaves theta ~ N(0, I): the law at the end matches
    direct draws (no mixing needed for this check)."""
    from brho.samplers import SoftCurveSampler
    p = prob()
    for innov in ("block", "block_random"):
        cfg = replace(C.SoftSamplerConfig(), prior_test=True, theta_innovation=innov,
                      s_init=(0.5, 0.3, 0.12, 0.6), block_s0=0.6)
        s = SoftCurveSampler(p, _soft_curve(), cfg)
        rng = np.random.default_rng(5)
        out = []
        for c in range(1500):
            st = s.init_state(rng)
            ad = StepAdapter(s.s_init, targets=s.targets, every=10 ** 9)
            for _ in range(15):
                s.step(st, ad, rng)
            out.append(st["theta"])
        th = np.array(out)
        frob = np.linalg.norm(np.cov(th.T) - np.eye(p.r)) / np.sqrt(p.r)
        ref = np.linalg.norm(np.cov(rng.standard_normal(th.shape).T) - np.eye(p.r)) / np.sqrt(p.r)
        assert frob < 1.3 * ref, (innov, frob, ref)
        assert np.abs(th.mean(0)).max() < 5 / np.sqrt(len(th)), innov


def test_block_sampler_moves_and_adaptation():
    """E8 bookkeeping: one step slot per block after A-D, solves per
    iteration, per-proposal adaptation (a move not proposed in a window
    keeps its step), likelihood-on steps run and adapt."""
    from brho.samplers import SoftCurveSampler
    p = prob()
    cfg = replace(C.SoftSamplerConfig(), theta_innovation="block_random")
    s = SoftCurveSampler(p, _soft_curve(), cfg)
    assert len(s.s_init) == 4 + 13 and s.solves_per_iteration == 4
    assert s.move_names[:4] == ("A0 coarse", "B", "C", "D") and s.move_names[4] == "P1"
    ad = StepAdapter(s.s_init, targets=s.targets, every=5)
    rng = np.random.default_rng(1)
    st = s.init_state(rng)
    for it in range(5):
        s.step(st, ad, rng)
        ad.tick(it, 10 ** 6)
    assert np.isfinite(st["phi"])
    moved = ad.s != np.array(s.s_init)
    assert moved[:4].all()                         # coarse, B, C, D adapted every window
    assert 1 <= moved[4:].sum() <= 5               # only the blocks that were proposed
    # old samplers: one proposal per move per iteration -> old rate acc / win
    ad2 = StepAdapter([0.1, 0.1], targets=[0.25, 0.25], every=4)
    for it, acc in enumerate([1, 1, 0, 1]):
        ad2.record(0, acc); ad2.record(1, 0)
        ad2.tick(it, 10)
    assert np.allclose(ad2.s, [0.112, 0.1 / 1.12])


def test_block_run_chunk_checkpoints_block_steps():
    """run_chunk with E8: the step vector has one entry per move, proposals
    are counted, and a resumed chunk continues the same chain."""
    import tempfile
    from brho.samplers import SoftCurveSampler, run_chunk
    p = prob()
    cfg = replace(C.SoftSamplerConfig(), prior_test=True, theta_innovation="block",
                  n_total=40, n_burn=20, thin=5, chunk=20, adapt_every=10)
    s = SoftCurveSampler(p, _soft_curve(), cfg)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "chain0.npz")
        assert run_chunk(s, 0, path) == 20
        assert run_chunk(s, 0, path) == 40
        z = np.load(path)
        assert z["s"].shape == (17,) and z["acc_tot_prop"].shape == (17,)
        assert np.all(z["acc_tot_prop"] == 40)
        assert z["kept_theta"].shape == (4, p.r)


def test_3d_block_subspaces():
    """3D: 2x2x2 primal and 3x3x3 dual blocks, coarse + primal span R^r."""
    from brho.blockpcn import from_problem
    p = prob3()
    bp = from_problem(p, blocks=2, energy=0.9)
    s = bp.summary()
    assert s["dim"] == 3 and len(s["primal_dims"]) == 8 and len(s["dual_dims"]) == 27
    assert s["spans"]
    # small 3D dual blocks can be empty at energy 0.9: no move, no solve
    assert all(mv["dim"] > 0 for mv in bp.moves[1:])
    assert bp.n_local == 35 - s["empty_blocks"] and bp.solves("sweep") == bp.n_local + 1
def test_multiscale_operator_independent_of_eigh_basis():
    """Local spectra on square/cube subdomains have exact ties; Sigma_ms must
    not depend on which basis eigh returns inside a tied eigenspace (it varies
    with machine and threads).  At the 3D study size (16^3, ell = 0.2, 2x2x2)
    the 95% energy cut falls inside a tied group: rebuilding with the
    eigenvectors randomly rotated inside every tied group must leave Sigma_ms
    unchanged with the tie-closed rule, and does change it with the plain
    energy cut (the rule used before 2026-10-01)."""
    from brho import multiscale as MSmod
    from brho.fv import Permeameter
    from brho.kle import build_kle
    ph = Permeameter((16, 16, 16))
    kp = dict(ell=0.2, sigma2=1.0)
    kle = build_kle(ph.points, ph.axes, "sqexp", kp, 0.95, "nystrom_sep", 40)
    orig_eigh, orig_trunc = np.linalg.eigh, MSmod.local_truncation

    def rotated_eigh(A, *a, **k):
        w, v = orig_eigh(A, *a, **k)
        v = v.copy()
        rng = np.random.default_rng(int(1e6 * abs(w).sum()) % 2 ** 32)
        i = 0
        while i < len(w):
            j = i + 1
            while j < len(w) and abs(w[j] - w[i]) <= 1e-9 * abs(w[i]):
                j += 1
            if j - i > 1:
                Q, _ = np.linalg.qr(rng.standard_normal((j - i, j - i)))
                v[:, i:j] = v[:, i:j] @ Q
            i = j
        return w, v

    def plain(mu, energy, rtol=None):
        return int(np.searchsorted(np.cumsum(mu) / mu.sum(), energy) + 1)

    def build(rotate, rule):
        MSmod.local_truncation = rule
        if rotate:
            MSmod.np.linalg.eigh = rotated_eigh
        try:
            return MSmod.MSInnovation(kle, ph.points, "sqexp", kp, 2, 0.95)
        finally:
            MSmod.np.linalg.eigh = orig_eigh
            MSmod.local_truncation = orig_trunc

    a, b = build(False, orig_trunc), build(True, orig_trunc)
    assert a.r_k == [17] * 8 and a.operator_check < 1e-12
    assert np.allclose(a.Sigma, b.Sigma, atol=1e-10), np.abs(a.Sigma - b.Sigma).max()
    c, d = build(False, plain), build(True, plain)
    assert c.r_k == [16] * 8
    assert np.abs(c.Sigma - d.Sigma).max() > 1e-3      # the old rule is basis-dependent


def test_block_without_coarse_move():
    """block_coarse=False: no coarse move and no solve for it; allowed only
    when primal + dual span R^r (else ValueError); a dual halo raises the
    weakest direction's coverage; the moves still preserve N(0, I)."""
    from brho.blockpcn import from_problem
    from brho.samplers import SoftCurveSampler
    p = prob()
    try:
        bp = from_problem(p, 2, 0.9, coarse=False)
        assert bp.solves("sweep") == 13 and bp.moves[0]["dim"] == 0
        assert bp.summary()["spans"]
    except ValueError as err:                        # not spanning: refused, with a reason
        assert "irreducible" in str(err)
    bh = from_problem(p, 2, 0.9, coarse=False, dual_halo=0.0625)
    b0 = from_problem(p, 2, 0.9)
    assert bh.summary()["coverage_min"] > 0.1        # every direction well reached
    assert b0.summary()["coarse"] and not bh.summary()["coarse"]
    assert bh.dual_halo == 0.0625 and bh.halo == 0.0
    cfg = replace(C.SoftSamplerConfig(), prior_test=True, theta_innovation="block",
                  block_coarse=False, block_dual_halo=0.0625,
                  s_init=(0.5, 0.3, 0.12, 0.6), block_s0=0.6)
    s = SoftCurveSampler(p, _soft_curve(), cfg)
    assert s.solves_per_iteration == 13 + 2 and s.move_names[0] == "A0 (off)"
    rng = np.random.default_rng(9)
    out = []
    for c in range(1500):
        st = s.init_state(rng)
        ad = StepAdapter(s.s_init, targets=s.targets, every=10 ** 9)
        for _ in range(10):
            s.step(st, ad, rng)
        out.append(st["theta"])
        assert ad.n[0] == 0                          # the coarse slot is never proposed
    th = np.array(out)
    frob = np.linalg.norm(np.cov(th.T) - np.eye(p.r)) / np.sqrt(p.r)
    ref = np.linalg.norm(np.cov(rng.standard_normal(th.shape).T) - np.eye(p.r)) / np.sqrt(p.r)
    assert frob < 1.3 * ref, (frob, ref)
    from brho import io
    assert io.exp_soft(False, "block", "nocoarse") == "e5_soft_block_nocoarse"
    assert io.exp_soft(True, "block") == "e5_prior_test_block"


def test_theta_repeat_control():
    """theta_repeat = k: move A k times per iteration, 2 + k solves; k = 1 is
    the original path (same random stream); prior preserved for k > 1."""
    from brho.samplers import SoftCurveSampler
    p = prob()
    base = replace(C.SoftSamplerConfig(), prior_test=True, s_init=(0.5, 0.3, 0.12, 0.6))
    s1 = SoftCurveSampler(p, _soft_curve(), base)
    assert s1.theta_repeat == 1 and s1.solves_per_iteration == 3
    sk = SoftCurveSampler(p, _soft_curve(), replace(base, theta_repeat=4))
    assert sk.solves_per_iteration == 6
    # k = 1 reproduces the pre-option sampler exactly (same draws, same order)
    ra, rb = np.random.default_rng(4), np.random.default_rng(4)
    sa, sb = s1.init_state(ra), s1.init_state(rb)
    ad_a = StepAdapter(base.s_init, targets=base.targets, every=10 ** 9)
    ad_b = StepAdapter(base.s_init, targets=base.targets, every=10 ** 9)
    s1b = SoftCurveSampler(p, _soft_curve(), replace(base, theta_repeat=1))
    for _ in range(50):
        s1.step(sa, ad_a, ra); s1b.step(sb, ad_b, rb)
    assert np.array_equal(sa["theta"], sb["theta"]) and sa["w"] == sb["w"]
    # k = 4: 4 proposals of move A per iteration, prior preserved
    rng = np.random.default_rng(6)
    out = []
    for c in range(1200):
        st = sk.init_state(rng)
        ad = StepAdapter(sk.s_init, targets=sk.targets, every=10 ** 9)
        for _ in range(5):
            sk.step(st, ad, rng)
        assert ad.n[0] == 20 and ad.n[1] == 5
        out.append(st["theta"])
    th = np.array(out)
    frob = np.linalg.norm(np.cov(th.T) - np.eye(p.r)) / np.sqrt(p.r)
    ref = np.linalg.norm(np.cov(rng.standard_normal(th.shape).T) - np.eye(p.r)) / np.sqrt(p.r)
    assert frob < 1.3 * ref, (frob, ref)
    try:
        SoftCurveSampler(p, _soft_curve(), replace(base, theta_innovation="block", theta_repeat=2))
        raise AssertionError("block + theta_repeat should be refused")
    except ValueError:
        pass
    from brho import io
    assert io.exp_soft(False, "global", "a14") == "e5_soft_a14"


def test_learned_noise_level():
    """learn_noise: s2 = sigma_rel^2 gets an exact inverse-gamma Gibbs draw.
    Likelihood off: s2 follows its prior IG(a0, b0) and theta its prior.
    Likelihood on: sigma_rel settles near the noise-matched value
    sigma_p / |d| (the residuals pin it), whatever the configured sigma_rel."""
    from brho.samplers import SoftCurveSampler
    p = Problem(C.ProblemConfig(truth_mode="kle", likelihood="relative",
                                sigma_rel=2e-3), log=lambda m: None)   # 32x32, 512 sensors
    cfg = replace(C.SoftSamplerConfig(), prior_test=True, learn_noise=True,
                  s_init=(0.5, 0.3, 0.12, 0.6))
    s = SoftCurveSampler(p, _soft_curve(), cfg)
    assert s.STATE == ("w", "lb", "s2") and abs(s.noise_b0 - 2 * 4e-6) < 1e-15
    rng = np.random.default_rng(3)
    st = s.init_state(rng)
    ad = StepAdapter(s.s_init, targets=s.targets, every=10 ** 9)
    prec = []
    for _ in range(20000):
        s.step(st, ad, rng)
        prec.append(1.0 / st["s2"])
    prec = np.array(prec)                       # 1/s2 ~ Gamma(a0, rate b0)
    assert abs(prec.mean() / (s.noise_a0 / s.noise_b0) - 1) < 0.05
    assert abs(prec.var() / (s.noise_a0 / s.noise_b0 ** 2) - 1) < 0.15
    # likelihood on: the learned sigma_rel approaches the matched value
    matched = p.syn.sigma_p / np.linalg.norm(p.syn.d_p)
    cfg_on = replace(cfg, prior_test=False, s_init=(0.01, 0.05, 0.1, 0.1))
    s_on = SoftCurveSampler(p, _soft_curve(), cfg_on)
    th0 = p.kle.project(p.syn.G_true)                  # start at the truth
    w0 = (np.log(p.syn.rho_true) - cfg.mu_rho) / cfg.s_rho
    lb0 = float(np.log(p.syn.beta_true))
    phi0, pobs0, keff0 = s_on.potential(th0, w0, lb0)
    st = dict(theta=th0, w=w0, lb=lb0, phi=phi0, pobs=pobs0, keff=keff0, s2=s_on.s2_fixed)
    ad = StepAdapter(s_on.s_init, targets=s_on.targets, every=50)
    sig = []
    for it in range(600):
        s_on.step(st, ad, rng)
        ad.tick(it, 10 ** 6)
        if it >= 300:
            sig.append(np.sqrt(st["s2"]))
    ratio = np.median(sig) / matched
    assert 0.7 < ratio < 1.5, (np.median(sig), matched, p.sigma_rel)
    assert p.sigma_rel / matched > 1.5           # the configured value is tempered


def test_scale_blocks():
    """E9: theta_innovation="scale" moves one block of KLE modes per theta move,
    each block with its own step slot; combines with theta_repeat (2 + k
    solves); exactly prior-preserving (prior test); edges >= r dropped."""
    from brho.samplers import SoftCurveSampler
    from brho.scaleblocks import ScaleBlocks
    p = prob()
    r = p.r
    sb = ScaleBlocks(r, "5,10,%d" % (r + 50), "1,1,2")
    assert sb.n == 3 and sb.ranges == [(1, 5), (6, 10), (11, r)]
    assert np.allclose(sb.p, [0.25, 0.25, 0.5])
    assert sum(len(b) for b in sb.blocks) == r
    for bad in ("1,1", "1,0,1"):
        try:
            ScaleBlocks(r, "5,10", bad)
            raise AssertionError("bad scale_probs accepted: " + bad)
        except ValueError:
            pass
    th = np.arange(r, dtype=float)
    tp = sb.propose(th, 1, 0.3, np.random.default_rng(0))
    assert np.array_equal(tp[:5], th[:5]) and np.array_equal(tp[10:], th[10:])
    assert not np.array_equal(tp[5:10], th[5:10])
    base = replace(C.SoftSamplerConfig(), prior_test=True, s_init=(0.5, 0.3, 0.12, 0.6),
                   theta_innovation="scale", scale_edges="5,10", theta_repeat=3)
    sk = SoftCurveSampler(p, _soft_curve(), base)
    assert sk.solves_per_iteration == 5 and sk.sc.n == 3
    assert len(sk.s_init) == len(sk.targets) == 6
    assert sk.move_names[0].startswith("A1 modes 1-5") and sk.move_names[4].startswith("A2")
    rng = np.random.default_rng(8)
    out = []
    for c in range(1200):
        st = sk.init_state(rng)
        ad = StepAdapter(sk.s_init, targets=sk.targets, every=10 ** 9)
        for _ in range(5):
            sk.step(st, ad, rng)
        theta_props = ad.n[0] + ad.n[4] + ad.n[5]
        assert theta_props == 15 and ad.n[1] == 5
        out.append(st["theta"])
    th = np.array(out)
    frob = np.linalg.norm(np.cov(th.T) - np.eye(r)) / np.sqrt(r)
    ref = np.linalg.norm(np.cov(rng.standard_normal(th.shape).T) - np.eye(r)) / np.sqrt(r)
    assert frob < 1.3 * ref, (frob, ref)
    assert abs(th.mean()) < 0.05
    from brho import io
    assert io.exp_soft(False, "scale", "noise") == "e5_soft_scale_noise"


def test_collapse_beta_exact():
    """E10 (collapse_beta): the joint log-beta proposal is MH-exact.
    (1) The closed form Phi(lb) = w/2 |d - e^{-lb} g|^2 equals a PDE solve.
    (2) With theta and w held fixed the joint move is an MH chain on lb alone;
        its draws match the exact conditional exp(-Phi(lb)) N(lb; m, s^2)
        computed by quadrature, also with a deliberately wrong proposal q.
    (3) Prior test with scale blocks + theta_repeat: prior preserved
        (theta ~ N(0, I), xi_b = (lb - m(rho)) / s(rho) ~ N(0, 1)).
    (4) Learned noise, theta_repeat and B all run through the joint path."""
    from brho.samplers import SoftCurveSampler
    p = prob()
    cfg = replace(C.SoftSamplerConfig(), collapse_beta=True)
    smp = SoftCurveSampler(p, _soft_curve(), cfg)
    rng = np.random.default_rng(12)
    st = smp.init_state(rng)
    if getattr(p.syn, "theta_true", None) is not None:
        st["theta"] = np.asarray(p.syn.theta_true, float).copy()
    rho = smp.rho(st["w"])
    phi0, pobs0, _ = smp.potential(st["theta"], st["w"], 0.0)
    g = np.asarray(pobs0, float)
    for lb in (-0.3, 0.4, 1.1):
        phi_solve = smp.potential(st["theta"], st["w"], lb)[0]
        assert abs(smp._phi_lb(g, lb) - phi_solve) < 1e-8 * max(1.0, phi_solve)
    # exact conditional of lb on a grid
    m, s = smp.m(rho), smp.s(rho)
    a, bb = float(smp.d_p @ g), float(g @ g)
    lb_hat = -np.log(a / bb)
    sd_hat = 1.0 / np.sqrt(smp.w_lik * bb * (a / bb) ** 2 + 1 / s ** 2)
    grid = np.linspace(lb_hat - 12 * sd_hat, lb_hat + 12 * sd_hat, 20001)
    logp = np.array([-smp._phi_lb(g, x) for x in grid]) - 0.5 * ((grid - m) / s) ** 2
    pw = np.exp(logp - logp.max()); pw /= pw.sum()
    mean_ex = float(pw @ grid); sd_ex = float(np.sqrt(pw @ (grid - mean_ex) ** 2))
    orig_q = smp._q_lb
    for label, q in (("Laplace q", orig_q),
                     ("wrong q", lambda gg, r_, c_: (lambda mu, P: (mu + 0.8 / np.sqrt(P), 0.3 * P))(*orig_q(gg, r_, c_)))):
        smp._q_lb = q
        st2 = dict(st)
        st2["lb"] = float(lb_hat)
        st2["phi"], st2["pobs"], st2["keff"] = smp.potential(st2["theta"], st2["w"], st2["lb"])
        draws, acc = [], 0
        for _ in range(3000):
            acc += smp._joint_lb(st2, st2["theta"], st2["w"], rng)
            draws.append(st2["lb"])
        d = np.array(draws[300:])
        assert abs(d.mean() - mean_ex) < 0.15 * sd_ex, (label, d.mean(), mean_ex, sd_ex)
        assert abs(d.std() / sd_ex - 1) < 0.15, (label, d.std(), sd_ex)
        if label == "Laplace q":
            assert acc / 3000 > 0.9          # the proposal is nearly the exact conditional
    smp._q_lb = orig_q
    # (3) prior test: scale blocks + theta_repeat + collapse
    base = replace(C.SoftSamplerConfig(), prior_test=True, collapse_beta=True,
                   s_init=(0.5, 0.3, 0.12, 0.6), theta_innovation="scale",
                   scale_edges="5,10", theta_repeat=2)
    sk = SoftCurveSampler(p, _soft_curve(), base)
    out, xb = [], []
    for c_ in range(1000):
        s0 = sk.init_state(rng)
        ad = StepAdapter(sk.s_init, targets=sk.targets, every=10 ** 9)
        for _ in range(5):
            sk.step(s0, ad, rng)
        out.append(s0["theta"]); xb.append(sk.extra_traces(s0)["xi_b"])
    th = np.array(out)
    frob = np.linalg.norm(np.cov(th.T) - np.eye(p.r)) / np.sqrt(p.r)
    ref = np.linalg.norm(np.cov(rng.standard_normal(th.shape).T) - np.eye(p.r)) / np.sqrt(p.r)
    assert frob < 1.3 * ref, (frob, ref)
    xb = np.array(xb)
    assert abs(xb.mean()) < 0.1 and abs(xb.std() - 1) < 0.1, (xb.mean(), xb.std())
    # (4) likelihood on, learned noise: runs, state stays consistent with a fresh solve
    sl = SoftCurveSampler(p, _soft_curve(), replace(C.SoftSamplerConfig(), collapse_beta=True,
                                                     learn_noise=True, theta_repeat=3,
                                                     s_init=(0.02, 0.02, 0.1, 0.2)))
    s1 = sl.init_state(rng)
    ad = StepAdapter(sl.s_init, targets=sl.targets, every=10 ** 9)
    for _ in range(40):
        sl.step(s1, ad, rng)
    phi_chk, pobs_chk, keff_chk = sl.potential(s1["theta"], s1["w"], s1["lb"])
    assert abs(phi_chk - s1["phi"]) < 1e-6 * max(1.0, phi_chk)
    assert np.allclose(pobs_chk, s1["pobs"]) and abs(keff_chk / s1["keff"] - 1) < 1e-9
    assert ad.n[0] == 120 and ad.n[1] == 40


def test_collapse_beta_slice_refresh():
    """E10 lb refresh (_slice_lb): (1) exact -- at fixed theta, w its draws
    match the conditional exp(-Phi(lb)) N(lb; m, s^2) by quadrature, also
    started on the plateau far above the optimum; (2) it frees the trap the
    E10 r = 57 run hit: with lb 2 above its optimum the Laplace joint move
    alone is never accepted, after one refresh it is."""
    from brho.samplers import SoftCurveSampler
    p = prob()
    smp = SoftCurveSampler(p, _soft_curve(), replace(C.SoftSamplerConfig(), collapse_beta=True))
    rng = np.random.default_rng(3)
    st = smp.init_state(rng)
    rho = smp.rho(st["w"])
    g = np.asarray(smp.potential(st["theta"], st["w"], 0.0)[1], float)
    m, s = smp.m(rho), smp.s(rho)
    a, bb = float(smp.d_p @ g), float(g @ g)
    lb_hat = -np.log(a / bb)
    sd_hat = 1.0 / np.sqrt(smp.w_lik * bb * (a / bb) ** 2 + 1 / s ** 2)
    grid = np.linspace(lb_hat - 12 * sd_hat, lb_hat + 12 * sd_hat, 20001)
    logp = np.array([-smp._phi_lb(g, x) for x in grid]) - 0.5 * ((grid - m) / s) ** 2
    pw = np.exp(logp - logp.max()); pw /= pw.sum()
    mean_ex = float(pw @ grid); sd_ex = float(np.sqrt(pw @ (grid - mean_ex) ** 2))

    def at(lb):
        s2 = dict(st); s2["lb"] = float(lb)
        s2["phi"], s2["pobs"], s2["keff"] = smp.potential(s2["theta"], s2["w"], s2["lb"])
        return s2

    for start in (lb_hat, lb_hat + 3.0):
        s2, draws = at(start), []
        for _ in range(4000):
            smp._slice_lb(s2, rng)
            draws.append(s2["lb"])
        d = np.array(draws[200:])
        assert abs(d.mean() - mean_ex) < 0.1 * sd_ex, (start - lb_hat, d.mean(), mean_ex, sd_ex)
        assert abs(d.std() / sd_ex - 1) < 0.1, (start - lb_hat, d.std(), sd_ex)
    phi_chk, pobs_chk, keff_chk = smp.potential(s2["theta"], s2["w"], s2["lb"])
    assert abs(phi_chk - s2["phi"]) < 1e-6 * max(1.0, phi_chk)
    assert np.allclose(pobs_chk, s2["pobs"]) and abs(keff_chk / s2["keff"] - 1) < 1e-9
    # (2) the trap and its release
    s3 = at(lb_hat + 2.0)
    assert sum(smp._joint_lb(s3, s3["theta"], s3["w"], rng) for _ in range(200)) == 0
    smp._slice_lb(s3, rng)
    assert sum(smp._joint_lb(s3, s3["theta"], s3["w"], rng) for _ in range(200)) > 100


def test_beta_upgrade_level_swap_exact():
    """Level-swap upgrade of hard-curve samples (brho/beta_upgrade.py).
    Likelihood off: hard-curve 'posterior' = prior, theta ~ N(0, I), rho ~
    lognormal.  The weighted, shifted samples must follow the restored model:
    theta' ~ N(0, I) (checked along c, the direction that moves), xi =
    (lb - m(rho)) / s(rho) ~ N(0, 1), rho keeps its prior; and log K is
    unchanged by the shift when T c = 1 exactly."""
    from brho.beta_upgrade import level_swap, constant_coords, summarize
    rng = np.random.default_rng(5)
    n, r = 400_000, 12
    theta = rng.standard_normal((n, r))
    # keep rho / s > |c| so the importance weights have finite variance (the
    # prior is the worst case; under the data a = theta . c is pinned)
    rho = 1.5 * np.exp(0.3 * rng.standard_normal(n))
    c = rng.standard_normal(r) * 0.2
    m = 0.7 + 0.2 * rho
    s = 0.25 + 0.1 * rho
    up = level_swap(theta, rho, c, m, s, rng)
    w = up["w"]
    xi = (up["lb"] - m) / s
    assert abs(w @ xi) < 0.02 and abs(np.sqrt(w @ xi ** 2) - 1) < 0.02, (w @ xi, w @ xi ** 2)
    u = up["theta"] @ (c / np.linalg.norm(c))
    assert abs(w @ u) < 0.02 and abs(np.sqrt(w @ u ** 2) - 1) < 0.02, (w @ u, w @ u ** 2)
    lr = np.log(rho) - np.log(1.5)
    assert abs(w @ lr) < 0.01 and abs(np.sqrt(w @ lr ** 2) - 0.3) < 0.01
    assert up["ess"] > 0.5 * n, up["ess"] / n
    # T c = 1 exactly -> log K = lb + rho T theta unchanged
    T = rng.standard_normal((30, r))
    c2, resid = constant_coords(np.column_stack([T[:, :-1], np.ones(30)]))
    assert resid < 1e-12
    T2 = np.column_stack([T[:, :-1], np.ones(30)])
    th, rh = theta[:50], rho[:50]
    up2 = level_swap(th, rh, c2, m[:50], s[:50], rng)
    logK0 = m[:50, None] + rh[:, None] * (th @ T2.T)
    logK1 = up2["lb"][:, None] + rh[:, None] * (up2["theta"] @ T2.T)
    assert np.allclose(logK0, logK1)
    q = summarize(np.arange(101.0), truth=50.0)
    assert abs(q["median"] - 50) < 1e-9 and q["covered"]
