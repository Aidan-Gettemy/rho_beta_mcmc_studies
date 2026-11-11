"""GLAM validation V2/V3 and machinery tests: no MPI, no solver (ThreadWorld ranks, toy forward models)."""
import glob
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from brho import config as C  # noqa: E402
from brho.glam.comm import ThreadWorld  # noqa: E402
from brho.glam.model import BetaPchip  # noqa: E402
from brho.glam.sampler import pcn_propose, run_glam  # noqa: E402
from brho.glam.stepsize import GLOBAL, LOCAL, TwoStepAdapter  # noqa: E402
from brho.glam.toy import LinearGaussianModel, ToyModel  # noqa: E402

FITS = os.environ.get("GLAM_FITS") or os.path.join(
    REPO, "runs", "juno_external_2026-10", "tables", "e3_external", "fits.json")  # runs/ is not in the sparse checkout: set GLAM_FITS


def _beta():
    return BetaPchip.from_fits_json(FITS)


def load_chain(out, rank):
    parts = sorted(glob.glob(os.path.join(out, "chains", f"rank{rank:03d}_part*.npz")))
    z = [np.load(p) for p in parts]
    cat = lambda k: np.concatenate([a[k] for a in z])  # noqa: E731
    return dict(start=cat("start"), count=cat("count"), theta=np.concatenate([a["theta"] for a in z]),
                w=cat("w"), phi=cat("phi"), kind=cat("trace_kind"), acc=cat("trace_acc"),
                bg=cat("trace_beta_g"), bl=cat("trace_beta_l"), phi_tr=cat("trace_phi"))


def expand(ch, n_iter):
    """per-iteration index of the record that holds the state after the move at that iteration"""
    idx = np.full(n_iter, -1)
    for j, (s, c) in enumerate(zip(ch["start"], ch["count"])):
        idx[s:s + c] = j
    return idx


# ------------------------------------------------------------------------------------------- moves / adaptation
def test_pcn_propose_blocks_and_prior_invariance():
    rng = np.random.default_rng(1)
    th = [rng.standard_normal(5) for _ in range(4)] + [rng.standard_normal(1)]
    p = pcn_propose(th, [2], 0.3, rng)
    assert all(np.array_equal(p[k], th[k]) for k in (0, 1, 3, 4)), "local move must only touch its block (and not rho)"
    assert not np.array_equal(p[2], th[2])
    # N(0, I) is invariant: start from the prior, propose, the marginal stays N(0, 1)
    x = rng.standard_normal((200000, 1)); y = np.sqrt(1 - 0.5 ** 2) * x + 0.5 * rng.standard_normal((200000, 1))
    assert abs(y.mean()) < 0.01 and abs(y.var() - 1) < 0.02


def test_adapter_gain_hold_and_constant_gain():
    """gain_n = gain / max(1, n - hold)**exponent: with hold H the first H+1 updates all use the full gain."""
    def run(**kw):
        def work(comm):
            ad = TwoStepAdapter(comm, 0.05, 0.05, target=0.25, adapt_stop=10 ** 6, adapt_interval=1, **kw)
            logits = []
            for it in range(12):
                for _ in range(10):
                    ad.record(GLOBAL, False)                 # acceptance 0 -> logit change = -gain_n * 0.25
                ad.adapt(it)
                logits.append(ad.logit[GLOBAL])
            return np.diff(np.r_[np.log(0.05) - np.log1p(-0.05), logits])
        return ThreadWorld(1).run(work)[0]
    d0 = run(initial_gain=0.5, gain_exponent=0.6, gain_hold=0)
    d5 = run(initial_gain=0.5, gain_exponent=0.6, gain_hold=5)
    dc = run(initial_gain=0.5, gain_exponent=0.0, gain_hold=0)
    n = np.arange(1, 13)
    assert np.allclose(d0, -0.5 * 0.25 / n ** 0.6, rtol=1e-6)
    assert np.allclose(d5, -0.5 * 0.25 / np.maximum(1, n - 5) ** 0.6, rtol=1e-6)
    assert np.allclose(d5[:6], -0.125)                          # still at the full gain through update 6
    assert np.allclose(dc, -0.125)                              # constant gain


def test_two_step_adapter_reaches_target_and_freezes():
    def work(comm):
        rng = np.random.default_rng(comm.Get_rank())
        ad = TwoStepAdapter(comm, 0.5, 0.5, target=0.25, adapt_stop=3000, adapt_interval=20)
        for it in range(4000):
            for kind in (GLOBAL, LOCAL):
                if kind == LOCAL and (it // 100) % 2 == 0:
                    continue                                 # local only proposed in alternate blocks of iterations
                p_acc = 1.0 - ad.beta[kind] if kind == GLOBAL else 0.6 * (1.0 - ad.beta[kind])
                ad.record(kind, rng.random() < p_acc)
            ad.adapt(it)
            if it == 2999 and not ad.frozen:
                pass
        return ad.beta.copy(), ad.frozen, len(ad.history)
    out = ThreadWorld(4).run(work)
    for beta, frozen, nh in out:
        assert frozen and nh <= 150
        assert abs(beta[0] - 0.75) < 0.05, beta          # 1 - beta = 0.25
        assert abs(beta[1] - (1 - 0.25 / 0.6)) < 0.05, beta
    assert all(np.allclose(out[0][0], o[0]) for o in out), "all ranks must hold identical betas"


# ------------------------------------------------------------------------------------------- V2: prior only
class PriorOnlyToy(ToyModel):
    def phi(self, state, field, sim):
        return 0.0 if self.rho_ok(state) else np.inf


def test_prior_only_global_moves_recover_prior():
    model = PriorOnlyToy(_beta(), n_sub=3, m=4, n_cells=30, n_times=2)
    cfg = C.GlamConfig(n_iter=4000, window1=500, window2=0, burnin_length=10 ** 6, global_stage_length=10 ** 6,
                       local_stage_length=10 ** 6, beta_g0=0.5, adapt_interval=50, save_every=10 ** 6, log_every=0)
    out = tempfile.mkdtemp()
    ThreadWorld(3).run(lambda comm: run_glam(model, cfg, comm, out, log=lambda *_: None))
    th, w = [], []
    for r in range(3):
        ch = load_chain(out, r)
        idx = expand(ch, cfg.n_iter)[cfg.window1:]
        th.append(ch["theta"][idx]); w.append(ch["w"][idx])
    th, w = np.concatenate(th), np.concatenate(w)
    assert abs(th.mean()) < 0.06 and abs(th.var() - 1.0) < 0.08, (th.mean(), th.var())
    # rho marginal: standard normal w truncated to the PCHIP range [0.05, 5]  (log rho = 0.75 w)
    from scipy.stats import norm, kstest
    lo, hi = np.log(0.05) / 0.75, np.log(5.0) / 0.75
    cdf = lambda x: (norm.cdf(np.clip(x, lo, hi)) - norm.cdf(lo)) / (norm.cdf(hi) - norm.cdf(lo))  # noqa: E731
    d = kstest(w[::20], cdf).statistic          # thinned (autocorrelation); loose threshold
    assert d < 0.07, d
    assert np.all((w >= lo) & (w <= hi)), "rho left the PCHIP range"


# ------------------------------------------------------------------------------------------- V3: linear Gaussian
def test_linear_gaussian_posterior_without_exchanges():
    model = LinearGaussianModel(n_sub=3, m=4, n_data=10, s=0.7, seed=3)
    cfg = C.GlamConfig(n_iter=30000, window1=3000, window2=0, burnin_length=10 ** 8, global_stage_length=10 ** 8,
                       local_stage_length=10 ** 8, beta_g0=0.2, adapt_interval=100, save_every=10 ** 6, log_every=0)
    out = tempfile.mkdtemp()
    ThreadWorld(4).run(lambda comm: run_glam(model, cfg, comm, out, log=lambda *_: None))
    X = []
    for r in range(4):
        ch = load_chain(out, r)
        idx = expand(ch, cfg.n_iter)[cfg.window1:]
        X.append(np.hstack([ch["theta"][idx], ch["w"][idx][:, None]]))
    X = np.concatenate(X)
    mu, cov = model.posterior()
    sd = np.sqrt(np.diag(cov))
    assert np.max(np.abs(X.mean(0) - mu) / sd) < 0.12, np.abs(X.mean(0) - mu) / sd
    assert np.max(np.abs(X.std(0) / sd - 1)) < 0.10, X.std(0) / sd
    c_emp = np.cov(X.T); c_ref = cov
    assert np.max(np.abs(c_emp - c_ref)) < 0.12 * sd.max() ** 2


# ------------------------------------------------------------------------------------------- machinery: exchanges
def _toy_run(n_iter, out, P=4, resume=False, **kw):
    model = ToyModel(_beta(), n_sub=4, m=6, n_cells=60, n_times=3, seed=5, sigma_conc=0.05, sigma_dp=0.05)
    base = dict(n_iter=n_iter, window1=60, window2=60, burnin_length=20, global_stage_length=20, local_stage_length=15,
                min_global_stage_length=10, min_local_stage_length=10, adapt_interval=10, beta_g0=0.1, beta_l0=0.2,
                quality_tolerance=50.0, minimum_anchor_distance=0.5, restart_scale=0.2, save_every=25, log_every=0)
    base.update(kw)
    cfg = C.GlamConfig(**base)
    res = ThreadWorld(P).run(lambda comm: run_glam(model, cfg, comm, out, resume=resume, log=lambda *_: None))
    return model, cfg, res


def test_exchange_schedule_adaptation_and_output():
    out = tempfile.mkdtemp()
    model, cfg, res = _toy_run(240, out)
    P = 4
    assert res[0]["n_exchanges"] >= 4
    phases = [e["phase"] for e in res[0]["exchanges"]]
    assert "LOCAL" in phases and "GLOBAL" in phases
    # betas: identical on all ranks, history only inside window 1, frozen afterwards
    assert all(np.allclose(res[0]["beta"], r["beta"]) for r in res)
    hist = res[0]["adapter"].history
    assert hist and max(h[0] for h in hist) < cfg.window1
    n_sub = model.n_sub
    for r in range(P):
        ch = load_chain(out, r)
        idx = expand(ch, cfg.n_iter if hasattr(cfg, "n_iter") else 240)
        assert np.all(idx >= 0), "every iteration must be covered by exactly one record"
        assert np.all(ch["count"] > 0) and int(ch["count"].sum()) == 240
        assert len(ch["kind"]) == 240
        # betas after window 1 are constant
        assert np.all(ch["bg"][cfg.window1:] == ch["bg"][cfg.window1]) and np.all(ch["bl"][cfg.window1:] == ch["bl"][cfg.window1])
        # local stages exist, and stop at window1 + window2 (a stage already running may finish)
        last_local = np.max(np.nonzero(ch["kind"] == LOCAL)[0])
        assert np.any(ch["kind"][:cfg.window1] == LOCAL)
        assert last_local <= cfg.window1 + cfg.window2 + cfg.local_stage_length
        # local accepted moves change only block r % n_sub and never rho
        d = ch["theta"].shape[1] // n_sub
        ex_its = {e["iteration"] for e in res[0]["exchanges"]}
        n_checked = 0
        for t in np.nonzero((ch["kind"] == LOCAL) & (ch["acc"] == 1))[0]:
            if t == 0 or idx[t] == idx[t - 1] or int(t) in ex_its:   # an exchange at t also replaces the state
                continue
            a, b = ch["theta"][idx[t - 1]], ch["theta"][idx[t]]
            changed = [k for k in range(n_sub) if not np.array_equal(a[k * d:(k + 1) * d], b[k * d:(k + 1) * d])]
            if ch["start"][idx[t]] == t and changed:       # a move (not an exchange restart) starts a record here
                assert changed == [r % n_sub] and ch["w"][idx[t - 1]] == ch["w"][idx[t]]
                n_checked += 1
        assert n_checked > 0, "no accepted local move was checked"


def test_anchor_stop_and_current_only_options_run():
    out = tempfile.mkdtemp()
    _, cfg, res = _toy_run(150, out, anchor_stop_iteration=80, anchor_current_only=True, window2=30)
    its = [e["iteration"] for e in res[0]["exchanges"]]
    assert its and max(its) <= 80
    assert res[0]["manager"].current_iteration_only is True


def test_resume_is_exact():
    full, part = tempfile.mkdtemp(), tempfile.mkdtemp()
    _, cfg, a = _toy_run(200, full)
    _toy_run(130, part)
    _, _, b = _toy_run(200, part, resume=True)
    for ra, rb in zip(a, b):
        assert all(np.array_equal(x, y) for x, y in zip(ra["theta"], rb["theta"]))
        assert ra["phi"] == rb["phi"]
    for r in range(4):
        ca, cb = load_chain(full, r), load_chain(part, r)
        assert np.array_equal(ca["start"], cb["start"]) and np.array_equal(ca["theta"], cb["theta"])
        from brho.glam import post
        pa, pb = post.load_chain(full, r), post.load_chain(part, r)
        assert np.array_equal(pa["re_conc"], pb["re_conc"]) and np.array_equal(pa["re_dp"], pb["re_dp"])   # RE trace resumes exactly


# ------------------------------------------------------------------------------------------- V0: external model plumbing
def test_external_model_build_simulate_and_run_with_mock_solver():
    import importlib.util
    spec = importlib.util.spec_from_file_location("t_ext", os.path.join(HERE, "test_external.py"))
    te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)
    from brho.darcy_external import DarcyRunner
    from brho.glam.model import ExternalModel, SimData
    from brho.glam_likelihood import GlamData
    from brho.multiscale_mat import MultiscaleField
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(4)
        PHI, U, sv, cells, T_list = te._synthetic_multiscale(rng, N=288, M=30, n_sub=3)
        field = MultiscaleField(T_list, cells, PHI, U, 1.0 / np.sqrt(sv), dtype=np.float64)
        cfg = te._write_cfg(d)
        runner = DarcyRunner(te._mock_exe(d), cfg, os.path.join(d, "w"), keep_velocity=True)
        beta = _beta()
        n_times = 3
        tracer = lambda vel: np.tile(np.linspace(0.1, 0.9, 288)[:, None], (1, n_times))  # noqa: E731
        model = ExternalModel(field, beta, runner, tracer, GlamData(np.ones((288, n_times)), 3000.0))
        st = model.sample_state(rng, model.block_dims)
        f = model.build(st)
        rho = model.rho_of(st[3][0])
        ref = np.exp(float(beta.log_beta(rho)) + rho * field.field(st[:3]))
        assert np.allclose(f.simulation_field, ref, rtol=1e-12)
        sim = model.simulate(f, rank=0)
        assert np.isfinite(sim.dp_pa) and sim.conc.shape == (288, n_times)
        assert abs(sim.dp_pa - runner.phys["v"] * runner.phys["mu"] * runner.phys["Lz"] / runner.keff(ref, "chk")) < 1e-9 * sim.dp_pa
        model.data = GlamData(sim.conc, sim.dp_pa)
        assert model.phi(st, f, sim) == 0.0
        assert model.phi(st, f, SimData(np.nan, None)) == np.inf
        bad = [b.copy() for b in st]; bad[3] = np.array([50.0])             # rho far above the PCHIP range
        assert not model.rho_ok(bad) and model.phi(bad, f, sim) == np.inf
        # short GLAM run through the real model class (2 ranks, mock solver, mock tracer)
        gc = C.GlamConfig(n_iter=30, window1=10, window2=10, burnin_length=8, global_stage_length=6, local_stage_length=6,
                          min_global_stage_length=3, min_local_stage_length=3, adapt_interval=5, save_every=50,
                          log_every=0, minimum_anchor_distance=0.1)
        out = os.path.join(d, "run")
        res = ThreadWorld(2).run(lambda comm: run_glam(model, gc, comm, out, log=lambda *_: None))
        assert res[0]["n_exchanges"] >= 2


def test_world_refuses_to_fall_back_to_serial_under_a_launcher(monkeypatch=None):
    import builtins
    from brho.glam import comm as cm
    old, real = os.environ.get("OMPI_COMM_WORLD_SIZE"), builtins.__import__

    def no_mpi(name, *a, **k):
        if name.startswith("mpi4py"):
            raise ImportError("no mpi4py")
        return real(name, *a, **k)
    builtins.__import__ = no_mpi
    try:
        os.environ.pop("OMPI_COMM_WORLD_SIZE", None)
        assert isinstance(cm.world(), cm.SerialComm)          # no launcher: serial is fine
        os.environ["OMPI_COMM_WORLD_SIZE"] = "64"
        try:
            cm.world()
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "mpi4py" in str(e)
    finally:
        builtins.__import__ = real
        if old is None:
            os.environ.pop("OMPI_COMM_WORLD_SIZE", None)
        else:
            os.environ["OMPI_COMM_WORLD_SIZE"] = old


def test_resume_survives_an_incomplete_newest_snapshot_and_a_run_with_no_snapshot():
    # preemption: the newest chain snapshot is missing on one rank -> resume from the previous common one
    full, part = tempfile.mkdtemp(), tempfile.mkdtemp()
    _, cfg, a = _toy_run(200, full)
    _toy_run(150, part)
    snaps = sorted(glob.glob(os.path.join(part, "anchor_state", "chain_r001_ite*.pkl")),
                   key=lambda p: int(os.path.basename(p)[len("chain_r001_ite"):-4]))
    os.remove(snaps[-1])                                   # rank 1 never wrote its newest snapshot
    _, _, b = _toy_run(200, part, resume=True)
    for ra, rb in zip(a, b):
        assert all(np.array_equal(x, y) for x, y in zip(ra["theta"], rb["theta"]))
    # killed before the first exchange: no snapshot at all -> start from scratch, stale parts are removed
    empty = tempfile.mkdtemp()
    _toy_run(25, empty, save_every=5)
    __import__("shutil").rmtree(os.path.join(empty, "anchor_state"))
    _, _, c = _toy_run(200, empty, resume=True)
    for ra, rc in zip(a, c):
        assert all(np.array_equal(x, y) for x, y in zip(ra["theta"], rc["theta"]))
    for r in range(4):
        ca, cc = load_chain(full, r), load_chain(empty, r)
        assert np.array_equal(ca["start"], cc["start"]) and np.array_equal(ca["theta"], cc["theta"])


def test_every_manager_setting_is_a_config_field_with_the_manager_default():
    import inspect
    from brho.glam.anchors import GlobalLocalAnchorManager as M
    from brho.glam.sampler import _NONE_IF_NEGATIVE, _PASSTHROUGH, make_manager, manager_kwargs
    sig = inspect.signature(M.__init__).parameters
    kw = manager_kwargs(C.GlamConfig())
    deliberate = {"num_cycles", "permeability_logdet_weight", "pressure_logdet_weight"}   # brho's own choices
    for k in _PASSTHROUGH:
        if k not in deliberate:
            assert kw[k] == sig[k].default, k
    # parameters the manager has that GlamConfig reaches under another name, derives, or deliberately omits
    covered = set(_PASSTHROUGH) | {"numsubd", "global_stage_length", "local_stage_length", "quality_tolerance",
                                   "minimum_anchor_distance", "restart_scale", "max_anchors", "initial_burnin_length",
                                   "minimum_global_stage_length", "minimum_local_stage_length", "global_stage_decay",
                                   "local_stage_decay", "disable_local_after_iteration", "current_iteration_only",
                                   "start_iteration", "seed", "extra_state_blocks", "extra_block_weights",
                                   "relative_error_fn", "restart_basis", "include_previous_anchors",
                                   "keep_own_mask", "selection_constrained_only"}
    assert set(sig) - {"self"} <= covered, set(sig) - {"self"} - covered
    # negative sentinel -> None, and values reach the manager
    cfg = C.GlamConfig(max_restart_attempts=7, frontier_target_phi=12.5, restart_phi_tolerance=9.0, random_reach_attempts=0)
    mgr = make_manager(cfg, 4, 0, n_chains=8)
    assert mgr.max_restart_attempts == 7 and mgr.restart_phi_tolerance == 9.0 and mgr.random_reach_attempts == 0
    assert mgr.max_anchors == 16
    assert make_manager(C.GlamConfig(), 4, 0).frontier_target_phi is None
    assert "max_restart_attempts" in C.to_dict(cfg)["GlamConfig"]


def test_anchor_pool_toggles_change_the_candidate_pool():
    P = 4
    raw = {}
    for name, kw in dict(default=dict(), no_prev=dict(anchor_include_previous=False),
                         final_only=dict(anchor_use_best_state=False),
                         neither=dict(anchor_include_previous=False, anchor_use_best_state=False),
                         old_alias=dict(anchor_current_only=True)).items():
        _, cfg, res = _toy_run(160, tempfile.mkdtemp(), **kw)
        raw[name] = [e["archive"]["candidate_count_raw"] for e in res[0]["exchanges"]]
        assert len(raw[name]) >= 3
    # the first exchange has no previous archive; later ones differ by the size of the previous archive
    assert raw["default"][0] == 2 * P and raw["final_only"][0] == P
    assert raw["no_prev"][1:] == [2 * P] * (len(raw["no_prev"]) - 1)
    assert raw["neither"] == [P] * len(raw["neither"])
    assert all(a > b for a, b in zip(raw["default"][1:], raw["no_prev"][1:]))
    assert raw["old_alias"] == raw["final_only"]                     # deprecated alias = anchor_use_best_state False


def test_include_previous_false_forgets_persistent_memory():
    from brho.glam.sampler import make_manager
    mgr = make_manager(C.GlamConfig(anchor_include_previous=False), 4, 0, n_chains=4)
    mgr.best_anchor = object()
    mgr.frontier_anchor_id, mgr.frontier_rounds, mgr.frontier_status = 3, 2, "ACTIVE"
    mgr.frontier_cooldown_until = {1: 5}
    mgr.successful_anchors, mgr.successful_region_ids = [object()], [1]
    mgr.successful_region_aliases = {2: 1}
    mgr.successful_region_merge_history = [{"x": 1}]
    mgr._forget_memory()
    assert mgr.best_anchor is None and mgr.frontier_anchor_id is None
    assert mgr.frontier_rounds == 0 and mgr.frontier_status == "NONE" and mgr.frontier_cooldown_until == {}
    assert mgr.successful_anchors == [] and mgr.successful_region_ids == []
    assert mgr.successful_region_aliases == {} and mgr.successful_region_merge_history == []
    # full toy runs complete (no invariant violation) with and without the toggle
    for kw in (dict(anchor_include_previous=False), dict()):
        _, cfg, res = _toy_run(160, tempfile.mkdtemp(), **kw)
        assert len(res[0]["exchanges"]) >= 3


def test_round_robin_local_blocks_cover_every_block_and_match_the_moves():
    """fixed: chain r moves block r % n_sub in every local stage (3 chains of 4 blocks never touch block 3);
    round_robin: block (r + k) % n_sub in the k-th local stage, so all blocks are visited; the manager's restart code
    perturbs the same block that the sampler then moves."""
    from brho.glam import sampler as S
    from brho.glam.anchors import Phase
    from brho.glam.sampler import make_manager
    mgr = make_manager(C.GlamConfig(local_block_schedule="round_robin"), 12, 5, n_chains=12)
    assert mgr.local_subdomain(5) == 5 and mgr.local_subdomain(0) == 0
    mgr.cycle = 3
    assert mgr.local_subdomain(5) == 8 and mgr.local_subdomain(10) == 1            # wraps around
    mgr.phase = Phase.LOCAL
    assert mgr.active_subdomain_indices(5).tolist() == [8] and mgr.subdomain_schedule(5)[0].tolist() == [8]
    fixed = make_manager(C.GlamConfig(), 12, 5, n_chains=12)
    fixed.cycle = 3
    assert fixed.local_subdomain(5) == 5
    try:
        make_manager(C.GlamConfig(local_block_schedule="nonsense"), 12, 0)
        raise AssertionError("an invalid schedule must be rejected")
    except ValueError:
        pass

    def visited(schedule):
        seen, orig = set(), S.pcn_propose
        def spy(theta, blocks, beta, rng):
            if len(blocks) == 1:
                seen.add(blocks[0])
            return orig(theta, blocks, beta, rng)
        S.pcn_propose = spy
        try:
            model = ToyModel(_beta(), n_sub=4, m=4, n_cells=30, n_times=2)
            cfg = C.GlamConfig(n_iter=240, window1=40, window2=200, burnin_length=10, global_stage_length=10,
                               local_stage_length=10, min_global_stage_length=10, min_local_stage_length=10,
                               adapt_interval=10, save_every=10 ** 6, log_every=0, local_block_schedule=schedule)
            ThreadWorld(3).run(lambda comm: run_glam(model, cfg, comm, tempfile.mkdtemp(), log=lambda *_: None))
        finally:
            S.pcn_propose = orig
        return seen
    assert visited("fixed") == {0, 1, 2}
    assert visited("round_robin") == {0, 1, 2, 3}

def test_run_template_lists_every_config_field_and_parses_to_the_intended_config():
    """cluster/slurm/glam_run_template.sbatch is the self-contained record of a run: every GlamConfig field must be
    written out (so a new field cannot be silently missing), the flags must parse, and the conda env is pinned."""
    import dataclasses, subprocess, sys
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    import run_glam
    path = os.path.join(REPO, "cluster", "slurm", "glam_run_template.sbatch")
    script = open(path).read()
    out = subprocess.run(["bash", "-c", f'source <(sed -n "/^GLAM_FLAGS=(/,/^)/p" {path}); printf "%s\\n" "${{GLAM_FLAGS[@]}}"'],
                         capture_output=True, text=True, check=True).stdout.split("\n")[:-1]
    cfg = C.from_cli(run_glam.build_parser().parse_args(out), C.GlamConfig)
    flags = {a[2:].replace("-", "_") for a in out if a.startswith("--")}
    assert {f.name for f in dataclasses.fields(C.GlamConfig)} <= flags, \
        {f.name for f in dataclasses.fields(C.GlamConfig)} - flags
    assert cfg.local_block_schedule == "round_robin" and cfg.max_anchors == 1 and not cfg.anchor_include_previous
    assert not cfg.anchor_use_best_state and cfg.restart_scale == cfg.exploit_restart_scale == 0.01
    assert cfg.window1 < cfg.n_iter
    assert "export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC" in script      # pinned, not inherited
    assert '[ "${CONDA_PREFIX:-}" = "$BRHO_CONDA_ENV" ]' in script


def test_r24_script_parses_to_the_intended_config():
    """cluster/slurm/glam_r24.sbatch: 24 chains on 12 blocks, doubled sigmas, longer adaptation, pinned env, h200."""
    import dataclasses, subprocess, sys
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    import run_glam
    path = os.path.join(REPO, "cluster", "slurm", "glam_r24.sbatch")
    script = open(path).read()
    out = subprocess.run(["bash", "-c", f'source <(sed -n "/^GLAM_FLAGS=(/,/^)/p" {path}); printf "%s\\n" "${{GLAM_FLAGS[@]}}"'],
                         capture_output=True, text=True, check=True).stdout.split("\n")[:-1]
    cfg = C.from_cli(run_glam.build_parser().parse_args(out), C.GlamConfig)
    flags = {a[2:].replace("-", "_") for a in out if a.startswith("--")}
    assert {f.name for f in dataclasses.fields(C.GlamConfig)} <= flags
    assert (cfg.sigma_conc, cfg.sigma_dp, cfg.target, cfg.window1, cfg.window2) == (0.002, 0.004, 0.25, 14000, 4000)
    assert cfg.local_block_schedule == "round_robin" and cfg.max_anchors == 1
    assert "#SBATCH --ntasks=24" in script and "#SBATCH --partition=h200" in script and "GLAM_TAG=r24" in script
    assert "export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC" in script
    assert "N_SUB=12" in script


def test_r24m_scripts_and_exchange_summary(tmp_path, monkeypatch):
    """glam_r24m(.pilot).sbatch: open archive, role split 25/25/50, no minimum distance; the exchange summary reads a toy run."""
    import dataclasses, pickle, subprocess, sys
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    import run_glam
    for name, n_iter, tag in (("glam_r24m.sbatch", 40000, "r24m"), ("glam_r24m_pilot.sbatch", 3000, "r24m-pilot")):
        path = os.path.join(REPO, "cluster", "slurm", name)
        script = open(path).read()
        out = subprocess.run(["bash", "-c", f'source <(sed -n "/^GLAM_FLAGS=(/,/^)/p" {path}); printf "%s\\n" "${{GLAM_FLAGS[@]}}"'],
                             capture_output=True, text=True, check=True).stdout.split("\n")[:-1]
        cfg = C.from_cli(run_glam.build_parser().parse_args(out), C.GlamConfig)
        flags = {a[2:].replace("-", "_") for a in out if a.startswith("--")}
        assert {f.name for f in dataclasses.fields(C.GlamConfig)} <= flags
        assert cfg.n_iter == n_iter and cfg.max_anchors < 0 and cfg.minimum_anchor_distance == 0.0
        assert cfg.anchor_include_previous and cfg.anchor_use_best_state
        assert abs(cfg.exploit_fraction + cfg.frontier_fraction + cfg.explore_fraction - 1.0) < 1e-12
        assert (cfg.exploit_fraction, cfg.frontier_fraction, cfg.explore_fraction) == (0.25, 0.25, 0.5)
        assert cfg.exploit_restart_scale < cfg.frontier_restart_scale < cfg.exploration_restart_scale
        assert (cfg.sigma_conc, cfg.sigma_dp, cfg.target) == (0.002, 0.004, 0.25)
        assert f"\nGLAM_TAG={tag} " in script and "#SBATCH --ntasks=24" in script and "#SBATCH --partition=h200" in script
        assert "export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC" in script
    monkeypatch.setenv("BRHO_ROOT", str(tmp_path))
    from brho import io
    exp = io.EXP_GLAM + "_t"
    model, cfg, res = _toy_run(200, tempfile.mkdtemp(), P=4, max_anchors=-1, minimum_anchor_distance=0.0)
    rdir = io.results_dir(exp); os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "exchanges.pkl"), "wb") as f:
        pickle.dump(res[0]["exchanges"], f)
    import glam_exchange_summary as G
    monkeypatch.setattr(sys, "argv", ["x", "--exp-tag", "t"])
    G.main()
    csv = open(os.path.join(io.tables_dir(exp), "exchanges.csv")).read().splitlines()
    assert csv[0].startswith("iteration,phase,cycle,anchors") and len(csv) == 1 + len(res[0]["exchanges"])


def test_r24m2_scripts_selective_archive():
    """glam_r24m2(.pilot).sbatch = r24m + minimum distance 3, score cut-off 0.5, quality tolerance 100, restart radii doubled."""
    import dataclasses, subprocess, sys
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    import run_glam
    for name, n_iter, tag in (("glam_r24m2.sbatch", 40000, "r24m2"), ("glam_r24m2_pilot.sbatch", 3000, "r24m2-pilot")):
        path = os.path.join(REPO, "cluster", "slurm", name)
        script = open(path).read()
        out = subprocess.run(["bash", "-c", f'source <(sed -n "/^GLAM_FLAGS=(/,/^)/p" {path}); printf "%s\\n" "${{GLAM_FLAGS[@]}}"'],
                             capture_output=True, text=True, check=True).stdout.split("\n")[:-1]
        cfg = C.from_cli(run_glam.build_parser().parse_args(out), C.GlamConfig)
        flags = {a[2:].replace("-", "_") for a in out if a.startswith("--")}
        assert {f.name for f in dataclasses.fields(C.GlamConfig)} <= flags
        assert cfg.n_iter == n_iter and cfg.max_anchors < 0
        assert (cfg.minimum_anchor_distance, cfg.minimum_diversity_score, cfg.quality_tolerance) == (3.0, 0.5, 100.0)
        assert (cfg.exploit_restart_scale, cfg.frontier_restart_scale, cfg.exploration_restart_scale, cfg.restart_scale) == (0.02, 0.06, 0.16, 0.06)
        assert cfg.restart_phi_tolerance == 30.0 and cfg.logdet_gain_scale == 10.0
        assert (cfg.exploit_fraction, cfg.frontier_fraction, cfg.explore_fraction) == (0.25, 0.25, 0.5)
        assert f"\nGLAM_TAG={tag} " in script and f"#SBATCH -J glam-{tag}\n" in script


def test_replay_archive_selection():
    """experiments/replay_archive_selection.py: load the newest anchor state and rethin it offline with other settings."""
    import pickle
    import sys
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    import replay_archive_selection as R
    from brho.glam.anchors import Anchor
    rng = np.random.default_rng(0)
    n_sub, m, n = 3, 40, 120
    axes = [rng.standard_normal(m) / np.sqrt(m) for _ in range(n_sub)]
    anchors = []
    for i in range(n):
        t = rng.uniform(0, 10)
        th = [t * axes[b] + 0.5 * rng.standard_normal(m) for b in range(n_sub)] + [np.array([0.1 * rng.standard_normal()])]
        anchors.append(Anchor(theta=th, phi=100.0 - t + rng.random(), anchor_id=i))
    state = dict(anchors=anchors, best_anchor=min(anchors, key=lambda a: a.phi), numsubd=n_sub)
    with tempfile.TemporaryDirectory() as d:
        for it in (10, 20):
            with open(os.path.join(d, f"anchor_state_ite{it:09d}_global.pkl"), "wb") as f:
                pickle.dump(state, f)
        st, it = R.load_state(d)
        assert it == 20 and len(st["anchors"]) == n
        assert R.load_state(d, 10)[1] == 10
    cfg = C.GlamConfig(max_anchors=-1, minimum_anchor_distance=0.5, minimum_diversity_score=0.5, logdet_gain_scale=10.0,
                       covariance_regularization=1e-6, quality_tolerance=100.0)

    class Args:
        eps, scale, mind, cap, minscore, alpha = "1e-6,0.1", "1,10", "0.5", "none,10", "", ""
    settings = R.grid(Args, cfg)
    assert settings[0]["label"] == "baseline" and len(settings) == 8       # baseline + 2*2*1*2 minus the duplicate of the baseline
    rows = R.replay(cfg, st, settings, seed=0, log=lambda *_: None)
    base = rows[0]
    assert base["n"] == n and base["rej_score"] == 0 and base["cover_max"] < 1e-6      # nothing to thin at eps 1e-6
    capped = [r for r in rows if r["cap"] == 10 and r["eps"] == 1e-6 and r["scale"] == 10.0][0]
    assert capped["n"] == 10 and capped["cap_hit"] == 1 and capped["cover_max"] > 1.0
    strict = [r for r in rows if r["eps"] == 0.1 and r["scale"] == 10.0 and r["cap"] == -1][0]
    assert strict["rej_score"] > 0 and strict["n"] < n                                  # large eps + scale 10: the cut-off bites
    for r in rows:
        assert 0.0 < r["top_share"] <= 1.0 and r["eff_rank"] >= 1.0 and r["dmin"] > 0.0


def _pcn_run(n_iter, out, resume=False, **kw):
    kw.setdefault("snapshot_every", 40)
    snap = kw.pop("snapshot_every")
    model = ToyModel(_beta(), n_sub=4, m=6, n_cells=60, n_times=3, seed=5, sigma_conc=0.05, sigma_dp=0.05)
    cfg = C.GlamConfig(n_iter=n_iter, window1=50, window2=0, burnin_length=10 ** 8, global_stage_length=10 ** 8,
                       local_stage_length=10 ** 8, min_global_stage_length=10 ** 8, min_local_stage_length=10 ** 8,
                       adapt_interval=10, beta_g0=0.1, save_every=25, log_every=0, **kw)
    res = ThreadWorld(3).run(lambda comm: run_glam(model, cfg, comm, out, resume=resume, log=lambda *_: None, snapshot_every=snap))
    return cfg, res


def test_pcn_snapshots_resume_and_extend_exactly():
    full, part = tempfile.mkdtemp(), tempfile.mkdtemp()
    cfg, a = _pcn_run(200, full, snapshot_every=0)
    _pcn_run(130, part)                                           # snapshots at 40, 80, 120 and the final one at 130
    for r in range(3):                                            # killed run: the final generation never got written
        os.remove(os.path.join(part, "anchor_state", f"chain_r{r:03d}_ite130.pkl"))
    _, b = _pcn_run(200, part, resume=True)                       # resumes from 120
    _pcn_run(120, full + "_x")
    _, c = _pcn_run(200, full + "_x", resume=True)                # finished run extended from its final snapshot
    for other in (b, c):
        for ra, rb in zip(a, other):
            assert all(np.array_equal(x, y) for x, y in zip(ra["theta"], rb["theta"]))
            assert ra["phi"] == rb["phi"]
    for d in (part, full + "_x"):
        for r in range(3):
            ca, cb = load_chain(full, r), load_chain(d, r)
            ia, ib = expand(ca, 200), expand(cb, 200)
            assert np.array_equal(ca["theta"][ia], cb["theta"][ib])
    assert len(glob.glob(os.path.join(part, "anchor_state", "chain_r000_ite*.pkl"))) <= 3
    import pytest
    model = ToyModel(_beta(), n_sub=4, m=6, n_cells=60, n_times=3, seed=5)
    gl = C.GlamConfig(n_iter=50, window1=10, window2=10, burnin_length=10, save_every=25, log_every=0)
    with pytest.raises(ValueError):                               # a GLAM run (exchanges on) refuses snapshot_every
        ThreadWorld(1).run(lambda comm: run_glam(model, gl, comm, tempfile.mkdtemp(), log=lambda *_: None, snapshot_every=10))


def test_fixed_steps_keep_the_given_step_sizes():
    out = tempfile.mkdtemp()
    model = ToyModel(_beta(), n_sub=4, m=6, n_cells=60, n_times=3, seed=5, sigma_conc=0.05, sigma_dp=0.05)
    cfg = C.GlamConfig(n_iter=120, window1=60, window2=60, burnin_length=20, global_stage_length=20, local_stage_length=15,
                       min_global_stage_length=10, min_local_stage_length=10, beta_g0=0.037, beta_l0=0.081, save_every=25, log_every=0)
    ThreadWorld(2).run(lambda comm: run_glam(model, cfg, comm, out, log=lambda *_: None, fixed_steps=True))
    from brho.glam import post
    for r in range(2):
        ch = post.load_chain(out, r)
        assert np.allclose(ch["beta_g"], 0.037) and np.allclose(ch["beta_l"], 0.081)
