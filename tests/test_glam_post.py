"""GLAM post-processing: stride bookkeeping with repetition counts, Welford merge, pointwise field statistics."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))

from brho.glam import post  # noqa: E402
from brho.glam.comm import ThreadWorld  # noqa: E402
from test_glam import _toy_run  # noqa: E402


def _expand(ch):
    """Brute force: per-iteration theta/w/phi from repetition-coded records."""
    idx = np.repeat(np.arange(len(ch["start"])), ch["count"])
    return ch["theta"][idx], ch["w"][idx], ch["phi"][idx]


def test_strided_samples_match_brute_force():
    out = tempfile.mkdtemp()
    model, cfg, _ = _toy_run(240, out)
    for stride, tail in [(1, 60), (7, 60), (50, 33)]:
        s = post.strided_samples(out, 4, tail, stride, cfg.n_iter)
        for r in range(4):
            ch = post.load_chain(out, r)
            th, w, ph = _expand(ch)
            assert post.n_iterations(ch) == cfg.n_iter and len(th) == cfg.n_iter
            t = np.arange(tail, cfg.n_iter, stride)
            m = s["rank"] == r
            assert np.array_equal(s["iteration"][m], t)
            assert np.allclose(s["theta"][m], th[t]) and np.allclose(s["w"][m], w[t]) and np.allclose(s["phi"][m], ph[t])


def test_welford_merge():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((37, 5, 3))
    a, b, c = post.Welford(), post.Welford(), post.Welford()
    for x in X[:10]: a.add(x)
    for x in X[10:30]: b.add(x)
    for x in X[30:]: c.add(x)
    a.merge(b).merge(c).merge(post.Welford())
    assert np.allclose(a.mean, X.mean(0)) and np.allclose(a.std(), X.std(0, ddof=1))


def test_field_statistics_match_numpy():
    out = tempfile.mkdtemp()
    model, cfg, _ = _toy_run(240, out)
    s = post.strided_samples(out, 4, cfg.window1, 20, cfg.n_iter)
    ref = {k: [] for k in ("pressure", "conc", "logperm", "dp")}
    for i in range(len(s["w"])):
        st = post.split_state(model, s["theta"][i], s["w"][i])
        f = model.build(st); sim = model.simulate(f, 0, want_pressure=True)
        ref["pressure"].append(sim.pressure); ref["conc"].append(sim.conc)
        ref["logperm"].append(f.prior_field["logK"]); ref["dp"].append(sim.dp_pa)
    res = ThreadWorld(3).run(lambda comm: post.field_statistics(model, s, comm))[0]
    assert res["n_used"] == len(s["w"]) and res["n_failed"] == 0
    for k, v in ref.items():
        v = np.array(v)
        assert np.allclose(res[f"{k}_mean"], v.mean(0)) and np.allclose(res[f"{k}_std"], v.std(0, ddof=1))


def test_gpu_projector_matches_cpu_projection_with_numpy_backend():
    import test_external as te
    from brho.multiscale_mat import MultiscaleField, attach_gpu_projector
    rng = np.random.default_rng(3)
    PHI, U, sv, cells, T_list = te._synthetic_multiscale(rng, N=288, M=30, n_sub=3)
    f = MultiscaleField(T_list, cells, PHI, U, 1.0 / np.sqrt(sv), dtype=np.float32)
    Y = rng.standard_normal(288).astype(np.float32)
    ref = f.whiten_project(Y); refk = f.whiten_project(np.stack([Y, 2 * Y], 1))
    attach_gpu_projector(f, np)
    assert np.allclose(f.whiten_project(Y), ref, rtol=1e-4, atol=1e-5)
    assert np.allclose(f.whiten_project(np.stack([Y, 2 * Y], 1)), refk, rtol=1e-4, atol=1e-5)


def test_best_record_is_the_lowest_phi_state():
    out = tempfile.mkdtemp()
    _, cfg, _ = _toy_run(240, out)
    for tail in (0, 100):
        b = post.best_record(out, 4, tail)
        best = (np.inf, None)
        for r in range(4):
            th, w, ph = _expand(post.load_chain(out, r))
            ph = ph[tail:]
            j = int(np.argmin(ph))
            if ph[j] < best[0]:
                best = (ph[j], (r, j + tail, th[j + tail], w[j + tail]))
        assert b["phi"] == best[0] and b["rank"] == best[1][0]
        assert np.allclose(b["theta"], best[1][2]) and np.isclose(b["w"], best[1][3])
        assert b["iteration"] >= tail


def test_plug_mask_and_best_fields():
    m = post.plug_mask(38, 38, 45).reshape((38, 38, 45), order="F")
    assert m.shape == (38, 38, 45) and np.all(m == m[:, :, :1])            # same disc in every layer
    assert 0.74 < m[:, :, 0].mean() < 0.80 and m[19, 19, 0] and not m[0, 0, 0]   # pi/4 of the square, centre in, corner out
    out = tempfile.mkdtemp()
    model, cfg, _ = _toy_run(160, out)
    b = post.best_record(out, 4)
    bf = post.best_fields(model, b)
    assert np.isclose(bf["phi"], b["phi"], rtol=1e-6) and bf["conc"].shape[0] == 60 and bf["logperm"].shape == (60,)


def test_conc_z_statistics_match_brute_force():
    out = tempfile.mkdtemp()
    model, cfg, _ = _toy_run(200, out)
    s = post.strided_samples(out, 4, cfg.window1, 20, cfg.n_iter)
    grid = (5, 4, 3)                                         # the toy has 60 cells
    plug = np.zeros((5, 4, 3), bool); plug[1:4, 1:3, :] = True; plug = plug.reshape(-1, order="F")
    zs, ps = [], []
    for i in range(len(s["w"])):
        st = post.split_state(model, s["theta"][i], s["w"][i])
        c = model.simulate(model.build(st), 0).conc.reshape((5, 4, 3, -1), order="F")
        disc = plug.reshape((5, 4, 3), order="F")[:, :, 0]
        zs.append(np.stack([c[:, :, k][disc].mean(0) for k in range(3)])); ps.append(c[disc].mean((0, 1)))
    res = ThreadWorld(3).run(lambda comm: post.field_statistics(model, s, comm, plug=plug, grid=grid))[0]
    assert np.allclose(res["conc_z_mean"], np.mean(zs, 0)) and np.allclose(res["conc_z_std"], np.std(zs, 0, ddof=1))
    assert np.allclose(res["conc_plug_mean"], np.mean(ps, 0)) and np.allclose(res["conc_plug_std"], np.std(ps, 0, ddof=1))
    assert res["conc_z_mean"].shape == (3, 3)


def test_field_figures_are_written():
    from brho.glam import fieldplots as fp
    nx, ny, nz, nt = 10, 10, 6, 4
    rng = np.random.default_rng(1)
    v = rng.standard_normal(nx * ny * nz); S = rng.random((nx * ny * nz, nt)); d = tempfile.mkdtemp()
    grid, dom = (nx, ny, nz), (10.0, 10.0)
    paths = [fp.plot_logperm_slice_stat(v, v, v, np.abs(v), grid, 2, os.path.join(d, "a.png"), dom),
             fp.plot_slices(v, grid, [1, 3, 6], os.path.join(d, "b.png"), dom),
             fp.plot_tracer_maps(S, grid, [1, 2, 3, 4], 3, os.path.join(d, "c.png"), dom),
             fp.plot_conc_vs_time(np.arange(nt), S.mean(0), S.mean(0), S.mean(0), 0.1 * S.std(0),
                                  os.path.join(d, "d.png")),
             fp.plot_conc_z_vs_time(np.arange(nt), np.arange(nz) + .5, rng.random((nz, nt)), 0.05 * np.ones((nz, nt)),
                                    [1, 3, 6], os.path.join(d, "e.png"), rng.random((nz, nt)), rng.random((nz, nt)))]
    assert all(os.path.getsize(p) > 5000 for p in paths)


def test_re_trace_reproduces_phi_at_every_iteration():
    # Phi = RE_conc^2 / (2 s^2) + RE_dp^2 / (2 s^2) for the toy (s = 0.05): holds through acceptances, rejections and restarts
    out = tempfile.mkdtemp()
    _, cfg, _ = _toy_run(240, out)
    for r in range(4):
        ch = post.load_chain(out, r)
        assert ch["re_conc"] is not None and len(ch["re_conc"]) == cfg.n_iter
        s2 = 2 * 0.05 ** 2
        phi = ch["re_conc"].astype(float) ** 2 / s2 + ch["re_dp"].astype(float) ** 2 / s2
        assert np.allclose(phi, ch["phi_trace"], rtol=2e-4), np.max(np.abs(phi / ch["phi_trace"] - 1))


def test_field_statistics_returns_per_sample_re():
    out = tempfile.mkdtemp()
    model, cfg, _ = _toy_run(200, out)
    s = post.strided_samples(out, 4, cfg.window1, 20, cfg.n_iter)
    res = ThreadWorld(3).run(lambda comm: post.field_statistics(model, s, comm))[0]
    assert np.array_equal(res["sample_index"], np.arange(len(s["w"])))
    for i in (0, 5, len(s["w"]) - 1):
        st = post.split_state(model, s["theta"][i], s["w"][i])
        phi, rc, rd = model.eval_state(st)
        assert np.isclose(res["sample_phi"][i], phi) and np.isclose(res["sample_re_conc"][i], rc)
        assert np.isclose(res["sample_re_dp"][i], rd)


def test_mpsrf_cholesky_matches_dense_and_diagonal_psrf():
    import run_glam  # noqa: F401  (experiments/ on the path for the next import)
    import glam_mpsrf as G
    from brho import mpsrf
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 300, 30)) @ rng.standard_normal((30, 30)) + 0.4 * rng.standard_normal((5, 1, 30))
    a = mpsrf.mpsrf_curve(x, skip=2, step=15, iter_burn=10)
    b = mpsrf.mpsrf_curve_direct(x, skip=2, step=15, iter_burn=10)
    assert np.nanmax(np.abs(a["mpsrf"] - b["mpsrf"])) < 1e-8
    ks, P = G.psrf_diag_curve(x, 2, 15, 10)
    assert np.array_equal(ks, a["k"]) and np.allclose(P[3:], a["psrf_i"][3:], rtol=1e-9)
    # dims specs and the read-out
    assert list(G.select_dims("first:3", [4, 4])[1]) == [0, 1, 2]
    assert list(G.select_dims("blocks:2", [4, 4])[1]) == [0, 1, 4, 5] and list(G.select_dims("rho", [4, 4])[1]) == [8]
    assert list(G.select_dims("block:1:2", [4, 4])[1]) == [4, 5] and len(G.select_dims("all", [4, 4])[1]) == 9
    its = np.arange(1, 21) * 10.0
    assert G.converged_at(its, np.r_[np.full(10, 1.0), np.full(10, 1.0)], 1.2, 10) == its[10]     # starts below: first eligible point
    assert G.converged_at(its, np.r_[np.full(12, 3.0), np.full(8, 1.0)], 1.2, 10) < its[12]       # interpolated crossing
    assert G.converged_at(its, np.full(20, 3.0), 1.2, 10) is None


def test_mpsrf_curve_survives_a_window_where_no_chain_moves():
    """A coordinate that is constant in every chain over the first window (rho during a local stage) makes W exactly singular: that
    curve point is NaN, the later points are computed (it used to raise LinAlgError and abort the whole MPSRF step)."""
    from brho import mpsrf
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 200, 1)) + 0.1
    x[:, :60, :] = 1.0                                    # frozen coordinate over the first 60 samples, same value in every chain
    r = mpsrf.mpsrf_curve(x, skip=2, step=15, iter_burn=10)
    assert np.isnan(r["mpsrf"][0]) and np.isfinite(r["mpsrf"][-1])


def test_windowed_acceptance():
    import plot_glam_diagnostics as D
    acc = np.array([1, 0, 1, 1, 0, 0, 0, 0, 1, 0] * 4, float); kind = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 4)
    c, r = D.windowed_acceptance(acc, kind, 0, 10, min_moves=5)
    assert np.allclose(r, 3 / 5) and np.allclose(c, [5, 15, 25, 35])
    c, r = D.windowed_acceptance(acc, kind, 1, 10, min_moves=5)
    assert np.allclose(r, 1 / 5)
    assert np.all(np.isnan(D.windowed_acceptance(acc, kind, 1, 10, min_moves=6)[1]))


def test_mpsrf_csv_roundtrip_and_standalone_replot():
    """The raw iteration-vs-MPSRF table is plain CSV (NaN kept) and experiments/replot_glam_mpsrf.py reads it back."""
    import importlib, subprocess, sys, tempfile
    d = tempfile.mkdtemp()
    G = importlib.import_module("glam_mpsrf")
    R = importlib.import_module("replot_glam_mpsrf")
    its = np.array([100., 200., 300., 400.])
    a = np.array([np.nan, 2.5, 1.3, 1.1]); b = np.array([np.nan, np.nan, 4.0, 1.25])
    path = os.path.join(d, "mpsrf_curves.csv")
    G.write_table(path, dict(study="e6_glam_x", stride=10, level=1.2, **{"dim[first:100]": 100}),
                  ["iteration", "first:100", "rho"], [its, a, b])
    meta, names, cols = R.read_table(path)
    assert names == ["iteration", "first:100", "rho"] and meta["stride"] == "10" and meta["dim[first:100]"] == "100"
    assert np.array_equal(cols["iteration"], its)
    assert np.array_equal(np.isnan(cols["first:100"]), np.isnan(a)) and np.allclose(cols["rho"][2:], b[2:])
    out = subprocess.run([sys.executable, "-I", os.path.join(REPO, "experiments", "replot_glam_mpsrf.py"), "--curves", path,
                          "--out-dir", d], capture_output=True, text=True, cwd=d)
    assert out.returncode == 0, out.stderr
    assert os.path.getsize(os.path.join(d, "mpsrf_curves.png")) > 1000
    assert "import brho" not in open(os.path.join(REPO, "experiments", "replot_glam_mpsrf.py")).read()   # standalone


def test_post_template_flags_exist_in_the_scripts_and_dims_specs_are_valid():
    """cluster/slurm/glam_post_template.sbatch: every flag of every step must be a real flag of its script, the
    conda environment is pinned, and the MPSRF --dims specs parse."""
    import re, subprocess
    path = os.path.join(REPO, "cluster", "slurm", "glam_post_template.sbatch")
    text = open(path).read()
    assert "export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC" in text
    assert '[ "${CONDA_PREFIX:-}" = "$BRHO_CONDA_ENV" ]' in text

    def array(name):
        out = subprocess.run(["bash", "-c", f'EXT_CFG=a EXT_REF_FIELD=b EXT_DATA_CONC=c; source <(sed -n "/^{name}=(/,/^)/p" {path}); '
                              f'printf "%s\\n" "${{{name}[@]}}"'], capture_output=True, text=True, check=True).stdout
        return out.split("\n")[:-1]

    scripts = dict(DIAG_FLAGS="plot_glam_diagnostics.py", TRACE_FLAGS="plot_glam_traces.py", POST_FLAGS="glam_postprocess.py",
                   FIELD_FLAGS="plot_glam_fields.py", EXPORT_FLAGS="export_glam_fields.py", MPSRF_FLAGS="glam_mpsrf.py")
    for arr, script in scripts.items():
        flags = [a for a in array(arr) if a.startswith("--")]
        assert flags, arr
        helptext = subprocess.run([sys.executable, os.path.join(REPO, "experiments", script), "--help"],
                                  capture_output=True, text=True).stdout
        for fl in flags:
            assert re.search(re.escape(fl) + r"(\s|,|\])", helptext), (script, fl)
    import glam_mpsrf as G
    dims = [6] * 12 + [0]
    for spec in array("MPSRF_DIMS"):
        label, idx = G.select_dims(spec, [500] * 12)
        assert len(idx) >= 1, spec


def test_reference_rho_recovers_the_sd_of_a_lognormal_field():
    import reference_rho as R
    rng = np.random.default_rng(3)
    grid = (20, 20, 12)
    logk = np.log(2.0) + 0.9 * rng.standard_normal(20 * 20 * 12)           # white field: sd 0.9 in every cell
    r = R.reference_rho(logk, grid)
    assert abs(r["plug"]["rho_eff"] - 0.9) < 0.03 and abs(r["all"]["rho_eff"] - 0.9) < 0.03
    assert abs(np.log(r["plug"]["beta_eff"]) - np.log(2.0)) < 0.03
    m = post.plug_mask(*grid)
    assert r["plug"]["n"] == int(m.sum()) and r["all"]["n"] == logk.size and len(r["plug"]["layer_sd"]) == 12
    # the plug is the only region that sets rho_eff: contaminate the outside and it must not change
    logk2 = logk.copy(); logk2[~m] = 5.0 + 3.0 * rng.standard_normal(int((~m).sum()))
    assert abs(R.reference_rho(logk2, grid)["plug"]["rho_eff"] - r["plug"]["rho_eff"]) < 1e-12


def test_projection_onto_the_model_space_matches_dense_least_squares():
    """fit_theta (LSQR through the field's own operators) reproduces numpy's dense least squares of P A theta = g."""
    from brho.glam import projection as PJ
    from brho.multiscale_mat import MultiscaleField
    rng = np.random.default_rng(5)
    N, M, sizes, m = 60, 25, [20, 20, 20], 8
    elem = [np.arange(i * 20, (i + 1) * 20) for i in range(3)]
    T = [rng.standard_normal((20, m)) for _ in range(3)]
    PHI = np.linalg.qr(rng.standard_normal((N, M)))[0]
    U = np.linalg.qr(rng.standard_normal((M, M)))[0]
    f = MultiscaleField(T, elem, PHI, U, rng.uniform(0.5, 2.0, M), dtype=np.float64)
    # dense matrix of P A
    cols = []
    for k in range(3 * m):
        th = np.zeros(3 * m); th[k] = 1.0
        cols.append(f.field([th[i * m:(i + 1) * m] for i in range(3)]))
    D = np.stack(cols, axis=1)
    op, dims = PJ.field_operator(f)
    x = rng.standard_normal(3 * m); y = rng.standard_normal(N)
    assert np.allclose(op.matvec(x), D @ x) and np.allclose(op.rmatvec(y), D.T @ y)       # operator and adjoint
    g_in = D @ rng.standard_normal(3 * m)                                                  # inside the model space
    r = PJ.fit_theta(f, g_in, iter_lim=500, atol=1e-12, btol=1e-12)
    assert r["residual"] < 1e-8
    g = rng.standard_normal(N)                                                             # generic target: residual > 0
    r = PJ.fit_theta(f, g, iter_lim=500, atol=1e-12, btol=1e-12)
    th_ls = np.linalg.lstsq(D, g, rcond=None)[0]
    assert abs(r["residual"] - np.linalg.norm(g - D @ th_ls) / np.linalg.norm(g)) < 1e-8
    assert np.allclose(r["yp"], D @ th_ls, atol=1e-7)


def test_check_reference_projection_script_end_to_end_with_a_stub_model():
    """Runs experiments/check_reference_projection.py main() with a small synthetic field and a fake solver: a reference
    field that lies inside the model space is recovered (resid ~ 0), the rows and files are written."""
    import json, types
    import run_glam, check_reference_projection as CR
    import brho.darcy_external as DE
    from brho import io
    from brho.glam.model import BetaPchip, SimData
    from brho.glam_likelihood import GlamData
    from brho.multiscale_mat import MultiscaleField
    rng = np.random.default_rng(11)
    grid = (6, 6, 4); N = 144; m = 10
    elem = [np.arange(i * 36, (i + 1) * 36) for i in range(4)]
    T = [rng.standard_normal((36, m)) for _ in range(4)]
    PHI = np.linalg.qr(rng.standard_normal((N, 60)))[0]
    U = np.linalg.qr(rng.standard_normal((60, 60)))[0]
    field = MultiscaleField(T, elem, PHI, U, rng.uniform(0.5, 2.0, 60), dtype=np.float64)
    beta = BetaPchip.from_fits_json(os.path.join(REPO, "runs", "juno_external_2026-10", "tables", "e3_external", "fits.json"))
    ref_beta, ref_rho = 9.869232667160132e-14, 1.0
    theta0 = [rng.standard_normal(m) for _ in range(4)]
    logk_ref = np.log(ref_beta) + ref_rho * field.field(theta0)                       # inside the model space
    tmp = tempfile.mkdtemp()
    kref = os.path.join(tmp, "Kref.dat"); np.savetxt(kref, np.exp(logk_ref))
    gfile = os.path.join(tmp, "G.dat"); np.savetxt(gfile, field.field(theta0))
    data = GlamData(rng.uniform(0, 1, (N, 3)), 3314.7)

    class Stub:
        pass
    stub = Stub(); stub.field, stub.beta, stub.data = field, beta, data
    stub.simulate = lambda K, rank=0, want_pressure=False: SimData(float(3314.7 * beta.Kexp / np.exp(np.mean(np.log(K)))),
                                                                 np.tile(np.clip(np.log(K)[:, None] * 0.0 + 0.5, 0, 1), (1, 3)))
    orig_build, orig_phys = run_glam.build_model, DE.physics_from_cfg
    run_glam.build_model = lambda a, c, r: stub
    DE.physics_from_cfg = lambda path: dict(nx=6, ny=6, nz=4, Lx=1, Ly=1, Lz=1, v=1, mu=1)
    old_argv, old_root = sys.argv, os.environ.get("BRHO_ROOT")
    os.environ["BRHO_ROOT"] = tmp
    sys.argv = ["x", "--exp-tag", "stub", "--ref-field", kref, "--gauss-field", gfile, "--rhos", "1.0", "--lsqr-iter", "400",
                "--cfg", "unused"]
    try:
        import importlib; importlib.reload(io)
        CR.main()
        exp = io.EXP_GLAM + "_stub"
        res = json.load(open(os.path.join(io.tables_dir(exp), "reference_projection.json")))
        rows = [r for r in res["rows"] if r["rho"] is not None]
        true_row = [r for r in rows if r["level"] == "true"][0]
        assert true_row["resid"] < 1e-5 and true_row["dlogk_rms_all"] < 1e-5, true_row          # reference is in the model space
        assert 0.1 < true_row["theta_msq"] < 10 and "phi" in true_row                            # |theta|^2 and the forward part ran
        curve_row = [r for r in rows if r["level"] == "curve"][0]
        assert curve_row["dlogk_rms_all"] > 1e-4          # the curve level differs from the true beta: the fit cannot be exact
        z = np.load(os.path.join(io.results_dir(exp), "reference_projection.npz"))
        assert "theta_rho1_true" in z.files and z["theta_rho1_true"].size == 4 * m
    finally:
        run_glam.build_model, DE.physics_from_cfg = orig_build, orig_phys
        sys.argv = old_argv
        if old_root is None: os.environ.pop("BRHO_ROOT", None)
        else: os.environ["BRHO_ROOT"] = old_root
        importlib.reload(io)


def test_reference_projection_job_flags_exist_and_conda_is_pinned():
    import re, subprocess
    path = os.path.join(REPO, "cluster", "slurm", "glam_reference_projection.sbatch")
    text = open(path).read()
    assert "export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC" in text
    assert '[ "${CONDA_PREFIX:-}" = "$BRHO_CONDA_ENV" ]' in text
    out = subprocess.run(["bash", "-c", f'source <(sed -n "/^CHECK_FLAGS=(/,/^)/p" {path}); printf "%s\\n" "${{CHECK_FLAGS[@]}}"'],
                         capture_output=True, text=True, check=True).stdout.split("\n")[:-1]
    helptext = subprocess.run([sys.executable, os.path.join(REPO, "experiments", "check_reference_projection.py"), "--help"],
                              capture_output=True, text=True).stdout
    flags = [a for a in out if a.startswith("--")]
    assert flags and all(re.search(re.escape(f) + r"(\s|,|\])", helptext) for f in flags), flags
    assert out[out.index("--ref-beta") + 1] == "9.869232667160132e-14" and out[out.index("--ref-rho") + 1] == "1.0"


def test_compare_fields_and_export_with_synthetic_files(tmp_path, monkeypatch):
    """plot_glam_fields.compare_fields statistics and export_glam_fields (.npy, VTK) on a small synthetic run."""
    import subprocess
    nx, ny, nz, nt = 8, 8, 5, 3
    n = nx * ny * nz
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(n)
    mean, std = ref + 0.1 * rng.standard_normal(n), 0.2 * np.ones(n)
    root = tmp_path / "root"
    monkeypatch.setenv("BRHO_ROOT", str(root))
    from brho import io
    exp = io.EXP_GLAM + "_t"
    pdir = os.path.join(io.results_dir(exp), "post"); os.makedirs(pdir)
    np.savez(os.path.join(pdir, "best_fields.npz"), logperm=ref + 0.05, conc=rng.random((n, nt)), pressure=rng.random(n),
             dp_pa=3300.0, phi=9.0, rho=0.97, beta=9.5e-14, rank=2, iteration=123)
    np.savez(os.path.join(pdir, "field_stats.npz"), logperm_mean=mean, logperm_std=std, conc_mean=rng.random((n, nt)))
    np.savetxt(tmp_path / "Kref.dat", np.exp(ref))
    np.savetxt(tmp_path / "tc.dat", rng.random((n, nt)))
    cfg = tmp_path / "DATA.cfg"
    cfg.write_text("")
    import export_glam_fields as X
    monkeypatch.setattr(X, "physics_from_cfg", lambda c: dict(nx=nx, ny=ny, nz=nz, Lx=0.008, Ly=0.008, Lz=0.005))
    monkeypatch.setattr(sys, "argv", ["x", "--exp-tag", "t", "--cfg", str(cfg), "--ref-field", str(tmp_path / "Kref.dat"),
                                      "--ref-conc", str(tmp_path / "tc.dat")])
    X.main()
    ed = os.path.join(pdir, "export")
    best = np.load(os.path.join(ed, "best_lnK.npy"))
    assert best.shape == (nx, ny, nz) and np.allclose(best.ravel(order="F"), ref + 0.05)
    assert np.allclose(np.load(os.path.join(ed, "best_minus_ref.npy")), 0.05)
    assert os.path.exists(os.path.join(ed, "ref_conc_t3.npy")) and os.path.exists(os.path.join(ed, "best_state.json"))
    vtk = open(os.path.join(ed, "glam_fields.vtk")).read().splitlines()
    assert vtk[3] == "DATASET STRUCTURED_POINTS" and vtk[4] == f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}"
    assert f"CELL_DATA {n}" in vtk and "SCALARS zscore double 1" in vtk
    i0 = vtk.index("SCALARS best_lnK double 1") + 2
    assert np.allclose([float(x) for x in vtk[i0:i0 + n]], ref + 0.05, atol=1e-6)      # x fastest, as in the solver
    import plot_glam_fields as F
    from brho.glam import post
    mask = post.plug_mask(nx, ny, nz)
    odir = tmp_path / "figs"; odir.mkdir()
    made = F.compare_fields(ref, ref + 0.05, mean, std, mask, (nx, ny, nz), [1, 3, 5], str(odir), (8.0, 8.0), exp)
    js = io.load_json(os.path.join(io.tables_dir(exp), "field_comparison.json"))
    assert abs(js["best"]["rmse"] - 0.05) < 1e-12 and js["posterior"]["coverage_2sd"] > 0.9
    assert all(os.path.getsize(p) > 0 for p in made)


def test_field_statistics_without_pressure_never_requests_it():
    calls = []

    class M:
        def rho_ok(self, st): return True
        def build(self, st):
            import types
            return types.SimpleNamespace(prior_field=dict(logK=np.ones(4)))
        def simulate(self, field, rank=0, want_pressure=False):
            import types
            calls.append(want_pressure)
            return types.SimpleNamespace(conc=np.ones((4, 2)), dp_pa=1.0, pressure=np.ones(4) if want_pressure else None)
        def phi_parts(self, st, field, sim): return (1.0, 0.1, 0.1)
    orig = post.split_state
    post.split_state = lambda model, th, w: None
    try:
        res = post.field_statistics(M(), dict(theta=np.zeros((3, 2)), w=np.zeros(3)), ThreadWorld(1).comm(0), want_pressure=False)
    finally:
        post.split_state = orig
    assert calls == [False] * 3 and res["n_used"] == 3 and "pressure_mean" not in res
