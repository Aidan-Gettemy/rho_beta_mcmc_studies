"""GLAM post-processing: stride bookkeeping with repetition counts, Welford merge, pointwise field statistics."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))

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
