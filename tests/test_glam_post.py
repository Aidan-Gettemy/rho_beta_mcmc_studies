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
