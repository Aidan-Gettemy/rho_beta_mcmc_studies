"""Checks for the domain-decomposed sampler with spline interfaces (brho/msm_dd.py)."""
import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho import msm_local as ML                      # noqa: E402
from brho import msm_sampler as MS                    # noqa: E402
from brho import msm_dd as DD                         # noqa: E402
from brho import msm_flux as MF                       # noqa: E402


@functools.lru_cache(maxsize=None)
def _setup():
    S = ML.MSMSetup(primal_local_dim=6)
    return S, MS.Data(S, 0.003, 2026)


def _global_solution(S, seed=5):
    theta = np.random.default_rng(seed).standard_normal(S.r)
    K = np.exp(S.kle.field(theta))
    p = S.phys.solve(K)
    n = S.n
    return K, p, K.reshape(n, n), p.reshape(n, n)


def test_local_solves_reproduce_global_solution_with_exact_faces():
    """With the exact face pressures of the global solution as Dirichlet data every block reproduces the global
    pressure on its cells, and the local potentials add up to the global one."""
    S, data = _setup()
    K, p, K2, p2 = _global_solution(S)
    phi_glob = 0.5 * data.w * float(np.sum((data.d - p[S.phys.obs_mask]) ** 2))
    for ms in (S.primal, S.dual):
        grid = DD.DDGrid(S, ms, data, exact=(K2, p2))
        phi_sum = 0.0
        for b in grid.blocks:
            pl = b.solver.solve(K[b.cells])
            assert np.abs(pl - p[b.cells]).max() < 1e-9 * np.abs(p).max(), (len(ms.r_k), b.k)
            phi_sum += 0.5 * data.w * float(np.sum((b.d_loc - pl[b.obs_loc]) ** 2))
        assert abs(phi_sum - phi_glob) < 1e-8 * max(1.0, phi_glob)


def test_single_block_is_the_global_permeameter():
    """One block covering the whole square with the external BCs equals brho.fv.Permeameter (bc='dirichlet')."""
    S, _ = _setup()
    K, p, _, _ = _global_solution(S, 9)
    n = S.n
    bs = DD.BlockSolver(n, n, dict(L=np.full(n, S.phys.p_left), R=np.full(n, S.phys.p_right), B=None, T=None))
    assert np.abs(bs.solve(K) - p).max() < 1e-10


def test_spline_reproduces_linear_pressure_without_smoothing():
    S, data = _setup()

    class Lin:                                              # stand-in data: p = 0.3 + 0.5 x - 0.2 y at the sensors
        pass
    d = Lin()
    xy = S.phys.points[S.phys.obs_mask]
    d.d = 0.3 + 0.5 * xy[:, 0] - 0.2 * xy[:, 1]
    sp = DD.InterfaceSpline(S, d, smoothing=0.0)
    q = np.array([[0.25, 0.13], [0.5, 0.77], [0.31, 0.5], [0.9, 0.03]])
    assert np.abs(sp(q) - (0.3 + 0.5 * q[:, 0] - 0.2 * q[:, 1])).max() < 1e-8


def test_block_fields_and_assembly_match_the_msm_objects():
    S, data = _setup()
    sp = DD.InterfaceSpline(S, data)
    rng = np.random.default_rng(0)
    for ms in (S.primal, S.dual):
        grid = DD.DDGrid(S, ms, data, spline=sp)
        assert grid.r_k == list(ms.r_k) and grid.p == ms.p
        c_all = rng.standard_normal(ms.p)
        cb = DD.split_blocks(ms, c_all)
        assert np.allclose(np.concatenate(cb), c_all)
        assert np.allclose(grid.theta(cb), ML.theta_from_c(ms, c_all))
        f = ms.local_field(c_all)                           # piecewise field of the repo's local KLE
        for b, ck in zip(grid.blocks, cb):
            assert np.abs(b.log_field(ck) - f[b.cells]).max() < 1e-12
        # every sensor belongs to exactly one block
        assert sum(len(b.d_loc) for b in grid.blocks) == data.n_obs


def test_interface_report_and_chain_smoke():
    S, data = _setup()
    sp = DD.InterfaceSpline(S, data)
    n = S.n
    K = np.exp(S.kle.field(data.theta_true)).reshape(n, n)
    p = S.phys.solve(K.ravel()).reshape(n, n)
    grid = DD.DDGrid(S, S.primal, data, spline=sp)
    rep = grid.interface_report(K, p, sp)
    assert len(rep) == 24                                    # 3 vertical lines x 4 + 3 horizontal lines x 4
    assert max(r["eps_int"] for r in rep) < 0.02             # study value is ~0.4% (mean), 0.6% worst
    o = DD.run_block_chain(grid.blocks[5], [7, 0, 5, 0], 400, 200, save_every=50)
    assert o["c"].shape == (8, 6) and o["acc"].shape == (400,)
    assert np.all(np.isfinite(o["phi"])) and 0.0 < o["acc"].mean() < 1.0


# ------------------------------------------------------------ flux data (brho/msm_flux.py)
def _flux(S, data):
    return MF.FluxData(S, data)


def test_flux_data_layout_and_noise():
    S, data = _setup()
    fd = _flux(S, data)
    assert fd.m == 14 * 16 and sorted(set(fd.line)) == [4, 8, 12, 16, 20, 24, 28]
    assert abs(fd.relerr_truth - 0.003) < 0.0008                       # noise as for the pressure data
    d2 = MS.Data(S, 0.003, 2026)
    assert np.array_equal(d2.d, data.d)                                # pressure data untouched


def test_global_flux_potential_at_truth():
    from brho import msm_flux as MF
    S, data = _setup()
    fd = _flux(S, data)
    gd = MF.FluxGlobalData(data, fd)
    phi, rel = gd.potential(data.theta_true)
    phi_p, rel_p = data.potential(data.theta_true)
    assert abs(phi - phi_p - fd.phi_truth) < 1e-8 * max(1.0, phi) and abs(rel - rel_p) < 1e-12


def test_block_flux_equals_true_flux_with_exact_faces():
    """Exact face pressures + true K: the one-sided block flux on every measured face is the true flux, so the
    block flux potentials add up to the noise-only potential (each face counted by both neighbours)."""
    S, data = _setup()
    fd = _flux(S, data)
    n = S.n
    K = np.exp(S.kle.field(data.theta_true))
    p = S.phys.solve(K)
    for ms, nlines in ((S.primal, 3), (S.dual, 4)):
        grid = DD.DDGrid(S, ms, data, exact=(K.reshape(n, n), p.reshape(n, n)), flux=fd)
        tot, nf = 0.0, 0
        for b in grid.blocks:
            pl = b.solver.solve(K[b.cells])
            tot += b.flux.phi(b, pl, K[b.cells])
            nf += b.flux.n_faces
        assert nf == 2 * 2 * nlines * 16                                # two lines families x lines x 16, both sides
        assert abs(tot - 2 * 0.5 * fd.w * float(np.sum((fd.q - fd.q_true)[np.isin(fd.line, [8, 16, 24] if nlines == 3 else [4, 12, 20, 28])] ** 2))) < 1e-8 * max(1.0, tot)


# ------------------------------------------------------------ Robin interface condition and overlapping blocks
def test_robin_with_exact_data_reproduces_the_global_solution():
    """Robin condition built from the EXACT face pressure and the EXACT flux on every face of every interface line
    (any alpha): the block solution is the global one, since the true solution satisfies alpha p + K dp/dn =
    alpha g + h identically."""
    S, data = _setup()
    fd = _flux(S, data)
    n = S.n
    K = np.exp(S.kle.field(data.theta_true))
    p = S.phys.solve(K)
    K2, p2 = K.reshape(n, n), p.reshape(n, n)

    def exact_line_flux(orient, line, kind="cubic"):
        t = np.arange(n)
        if orient == "v":
            return MF.face_flux(K2, p2, np.full(n, line - 1), t, np.full(n, line), t)
        return MF.face_flux(K2, p2, t, np.full(n, line - 1), t, np.full(n, line))
    fd.line_flux = exact_line_flux
    for ms in (S.primal, S.dual):
        for alpha in (0.3, 5.0):
            grid = DD.DDGrid(S, ms, data, exact=(K2, p2), flux=fd, robin_alpha=alpha)
            assert all(b.robin for b in grid.blocks)
            for b in grid.blocks:
                assert np.abs(b.solver.solve(K[b.cells]) - p[b.cells]).max() < 1e-9 * np.abs(p).max(), (alpha, b.k)


def test_robin_alpha_infinite_is_dirichlet_and_small_alpha_uses_flux():
    S, data = _setup()
    fd = _flux(S, data)
    sp = DD.InterfaceSpline(S, data)
    g_d = DD.DDGrid(S, S.primal, data, spline=sp)
    g_r = DD.DDGrid(S, S.primal, data, spline=sp, flux=fd, robin_alpha=1e12)
    K = np.exp(S.kle.field(data.theta_true))
    for bd, br in zip(g_d.blocks, g_r.blocks):
        assert np.abs(bd.solver.solve(K[bd.cells]) - br.solver.solve(K[br.cells])).max() < 1e-6
    g_s = DD.DDGrid(S, S.primal, data, spline=sp, flux=fd, robin_alpha=0.05)
    assert any(np.abs(bd.solver.solve(K[bd.cells]) - bs.solver.solve(K[bs.cells])).max() > 1e-5
               for bd, bs in zip(g_d.blocks, g_s.blocks))


def test_overlap_blocks_core_sensors_and_exact_faces():
    from brho import msm_dd_overlap as OV
    S, data = _setup()
    K = np.exp(S.kle.field(data.theta_true))
    p = S.phys.solve(K)
    n = S.n
    for ms in (S.primal, S.dual):
        ov = OV.OverlapDecomp(S, ms, 2)
        assert ov.B.shape == (S.r, ov.p) and all(r >= r0 for r, r0 in zip(ov.r_k, ms.r_k))
        grid = DD.DDGrid(S, ov, data, exact=(K.reshape(n, n), p.reshape(n, n)))
        assert sum(len(b.d_loc) for b in grid.blocks) == data.n_obs         # every sensor counted once (core only)
        phi_glob = 0.5 * data.w * float(np.sum((data.d - p[S.phys.obs_mask]) ** 2))
        phi_sum = 0.0
        for b in grid.blocks:
            pl = b.solver.solve(K[b.cells])
            assert np.abs(pl - p[b.cells]).max() < 1e-9 * np.abs(p).max()
            phi_sum += 0.5 * data.w * float(np.sum((b.d_loc - pl[b.obs_loc]) ** 2))
        assert abs(phi_sum - phi_glob) < 1e-8 * max(1.0, phi_glob)
