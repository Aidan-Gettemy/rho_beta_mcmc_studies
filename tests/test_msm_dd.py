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
from brho import msm_dd_gibbs as GB                   # noqa: E402
from brho import msm_dd_neumann as NB                 # noqa: E402


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


def test_gibbs_block_potential_matches_the_block_and_tiny_prior_keeps_h():
    """GibbsBlock re-implements the Robin block solve: at h = h_hat its potential equals Block.potential, and with a
    vanishing prior width the exact draw of h returns h_hat; with a wide prior the draw moves (it is data-driven)."""
    S, data = _setup()
    fd = _flux(S, data)
    grid = DD.DDGrid(S, S.primal, data, spline=DD.InterfaceSpline(S, data), flux=fd, robin_alpha=0.1)
    rng = np.random.default_rng(3)
    b = grid.blocks[5]                                            # an interior block: Robin on every side
    gb = GB.GibbsBlock(b, eps=1e-9, corr_len=4.0)
    assert gb.nh > 0
    c = rng.standard_normal(b.r_k)
    phi_ref, _ = b.potential(c)
    phi, Ainv, Gs, p = gb.potential(c, gb.h_hat0)
    assert abs(phi - phi_ref) < 1e-8 * max(1.0, abs(phi_ref))
    h = gb.draw_h(c, gb.h_hat0, Ainv, Gs, p, rng)
    assert np.abs(h - gb.h_hat0).max() < 1e-6 * np.abs(gb.h_hat0).max()
    gw = GB.GibbsBlock(b, eps=0.5, corr_len=4.0)
    h2 = gw.draw_h(c, gw.h_hat0, *gw.potential(c, gw.h_hat0)[1:], rng)
    assert np.abs(h2 - gw.h_hat0).max() > 1e-3 * np.abs(gw.h_hat0).max()


def test_gibbs_interior_flux_and_per_face_prior():
    """Interior flux faces add a likelihood term that is linear in h (the exact draw pulls the interior flux towards the
    data when the prior is wide); eps_meas sets the prior sd of h on measured faces only."""
    S, data = _setup()
    fd = _flux(S, data)
    grid = DD.DDGrid(S, S.primal, data, spline=DD.InterfaceSpline(S, data), flux=fd, robin_alpha=0.1)
    b = grid.blocks[5]
    rng = np.random.default_rng(4)
    c = rng.standard_normal(b.r_k)
    g0 = GB.GibbsBlock(b, eps=0.5, corr_len=4.0, fd=fd)
    g1 = GB.GibbsBlock(b, eps=0.5, corr_len=4.0, fd=fd, interior=True)
    assert g0.n_interior() == 0 and g1.n_interior() > 0
    phi0 = g0.potential(c, g0.h_hat0)[0]
    phi1, Ainv, Gs, p = g1.potential(c, g1.h_hat0)
    assert phi1 > phi0
    g1.wq = 1e8 * g1.wq                                            # flux dominates: the draw must fit the interior flux
    K = np.exp(b.log_field(c))
    res = lambda pp: np.linalg.norm(g1.fq - g1._T(K) * (pp[g1.fa] - pp[g1.fb]))
    h = g1.draw_h(c, g1.h_hat0, Ainv, Gs, p, rng)
    assert res(Ainv @ g1.rhs(Gs, h)) < res(p)
    gm = GB.GibbsBlock(b, eps=0.5, corr_len=4.0, fd=fd, eps_meas=0.01)
    d = np.sqrt(np.diag(gm.Sigma0))
    meas = np.array(gm.h_meas)
    assert meas.any() and (~meas).any() and d[meas].max() < d[~meas].min()


class _OracleFlux:
    """Exact flux at every face of the interface lines of a given global solution."""

    def __init__(self, K2, p2):
        self.K2, self.p2, self.n = K2, p2, K2.shape[0]

    def has_line(self, orient, line):
        return line in MF.interface_lines(self.n)

    def line_flux(self, orient, line, kind="cubic"):
        n, t = self.n, np.arange(self.n)
        if orient == "v":
            return MF.face_flux(self.K2, self.p2, np.full(n, line - 1), t, np.full(n, line), t)
        return MF.face_flux(self.K2, self.p2, t, np.full(n, line - 1), t, np.full(n, line))


def test_neumann_block_reproduces_global_solution_and_face_pressures_with_exact_flux():
    """Exact outflow as the Neumann datum reproduces the global pressure on the block (up to the constant for blocks without a
    Dirichlet side), the predicted face pressure equals the exact face value, and the constant is integrated out correctly."""
    S, data = _setup()
    K, p, K2, p2 = _global_solution(S)
    grid = DD.DDGrid(S, S.primal, data, exact=(K2, p2), robin_alpha=0.1)
    for b in grid.blocks:
        b.set_robin(_OracleFlux(K2, p2), 0.1, "cubic")
    saw_singular = saw_regular = False
    for b in grid.blocks:
        nb = NB.NeumannBlock(b, eps=0.03, g_rel=0.0045)
        Ainv, Gs = nb.system(K[b.cells])
        h = nb.h_hat0                                         # exact (and, for singular blocks, sum(h) = 0 up to rounding)
        ps = Ainv @ nb.rhs(Gs, h)
        pt = p[b.cells]
        if nb.singular:
            saw_singular = True
            assert abs(h.sum()) < 1e-8 * np.abs(h).sum() + 1e-12 and abs(ps.sum()) < 1e-10
            ps = ps + pt.mean()
        else:
            saw_regular = True
        assert np.abs(ps - pt).max() < 1e-6 * np.abs(pt).max()
        pf = ps[nb.h_cells] + h / (2.0 * K[b.cells][nb.h_cells])
        assert np.abs(pf - nb.g_face).max() < 1e-6 * np.abs(nb.g_face).max()
        r = nb.z - np.concatenate([ps[b.obs_loc] , pf])
        if nb.singular:
            w = nb.wv
            t0 = float(np.sum(w * r) / np.sum(w))
            prof = min(0.5 * np.sum(w * (r - t) ** 2) for t in t0 + np.linspace(-1e-2, 1e-2, 2001))
            assert abs(nb._phi_data(r) - prof) < 1e-6 * max(1.0, prof)
    assert saw_singular and saw_regular


def test_neumann_draw_h_keeps_zero_net_flux_and_prior_when_tiny():
    S, data = _setup()
    K, p, K2, p2 = _global_solution(S)
    grid = DD.DDGrid(S, S.primal, data, exact=(K2, p2), robin_alpha=0.1)
    for b in grid.blocks:
        b.set_robin(_OracleFlux(K2, p2), 0.1, "cubic")
    rng = np.random.default_rng(3)
    for b in grid.blocks:
        nb = NB.NeumannBlock(b, eps=0.03, g_rel=0.0045)
        c = rng.standard_normal(b.r_k)
        phi, Ainv, Gs, pp = nb.potential(c, nb.h_hat0)
        h = nb.draw_h(c, nb.h_hat0, Ainv, Gs, pp, rng)
        if nb.singular:
            assert abs(h.sum()) < 1e-8 * np.abs(h).sum()
        tiny = NB.NeumannBlock(b, eps=1e-7, g_rel=0.0045)
        h2 = tiny.draw_h(c, tiny.h_hat0, Ainv, Gs, pp, rng)
        assert np.abs(h2 - tiny.h_hat0).max() < 1e-5 * np.abs(tiny.h_hat0).max()


def test_overlap_flux_lines_moved_to_extended_boundaries():
    import argparse, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import run_dd_chains as R
    base = dict(grid="primal", sigma_rel=0.00013258252, data_seed=2026, data_relerr=0.003, smoothing=1e-3, neighbors=60,
                flux_noise_rel=0.003, flux_sigma_rel=0.00020045, overlap_halo=2, robin_alpha=0.1, robin_interp="cubic",
                primal_local_dim=6, dual_energy=0.95, overlap_energy=0.95, flux_lines="boundaries")
    S, data, sp, g = R.build(argparse.Namespace(**base))
    fd = g.flux_data
    assert fd.lines == [6, 10, 14, 18, 22, 26] and fd.m == 2 * 6 * 16
    assert abs(fd.sigma_rel - 0.003 / np.sqrt(fd.m)) < 1e-12                  # noise-level weight per datum
    # every interface side of every extended block lies on a measured line and is Robin
    sides = [(b, s) for b in g.blocks for s, v in b.bc.items() if isinstance(v, dict)]
    assert len(sides) == 48
    # default lines are unchanged when the option is off
    base.update(flux_lines="default")
    _, _, _, g0 = R.build(argparse.Namespace(**base))
    assert g0.flux_data.m == 224 and g0.flux_data.lines == MF.interface_lines(32)


def test_overlap_blend_is_a_partition_of_unity_and_reduces_to_core_assembly_without_halo():
    import argparse, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import run_dd_chains as R
    base = dict(grid="primal", sigma_rel=0.00013258252, data_seed=2026, data_relerr=0.003, smoothing=1e-3, neighbors=60,
                flux_noise_rel=0.003, flux_sigma_rel=0.00020045, overlap_halo=2, robin_alpha=0.1, robin_interp="cubic",
                primal_local_dim=6, dual_energy=0.95, overlap_energy=0.95, overlap_blend=True)
    ms = R.build(argparse.Namespace(**base))[3].ms
    W = ms.assembly_weights(32)
    assert np.allclose(W.sum(0), 1.0) and W.min() >= 0
    assert W[0][ms.block_core[0]].max() == 1.0                   # cells away from every halo belong to their block alone
    assert not np.allclose(ms.B, R.build(argparse.Namespace(**dict(base, overlap_blend=False)))[3].ms.B)
    B_blend0 = R.build(argparse.Namespace(**dict(base, overlap_halo=0, overlap_blend=True)))[3].ms.B
    B_core0 = R.build(argparse.Namespace(**dict(base, overlap_halo=0, overlap_blend=False)))[3].ms.B
    assert np.allclose(B_blend0, B_core0)                        # no halo: the cores partition the grid
