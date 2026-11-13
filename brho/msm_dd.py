"""
Domain-decomposed MCMC (DD-MCMC) of Xu, Liao & Li (2022), "Domain-decomposed Bayesian inversion based on local
Karhunen-Loeve expansions", with a 2D SPLINE interface treatment in place of the paper's 1D Gaussian process
(docs/dd_mcmc_experiment.md).  Same problem as the MSM study (brho/msm_local.py, brho/msm_sampler.py): 32 x 32
Dirichlet permeameter, checkerboard sensors, truth and data of MS.Data.

Method
  * The unit square is split into rectangular blocks (the primal 4 x 4 or the dual 5 x 5 decomposition of
    brho.multiscale.MSInnovation; block cells, local KLE psi / mu are taken from that object).
  * The pressure on the interfaces is NOT unknown to the local problems: a thin-plate spline fitted once to all
    sensor data (scipy RBFInterpolator, smoothing SPLINE_SMOOTHING) is evaluated at the face centres of every
    interface and used as a Dirichlet value (half-cell face, T = 2 K) of the local finite-volume problem.
    External sides keep the global BCs (p = p_left / p_right on x = 0 / 1, no flow on y = 0 / 1).
  * Block i has its own local KLE field a_i = psi_i sqrt(mu_i) c_i (raw, not projected), prior c_i ~ N(0, I),
    its own forward model and a local likelihood that is the GLOBAL potential restricted to the block's sensors,
    Phi_i = w/2 sum_{s in block} (d_s - p_s)^2 with the global weight w of MS.Data.  Hence sum_i Phi_i = Phi when
    the interface pressures are exact (tests/test_msm_dd.py).
  * With the interface data frozen the blocks are independent; each runs its own pCN chain (one step size per
    block, adapted to 25% acceptance for the first M iterations, then frozen; acceptance on the likelihood only).
  * Global field: theta = B c_all, B = the repo's assemble-and-project matrix (global-KLE L2 projection of the
    zero-extended local modes), i.e. Definition 2 / Theorem 1 of the paper; c_all concatenates the blocks' local
    coefficients in block order (the column order of B).
"""
import numpy as np
from scipy.interpolate import RBFInterpolator

from .samplers import StepAdapter

SPLINE_KERNEL = "thin_plate_spline"
SPLINE_SMOOTHING = 1e-3
SPLINE_NEIGHBORS = 60
SIDES = ("L", "R", "B", "T")


# ------------------------------------------------------------ interface spline
class InterfaceSpline:
    """Thin-plate spline of the pressure through all sensor data (xy, d).  One object serves every interface of
    every decomposition; only the evaluation points differ."""

    def __init__(self, S, data, smoothing=SPLINE_SMOOTHING, neighbors=SPLINE_NEIGHBORS):
        self.xy = np.asarray(S.phys.points)[S.phys.obs_mask]        # sensor order of data.d
        self.smoothing, self.neighbors = smoothing, neighbors
        self.f = RBFInterpolator(self.xy, np.asarray(data.d, float), kernel=SPLINE_KERNEL,
                                 smoothing=smoothing, neighbors=min(neighbors, len(self.xy)))

    def __call__(self, xy):
        return self.f(np.asarray(xy, float))


def _side_faces(n, ix0, ix1, iy0, iy1, side):
    """Face centres (m, 2) of one interior side of the block [ix0..ix1] x [iy0..iy1] and the two cells that share
    each face (lower/left cell (ia, ja), upper/right cell (ib, jb))."""
    h = 1.0 / n
    if side in ("L", "R"):
        iy = np.arange(iy0, iy1 + 1)
        ix_face = ix0 if side == "L" else ix1 + 1                   # face between cell ix_face-1 and ix_face
        xy = np.stack([np.full(len(iy), ix_face * h), (iy + 0.5) * h], 1)
        cells = (np.full(len(iy), ix_face - 1), iy, np.full(len(iy), ix_face), iy)
    else:
        ix = np.arange(ix0, ix1 + 1)
        iy_face = iy0 if side == "B" else iy1 + 1
        xy = np.stack([(ix + 0.5) * h, np.full(len(ix), iy_face * h)], 1)
        cells = (ix, np.full(len(ix), iy_face - 1), ix, np.full(len(ix), iy_face))
    return xy, cells


def exact_face_values(K2, p2, cells):
    """Face pressure of a finite-volume solution (two-point flux continuity, half-cell links):
    p_face = (K_a p_a + K_b p_b) / (K_a + K_b)."""
    ia, ja, ib, jb = cells
    Ka, Kb = K2[ia, ja], K2[ib, jb]
    return (Ka * p2[ia, ja] + Kb * p2[ib, jb]) / (Ka + Kb)


# ------------------------------------------------------------ local forward model
class BlockSolver:
    """Cell-centred finite-volume solve of -div(K grad p) = 0 on an nxl x nyl block of unit-aspect cells
    (harmonic transmissibilities, as brho.fv.Permeameter) with Dirichlet half-cell faces on the sides in
    `bc` (side -> value array along that side) and no flow on the others.  Dense symmetric positive definite
    system (at most 64 unknowns in this study)."""

    def __init__(self, nxl, nyl, bc):
        self.nxl, self.nyl, self.N = nxl, nyl, nxl * nyl
        idx = np.arange(self.N).reshape(nxl, nyl)
        self.ax, self.bx = idx[:-1, :].ravel(), idx[1:, :].ravel()
        self.ay, self.by = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        side_cells = dict(L=idx[0, :], R=idx[-1, :], B=idx[:, 0], T=idx[:, -1])
        # side value: array g (Dirichlet) or dict(g=, h=, ainv=) for a mixed Dirichlet/Robin side: per face a datum g (pressure),
        # h = K dp/dn_out (minus the outward Darcy flux) and ainv = 1/alpha (0 = Dirichlet face); the Robin face
        # alpha p + K dp/dn = alpha g + h is a link of conductance 1/(1/T + ainv) to the pressure g + ainv h (T = 2K).
        self.dirichlet = []
        for s, v in bc.items():
            if v is None:
                continue
            if isinstance(v, dict):
                g, h, ainv = (np.asarray(v[k], float) for k in ("g", "h", "ainv"))
            else:
                g = np.asarray(v, float)
                h = ainv = np.zeros_like(g)
            self.dirichlet.append((side_cells[s], g + ainv * h, ainv))
        if not self.dirichlet:
            raise ValueError("a block needs at least one Dirichlet (or Robin) side")
        for ids, v, _ in self.dirichlet:
            if len(ids) != len(v):
                raise ValueError("boundary value array does not match its side")
        self.diag_pos = np.arange(self.N) * (self.N + 1)

    def solve(self, K):
        K = np.asarray(K, float).ravel()
        N = self.N
        ax, bx, ay, by = self.ax, self.bx, self.ay, self.by
        Tx = 2.0 * K[ax] * K[bx] / (K[ax] + K[bx])
        Ty = 2.0 * K[ay] * K[by] / (K[ay] + K[by])
        A = np.zeros((N, N))
        A[ax, bx] = -Tx
        A[bx, ax] = -Tx
        A[ay, by] = -Ty
        A[by, ay] = -Ty
        diag = (np.bincount(ax, Tx, N) + np.bincount(bx, Tx, N)
                + np.bincount(ay, Ty, N) + np.bincount(by, Ty, N))
        b = np.zeros(N)
        for ids, vals, ainv in self.dirichlet:
            t = 2.0 * K[ids]
            if np.any(ainv):
                t = t / (1.0 + t * ainv)                                 # series conductance of the Robin face
            diag[ids] += t
            b[ids] += t * vals
        A.flat[self.diag_pos] = diag
        return np.linalg.solve(A, b)


class Block:
    """One subdomain: geometry, local KLE, sensors, boundary data and the local forward model / potential."""

    def __init__(self, S, ms, k, data, bc_values, kinds, faces):
        n = S.n
        cells = np.asarray(ms.block_cells[k])
        ix, iy = cells // n, cells % n
        self.k, self.cells = k, cells
        self.ix0, self.ix1, self.iy0, self.iy1 = int(ix.min()), int(ix.max()), int(iy.min()), int(iy.max())
        self.nxl, self.nyl = self.ix1 - self.ix0 + 1, self.iy1 - self.iy0 + 1
        if self.nxl * self.nyl != len(cells) or not np.all(np.diff(cells) > 0):
            raise ValueError(f"block {k} is not an ascending rectangle of cells")
        self.psi = np.asarray(ms.block_psi[k])                       # (N_loc, r_k)
        self.sqmu = np.sqrt(np.asarray(ms.block_mu[k]))
        self.r_k = self.psi.shape[1]
        self.kinds, self.faces, self.bc = kinds, faces, bc_values
        self.solver = BlockSolver(self.nxl, self.nyl, bc_values)
        # local sensors: positions in the block's cell order and their data
        gpos = -np.ones(S.phys.N, int)
        gpos[np.flatnonzero(S.phys.obs_mask)] = np.arange(data.n_obs)
        sens = gpos[cells]
        core = getattr(ms, "block_core", None)                       # overlapping blocks: only the core's sensors count
        if core is not None and not getattr(ms, "halo_sensors", False):
            sens = np.where(np.isin(cells, core[k]), sens, -1)
        self.obs_loc = np.flatnonzero(sens >= 0)
        self.d_loc = np.asarray(data.d, float)[sens[self.obs_loc]]
        self.dnorm_loc = float(np.linalg.norm(self.d_loc))
        self.w = data.w
        self.flux = None                                             # BlockFlux (brho/msm_flux.py) or None
        self.robin = False

    def set_robin(self, fd, alpha, kind="cubic"):
        """Robin condition alpha p + K dp/dn = alpha g + h on every interface side that lies on a flux-data line: g the
        spline pressure already in self.bc, h = -(outward Darcy flux) from the flux data interpolated along the line
        (brho/msm_flux.py).  Sides off the flux lines stay Dirichlet.  The solver is rebuilt; no flux penalty is used."""
        sign = dict(L=+1.0, R=-1.0, B=+1.0, T=-1.0)
        bc = {}
        for side, v in self.bc.items():
            if v is None or self.kinds[side] != "iface":
                bc[side] = v
                continue
            xy, (ia, ja, ib, jb) = self.faces[side]
            vertical = side in ("L", "R")
            line = int(ib[0]) if vertical else int(jb[0])
            orient = "v" if vertical else "h"
            if not fd.has_line(orient, line):
                bc[side] = v
                continue
            along = ja if vertical else ia
            q = fd.line_flux(orient, line, kind)[along]
            bc[side] = dict(g=np.asarray(v, float), h=sign[side] * q, ainv=np.full(len(v), 1.0 / alpha))
        self.bc = bc
        self.solver = BlockSolver(self.nxl, self.nyl, bc)
        self.robin = True

    def log_field(self, c):
        return self.psi @ (self.sqmu * np.asarray(c, float))

    def pressure(self, c):
        return self.solver.solve(np.exp(self.log_field(c)))

    def potential(self, c):
        """(Phi_i, local relative error) of local coefficients c."""
        lf = self.log_field(c)
        p = self.solver.solve(np.exp(lf))
        r2 = float(np.sum((self.d_loc - p[self.obs_loc]) ** 2))
        phi = 0.5 * self.w * r2
        if self.flux is not None:                                    # interface-flux penalty (flux studies)
            phi += self.flux.phi(self, p, np.exp(lf))
        return phi, np.sqrt(r2) / self.dnorm_loc


class DDGrid:
    """All blocks of one decomposition `ms` (an MSInnovation) with boundary data from `spline` (the method) or
    from an exact finite-volume solution `exact = (K2, p2)` (tests)."""

    def __init__(self, S, ms, data, spline=None, exact=None, flux=None, robin_alpha=None, robin_interp="cubic"):
        if (spline is None) == (exact is None):
            raise ValueError("pass exactly one of spline / exact")
        self.S, self.ms, self.n = S, ms, S.n
        n = S.n
        extents = []
        for k in range(len(ms.r_k)):
            c = np.asarray(ms.block_cells[k])
            extents.append((int((c // n).min()), int((c // n).max()), int((c % n).min()), int((c % n).max())))
        self.blocks = []
        self.block_of_cell = np.empty(S.phys.N, int)
        for k, c in enumerate(ms.block_cells):
            self.block_of_cell[np.asarray(c)] = k
        for k, (ix0, ix1, iy0, iy1) in enumerate(extents):
            nyl, nxl = iy1 - iy0 + 1, ix1 - ix0 + 1
            bc, kinds, faces = {}, {}, {}
            for side in SIDES:
                ext = dict(L=ix0 == 0, R=ix1 == n - 1, B=iy0 == 0, T=iy1 == n - 1)[side]
                length = nyl if side in ("L", "R") else nxl
                if ext:
                    if side == "L":
                        bc[side], kinds[side] = np.full(length, S.phys.p_left), "ext"
                    elif side == "R":
                        bc[side], kinds[side] = np.full(length, S.phys.p_right), "ext"
                    else:
                        bc[side], kinds[side] = None, "noflow"
                    continue
                xy, cells = _side_faces(n, ix0, ix1, iy0, iy1, side)
                bc[side] = spline(xy) if spline is not None else exact_face_values(exact[0], exact[1], cells)
                kinds[side], faces[side] = "iface", (xy, cells)
            self.blocks.append(Block(S, ms, k, data, bc, kinds, faces))
        self.r_k = list(ms.r_k)
        self.p = int(ms.p)
        if flux is not None and robin_alpha is not None:             # flux enters the boundary condition (Robin), not the likelihood
            for b in self.blocks:
                b.set_robin(flux, robin_alpha, robin_interp)
        elif flux is not None:                                       # flux studies: FluxData of brho/msm_flux.py
            from .msm_flux import BlockFlux
            for b in self.blocks:
                b.flux = BlockFlux(flux, b)

    def theta(self, c_blocks):
        """Global theta (..., r) from the blocks' local coefficients (list of (..., r_k) arrays)."""
        return assemble_theta(self.ms, c_blocks)

    def interface_report(self, K2, p2, spline):
        """Relative L2 error of the spline's face pressures against the face pressures of the finite-volume
        solution (K2, p2) (n x n arrays), one entry per interface segment (counted once, from the block on the
        left / below)."""
        out = []
        for b in self.blocks:
            for side, orient in (("R", "vertical"), ("T", "horizontal")):
                if b.kinds[side] != "iface":
                    continue
                xy, cells = b.faces[side]
                truth = exact_face_values(K2, p2, cells)
                est = spline(xy)
                nb = int(self.block_of_cell[cells[2][0] * self.n + cells[3][0]])
                out.append(dict(block=b.k, neighbour=nb, orientation=orient, n_faces=int(len(truth)),
                                eps_int=float(np.linalg.norm(est - truth) / np.linalg.norm(truth))))
        return out


def assemble_theta(ms, c_blocks):
    """theta = B c_all; c_all = the blocks' coefficients concatenated in block order (column order of B)."""
    c = np.concatenate([np.asarray(x, float) for x in c_blocks], axis=-1)
    return c @ ms.B.T


def split_blocks(ms, c_all):
    """Inverse of the concatenation: list of the blocks' coefficient arrays from c_all (..., p)."""
    c_all = np.asarray(c_all)
    off = np.concatenate([[0], np.cumsum(ms.r_k)])
    return [c_all[..., off[k]:off[k + 1]] for k in range(len(ms.r_k))]


# ------------------------------------------------------------ chains
def pilot_step(s_hist, frac=0.3):
    """Frozen step size from a pilot run: geometric mean of the saved step sizes of all chains over the last `frac`
    of the run.  s_hist: (chains, n_save) or (chains, n_save, 1)."""
    s = np.asarray(s_hist, float)
    s = s.reshape(s.shape[0], s.shape[1])
    return float(np.exp(np.log(s[:, int(s.shape[1] * (1 - frac)):]).mean()))


def run_block_chain(blk, seed_key, n_iter, n_adapt, s0=0.2, save_every=50, target=0.25, every=100, init_scale=1.0):
    """One pCN chain of one block (one forward solve per iteration).  seed_key: a sequence of ints,
    e.g. [seed0, grid_tag, block, chain].  Returns
        c (n_iter // save_every, r_k)  state at iterations 0, save_every, ... (before the step)
        s_hist (n_save,)               step size at those iterations
        acc (n_iter,) uint8, phi (n_iter,) float32, relerr (n_iter,) float32   (state after each iteration)"""
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_key)))
    ad = StepAdapter([s0], targets=[target], every=every)
    c = init_scale * rng.standard_normal(blk.r_k)                 # dispersed start (prior draw, optionally widened)
    phi, rel = blk.potential(c)
    n_save = n_iter // save_every
    out = dict(c=np.empty((n_save, blk.r_k), np.float32), s_hist=np.empty(n_save, np.float32),
               acc=np.zeros(n_iter, np.uint8), phi=np.empty(n_iter, np.float32),
               relerr=np.empty(n_iter, np.float32))
    for it in range(n_iter):
        if it % save_every == 0:
            j = it // save_every
            out["c"][j], out["s_hist"][j] = c, ad.s[0]
        s = ad.s[0]
        c_new = np.sqrt(1.0 - s * s) * c + s * rng.standard_normal(blk.r_k)
        phi_new, rel_new = blk.potential(c_new)
        accepted = np.log(rng.random()) < phi - phi_new
        if accepted:
            c, phi, rel = c_new, phi_new, rel_new
        ad.record(0, accepted)
        ad.tick(it, n_adapt)
        out["acc"][it], out["phi"][it], out["relerr"][it] = accepted, phi, rel
    return out
