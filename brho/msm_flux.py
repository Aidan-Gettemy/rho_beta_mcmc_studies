"""
Interface FLUX data for the domain-decomposed study (docs/dd_flux_experiment.md).

Flux is measured on the faces of the interface lines of BOTH decompositions (primal lines at i/4, dual lines at
(i + 1/2)/4; for n = 32: 4, 8, 12, 16, 20, 24, 28), at every other face along each line (stride 2, even along-line index),
so 14 lines x 16 faces = 224 data.  The flux through a face between cell a (lower/left) and cell b is the
finite-volume flux of brho.fv / brho.msm_dd, in the +x (vertical line) or +y (horizontal line) direction,
    q = T_ab (p_a - p_b),   T_ab = 2 K_a K_b / (K_a + K_b).
Noise is added as for the pressure data: sigma_q = noise_rel ||q_true|| / sqrt(n_q) (own random stream, so the
pressure data of MS.Data are unchanged).  The likelihood term has the same global-norm form as the pressure term,
    Phi_q = w_q / 2 ||q_obs - q(K)||^2,  w_q = 1 / (sigma_rel_q^2 ||q_obs||^2),
with a likelihood level sigma_rel_q that may be larger than noise_rel (the spline interface pressure of the blocks
is only accurate to ~7% in flux, see the spec).

  * Global sampler: FluxGlobalData wraps MS.Data; potential(theta) = Phi_p + Phi_q from one full forward solve.
  * Blocks: Block.flux (a BlockFlux) adds Phi_q,i for the measured faces on the block's own interface sides, with the
    ONE-SIDED flux of the block (2 K_cell (p_face - p_cell), half-cell link to the Dirichlet face value).  A face on
    the interface of two blocks is therefore counted once by each neighbour.
"""
import numpy as np

SIDES = ("L", "R", "B", "T")


def interface_lines(n):
    """Face indices b (face between cell b-1 and b) of the primal and dual interface lines, sorted."""
    primal = {k * n // 4 for k in (1, 2, 3)}
    dual = {(2 * i + 1) * n // 8 for i in range(4)}
    return sorted(primal | dual)


def face_flux(K2, p2, ia, ja, ib, jb):
    Ka, Kb = K2[ia, ja], K2[ib, jb]
    return 2.0 * Ka * Kb / (Ka + Kb) * (p2[ia, ja] - p2[ib, jb])


class FluxData:
    def __init__(self, S, data, noise_rel=0.003, sigma_rel=0.08, stride=2, offset=0):
        self.S, self.n = S, S.n
        self.noise_rel, self.sigma_rel, self.stride, self.offset = noise_rel, sigma_rel, stride, offset
        n = self.n
        along = np.arange(offset, n, stride)
        ia, ja, ib, jb, orient, line, pos = [], [], [], [], [], [], []
        for o in ("v", "h"):
            for b in interface_lines(n):
                for t in along:
                    if o == "v":
                        ia.append(b - 1); ja.append(t); ib.append(b); jb.append(t)
                    else:
                        ia.append(t); ja.append(b - 1); ib.append(t); jb.append(b)
                    orient.append(o); line.append(b); pos.append(t)
        self.ia, self.ja, self.ib, self.jb = (np.array(x) for x in (ia, ja, ib, jb))
        self.orient, self.line, self.pos = np.array(orient), np.array(line), np.array(pos)
        self.m = len(self.ia)
        self.index = {(o, b, t): i for i, (o, b, t) in enumerate(zip(orient, line, pos))}
        K2 = np.exp(S.kle.field(data.theta_true)).reshape(n, n)
        p2 = S.phys.solve(K2.ravel()).reshape(n, n)
        self.q_true = face_flux(K2, p2, self.ia, self.ja, self.ib, self.jb)
        self.sigma_q = noise_rel * np.linalg.norm(self.q_true) / np.sqrt(self.m)
        rng = np.random.default_rng(np.random.SeedSequence([int(data.seed), 99]))
        self.q = self.q_true + self.sigma_q * rng.standard_normal(self.m)
        self.qnorm = float(np.linalg.norm(self.q))
        self.w = 1.0 / (sigma_rel ** 2 * self.qnorm ** 2)
        self.relerr_truth = float(np.linalg.norm(self.q - self.q_true) / self.qnorm)
        self.phi_truth = 0.5 * self.w * float(np.sum((self.q - self.q_true) ** 2))

    def summary(self):
        return dict(n_flux=self.m, lines=interface_lines(self.n), stride=self.stride, offset=self.offset,
                    noise_rel=self.noise_rel, sigma_rel=self.sigma_rel, sigma_q=float(self.sigma_q),
                    flux_norm=self.qnorm, relerr_at_truth=self.relerr_truth, phi_at_truth=self.phi_truth)


class FluxGlobalData:
    """MS.Data plus the flux likelihood term; same interface as MS.Data (everything else is delegated)."""

    def __init__(self, data, flux):
        self._d, self.flux = data, flux

    def __getattr__(self, name):                       # d, dnorm, w, theta_true, S, n_obs, ...
        return getattr(self._d, name)

    def potential(self, theta):
        S, f, n = self._d.S, self.flux, self._d.S.n
        K = np.exp(S.kle.field(theta))
        p = S.phys.solve(K)
        r2 = float(np.sum((self._d.d - p[S.phys.obs_mask]) ** 2))
        q = face_flux(K.reshape(n, n), p.reshape(n, n), f.ia, f.ja, f.ib, f.jb)
        phi_q = 0.5 * f.w * float(np.sum((f.q - q) ** 2))
        return 0.5 * self._d.w * r2 + phi_q, np.sqrt(r2) / self._d.dnorm

    def summary(self):
        s = self._d.summary()
        s["flux"] = self.flux.summary()
        return s


class BlockFlux:
    """Flux term of one block: for each interface side, the measured faces (local index along the side) and data."""

    def __init__(self, fd, blk):
        self.fd, self.items = fd, []
        n = fd.n
        for side, (xy, cells) in blk.faces.items():
            ia, ja, ib, jb = cells
            vertical = side in ("L", "R")
            b = int(ib[0]) if vertical else int(jb[0])
            along = ja if vertical else ia
            sel, data_i = [], []
            for loc, t in enumerate(along):
                i = fd.index.get(("v" if vertical else "h", b, int(t)))
                if i is not None:
                    sel.append(loc); data_i.append(i)
            if sel:
                self.items.append((side, np.array(sel), np.array(data_i)))
        self.n_faces = sum(len(s) for _, s, _ in self.items)

    def phi(self, blk, p, K_loc):
        """Phi_q,i of the block solution p (flat, block cell order) with permeability K_loc (flat)."""
        P, Kl = p.reshape(blk.nxl, blk.nyl), K_loc.reshape(blk.nxl, blk.nyl)
        tot = 0.0
        for side, sel, di in self.items:
            pf = blk.bc[side]
            if side == "L":
                q = 2.0 * Kl[0, :] * (pf - P[0, :])
            elif side == "R":
                q = 2.0 * Kl[-1, :] * (P[-1, :] - pf)
            elif side == "B":
                q = 2.0 * Kl[:, 0] * (pf - P[:, 0])
            else:
                q = 2.0 * Kl[:, -1] * (P[:, -1] - pf)
            tot += float(np.sum((self.fd.q[di] - q[sel]) ** 2))
        return 0.5 * self.fd.w * tot
