"""Plain Gibbs for the Robin interface flux h (prototype).

State of a block chain: local coefficients c and the vector h of Robin flux data on the block's interface faces.
For fixed c the block pressure is linear in h, so with the prior h ~ N(h_hat, Sigma0) and the Gaussian sensor likelihood
(weight w) the conditional of h is Gaussian in closed form:
    P = Sigma0^-1 + w S^T S,  mean = P^-1 [Sigma0^-1 h_hat + w S^T (d - p0 + S h0)],
S = d p_sensors / d h at the current c.  One iteration = a pCN step for c given h, then an exact draw of h given c.
Each block has its own copy of h (a face shared by two blocks is drawn separately for each).
"""
import numpy as np
from .samplers import StepAdapter


class GibbsBlock:
    """Gibbs block.  Optional extras (need the FluxData `fd`):
      interior=True   the measured flux faces that lie INSIDE the block (both cells in the block) enter as likelihood terms
                      Phi_q = w_q/2 sum (q_obs - T_ab (p_a - p_b))^2 with w_q = fd.w; they are linear in p, hence in h.
      eps_meas        prior sd factor of h on the faces that carry a flux measurement (default: eps for all faces);
                      eps applies to the interpolated faces.
    """

    def __init__(self, blk, eps=0.03, corr_len=4.0, fd=None, interior=False, eps_meas=None):
        self.b = blk
        sv = blk.solver
        self.N, self.nxl, self.nyl = sv.N, sv.nxl, sv.nyl
        idx = np.arange(self.N).reshape(self.nxl, self.nyl)
        side_cells = dict(L=idx[0, :], R=idx[-1, :], B=idx[:, 0], T=idx[:, -1])
        self.entries = []                       # (cell ids, g, ainv, h index slice or None)
        self.h_cells, self.h_ainv, self.h_hat, self.side_of, self.pos = [], [], [], [], []
        self.h_meas = []
        off = 0
        for side, v in blk.bc.items():
            if v is None:
                continue
            ids = side_cells[side]
            if isinstance(v, dict):
                g, h, ainv = (np.asarray(v[k], float) for k in ("g", "h", "ainv"))
            else:
                g, h, ainv = np.asarray(v, float), np.zeros(len(v)), np.zeros(len(v))
            sl = None
            if np.any(ainv > 0):
                sl = slice(off, off + len(ids)); off += len(ids)
                self.h_cells += list(ids); self.h_ainv += list(ainv); self.h_hat += list(h)
                self.side_of += [side] * len(ids); self.pos += list(range(len(ids)))
                if fd is not None:
                    _, (fia, fja, fib, fjb) = blk.faces[side]
                    vert = side in ("L", "R")
                    line = int(fib[0]) if vert else int(fjb[0])
                    along = fja if vert else fia
                    self.h_meas += [("v" if vert else "h", line, int(t)) in fd.index for t in along]
                else:
                    self.h_meas += [False] * len(ids)
            self.entries.append((ids, g, ainv, sl))
        self.nh = off
        self.h_cells, self.h_ainv, self.h_hat = (np.array(x) for x in (self.h_cells, self.h_ainv, self.h_hat))
        self.h_hat0 = self.h_hat.copy()
        if self.nh:
            rms = np.sqrt(np.mean(self.h_hat ** 2))
            Sg = np.zeros((self.nh, self.nh))
            for i in range(self.nh):
                for j in range(self.nh):
                    if self.side_of[i] == self.side_of[j]:
                        d = abs(self.pos[i] - self.pos[j])
                        Sg[i, j] = np.exp(-d / corr_len) if corr_len > 0 else float(i == j)
            sd = np.where(np.array(self.h_meas), eps if eps_meas is None else eps_meas, eps) * rms
            self.Sigma0 = sd[:, None] * (Sg + 1e-8 * np.eye(self.nh)) * sd[None, :]
            self.Sigma0_inv = np.linalg.inv(self.Sigma0)
        self.A_nodiag = None
        # interior flux faces (local flat cell indices a, b and data)
        self.fa = self.fb = self.fq = None
        self.wq = 0.0
        if interior and fd is not None:
            ok = ((fd.ia >= blk.ix0) & (fd.ia <= blk.ix1) & (fd.ib >= blk.ix0) & (fd.ib <= blk.ix1) &
                  (fd.ja >= blk.iy0) & (fd.ja <= blk.iy1) & (fd.jb >= blk.iy0) & (fd.jb <= blk.iy1))
            sel = np.flatnonzero(ok)
            self.fa = (fd.ia[sel] - blk.ix0) * self.nyl + (fd.ja[sel] - blk.iy0)
            self.fb = (fd.ib[sel] - blk.ix0) * self.nyl + (fd.jb[sel] - blk.iy0)
            self.fq = np.asarray(fd.q[sel], float)
            self.wq = float(fd.w)

    def n_interior(self):
        return 0 if self.fa is None else len(self.fa)

    def _T(self, K):
        Ka, Kb = K[self.fa], K[self.fb]
        return 2.0 * Ka * Kb / (Ka + Kb)

    def sensor_r2(self, p):
        return float(np.sum((self.b.d_loc - p[self.b.obs_loc]) ** 2))

    def system(self, K):
        """(A^-1, G, base rhs pieces) for permeability K (flattened block)."""
        sv = self.b.solver
        K = np.asarray(K, float).ravel()
        N = self.N
        ax, bx, ay, by = sv.ax, sv.bx, sv.ay, sv.by
        Tx = 2.0 * K[ax] * K[bx] / (K[ax] + K[bx]); Ty = 2.0 * K[ay] * K[by] / (K[ay] + K[by])
        A = np.zeros((N, N))
        A[ax, bx] = -Tx; A[bx, ax] = -Tx; A[ay, by] = -Ty; A[by, ay] = -Ty
        diag = (np.bincount(ax, Tx, N) + np.bincount(bx, Tx, N) + np.bincount(ay, Ty, N) + np.bincount(by, Ty, N))
        Gs = []
        for ids, g, ainv, sl in self.entries:
            t = 2.0 * K[ids]
            G = t / (1.0 + t * ainv)
            diag[ids] += G
            Gs.append(G)
        A.flat[sv.diag_pos] = diag
        return np.linalg.inv(A), Gs

    def rhs(self, Gs, h):
        b = np.zeros(self.N)
        for (ids, g, ainv, sl), G in zip(self.entries, Gs):
            hv = h[sl] if sl is not None else 0.0
            b[ids] += G * (g + ainv * hv)
        return b

    def potential(self, c, h):
        Ainv, Gs = self.system(np.exp(self.b.log_field(c)))
        p = Ainv @ self.rhs(Gs, h)
        phi = 0.5 * self.b.w * self.sensor_r2(p)
        if self.fa is not None:
            K = np.exp(self.b.log_field(c))
            phi += 0.5 * self.wq * float(np.sum((self.fq - self._T(K) * (p[self.fa] - p[self.fb])) ** 2))
        return phi, Ainv, Gs, p

    def phi_from(self, c, p, h):
        phi = 0.5 * self.b.w * self.sensor_r2(p)
        if self.fa is not None:
            K = np.exp(self.b.log_field(c))
            phi += 0.5 * self.wq * float(np.sum((self.fq - self._T(K) * (p[self.fa] - p[self.fb])) ** 2))
        return phi

    def draw_h(self, c, h, Ainv, Gs, p, rng):
        """Exact draw from p(h | c, data)."""
        if not self.nh:
            return h
        obs = self.b.obs_loc
        Gain = np.zeros(self.nh)
        k = 0
        for (ids, g, ainv, sl), G in zip(self.entries, Gs):
            if sl is not None:
                Gain[sl] = G * ainv
        S = Ainv[np.ix_(obs, self.h_cells)] * Gain                 # (n_obs, nh): d p_obs / d h
        r = self.b.d_loc - p[obs] + S @ h
        P = self.Sigma0_inv + self.b.w * S.T @ S
        rhsv = self.Sigma0_inv @ self.h_hat + self.b.w * S.T @ r
        if self.fa is not None:                                    # interior flux rows: q = T (p_a - p_b), linear in h
            T = self._T(np.exp(self.b.log_field(c)))
            Af = Ainv[np.ix_(self.fa, self.h_cells)] - Ainv[np.ix_(self.fb, self.h_cells)]
            Sq = T[:, None] * Af * Gain
            rq = self.fq - T * (p[self.fa] - p[self.fb]) + Sq @ h
            P = P + self.wq * Sq.T @ Sq
            rhsv = rhsv + self.wq * Sq.T @ rq
        L = np.linalg.cholesky(P)
        mean = np.linalg.solve(P, rhsv)
        z = rng.standard_normal(self.nh)
        return mean + np.linalg.solve(L.T, z)


def run_gibbs_block_chain(gb, seed_key, n_iter, n_adapt, s0=0.2, save_every=50, target=0.25, every=100, gibbs=True, init_scale=1.0):
    blk = gb.b
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_key)))
    ad = StepAdapter([s0], targets=[target], every=every)
    c = init_scale * rng.standard_normal(blk.r_k)
    h = gb.h_hat0.copy()
    phi, Ainv, Gs, p = gb.potential(c, h)
    n_save = n_iter // save_every
    out = dict(c=np.empty((n_save, blk.r_k), np.float32), s_hist=np.empty(n_save, np.float32), acc=np.zeros(n_iter, np.uint8),
               relerr=np.empty(n_iter, np.float32), h=np.empty((n_save, gb.nh), np.float32))
    for it in range(n_iter):
        if it % save_every == 0:
            j = it // save_every
            out["c"][j], out["s_hist"][j], out["h"][j] = c, ad.s[0], h
        s = ad.s[0]
        c_new = np.sqrt(1.0 - s * s) * c + s * rng.standard_normal(blk.r_k)
        phi_new, Ainv_n, Gs_n, p_n = gb.potential(c_new, h)
        accepted = np.log(rng.random()) < phi - phi_new
        if accepted:
            c, phi, Ainv, Gs, p = c_new, phi_new, Ainv_n, Gs_n, p_n
        ad.record(0, accepted); ad.tick(it, n_adapt)
        out["acc"][it] = accepted
        if gibbs and gb.nh:
            h = gb.draw_h(c, h, Ainv, Gs, p, rng)
            p = Ainv @ gb.rhs(Gs, h)
            phi = gb.phi_from(c, p, h)
        out["relerr"][it] = np.sqrt(gb.sensor_r2(p)) / blk.dnorm_loc
    return out
