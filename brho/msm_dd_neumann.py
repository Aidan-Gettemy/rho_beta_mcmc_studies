"""Neumann + interface-pressure block (prototype).

The flux h across each interface face is the boundary condition of the block (outflow = -h; Neumann), and the interface pressure
g (the spline) is DATA: the block's predicted face pressure p_f = p_a + h_f / (2 K_a) is compared with g_f, noise sigma_g.
Likelihood of block k:   sensors  d = p_s + e (weight w),   faces  g = p_f + e_g (weight 1/sigma_g^2),   [interior flux rows as in GibbsBlock].
Everything is linear in h, so h | c keeps an exact Gaussian conditional (prior N(h_hat, Sigma0) as in GibbsBlock).

Blocks without a Dirichlet side (no fixed pressure) are singular up to an additive constant b:  p = p* + b 1  with the gauge
1^T p* = 0.  The constant is integrated out (flat prior) of the Gaussian likelihood in closed form, and the fluxes must satisfy the
compatibility condition sum(h) = 0, which the exact draw of h enforces by conditioning (not by an ad-hoc projection).
"""
import numpy as np
from .msm_dd_gibbs import GibbsBlock


class NeumannBlock(GibbsBlock):
    def __init__(self, blk, eps=0.03, corr_len=4.0, fd=None, interior=False, eps_meas=None, g_rel=0.0045, g_sigma=None):
        super().__init__(blk, eps, corr_len, fd=fd, interior=interior, eps_meas=eps_meas)
        self.dir_entries = [(ids, g) for ids, g, ainv, sl in self.entries if sl is None and not np.all(ainv > 0)]
        self.dir_entries = [e for e in self.dir_entries if len(e[0])]
        self.singular = len(self.dir_entries) == 0
        # interface pressure datum per h entry (same order as h_cells)
        gf = []
        for ids, g, ainv, sl in self.entries:
            if sl is not None:
                gf += list(g)
        self.g_face = np.array(gf)
        rms_g = float(np.sqrt(np.mean(self.g_face ** 2))) if self.nh else 1.0
        self.sigma_g = float(g_sigma) if g_sigma is not None else g_rel * rms_g
        self.wg = 1.0 / self.sigma_g ** 2 if self.sigma_g > 0 else 0.0
        self.rows = np.concatenate([self.b.obs_loc, self.h_cells]).astype(int)
        self.ns = len(self.b.obs_loc)
        self.z = np.concatenate([self.b.d_loc, self.g_face])
        self.wv = np.concatenate([np.full(self.ns, self.b.w), np.full(self.nh, self.wg)])
        self.one = np.ones(self.nh)
        if self.singular and self.nh:                                   # initial h: prior conditioned on sum(h) = 0
            u = self.Sigma0 @ self.one
            self.h_hat0 = self.h_hat0 - u * self.h_hat0.sum() / (self.one @ u)

    # -- forward model ---------------------------------------------------------------------------------------------
    def system(self, K):
        sv = self.b.solver
        K = np.asarray(K, float).ravel()
        N = self.N
        ax, bx, ay, by = sv.ax, sv.bx, sv.ay, sv.by
        Tx = 2.0 * K[ax] * K[bx] / (K[ax] + K[bx]); Ty = 2.0 * K[ay] * K[by] / (K[ay] + K[by])
        A = np.zeros((N, N))
        A[ax, bx] = -Tx; A[bx, ax] = -Tx; A[ay, by] = -Ty; A[by, ay] = -Ty
        diag = (np.bincount(ax, Tx, N) + np.bincount(bx, Tx, N) + np.bincount(ay, Ty, N) + np.bincount(by, Ty, N))
        Gs = []
        for ids, g in self.dir_entries:
            t = 2.0 * K[ids]
            diag[ids] += t
            Gs.append(t)
        A.flat[sv.diag_pos] = diag
        if self.singular:
            A = A + (diag.mean() / N) * np.ones((N, N))               # gauge 1^T p = 0 (A 1 = 0, rhs sums to 0)
        return np.linalg.inv(A), (Gs, K)

    def rhs(self, Gs, h):
        G, _ = Gs
        b = np.zeros(self.N)
        for (ids, g), t in zip(self.dir_entries, G):
            b[ids] += t * g
        np.add.at(b, self.h_cells, h)                                   # cell mass balance: A p = h on interface cells
        return b

    def _resid(self, p, h, K):
        y = np.concatenate([p[self.b.obs_loc], p[self.h_cells] + h / (2.0 * K[self.h_cells])])
        return self.z - y

    def _phi_data(self, r):
        phi = 0.5 * float(np.sum(self.wv * r * r))
        if self.singular:                                               # flat-prior integral over the constant
            phi -= 0.5 * float(np.sum(self.wv * r)) ** 2 / float(np.sum(self.wv))
        return phi

    def _phi_flux(self, p, K):
        if self.fa is None:
            return 0.0
        return 0.5 * self.wq * float(np.sum((self.fq - self._T(K) * (p[self.fa] - p[self.fb])) ** 2))

    def phi_from(self, c, p, h):
        K = np.exp(self.b.log_field(c))
        return self._phi_data(self._resid(p, h, K)) + self._phi_flux(p, K)

    def potential(self, c, h):
        K = np.exp(self.b.log_field(c))
        Ainv, Gs = self.system(K)
        p = Ainv @ self.rhs(Gs, h)
        return self._phi_data(self._resid(p, h, K)) + self._phi_flux(p, K), Ainv, Gs, p

    def sensor_r2(self, p):                                             # diagnostics: sensor misfit at the best constant
        r = self.b.d_loc - p[self.b.obs_loc]
        if self.singular:
            r = r - r.mean()
        return float(np.sum(r * r))

    # -- exact Gibbs draw of h -------------------------------------------------------------------------------------
    def draw_h(self, c, h, Ainv, Gs, p, rng):
        if not self.nh:
            return h
        K = Gs[1]
        S = Ainv[np.ix_(self.rows, self.h_cells)].copy()
        S[self.ns + np.arange(self.nh), np.arange(self.nh)] += 1.0 / (2.0 * K[self.h_cells])
        r0 = self._resid(p, h, K)
        W = np.diag(self.wv)
        if self.singular:
            W = W - np.outer(self.wv, self.wv) / self.wv.sum()
        P = self.Sigma0_inv + S.T @ W @ S
        rhsv = self.Sigma0_inv @ self.h_hat + S.T @ W @ (r0 + S @ h)
        if self.fa is not None:
            T = self._T(K)
            Sq = T[:, None] * (Ainv[np.ix_(self.fa, self.h_cells)] - Ainv[np.ix_(self.fb, self.h_cells)])
            rq = self.fq - T * (p[self.fa] - p[self.fb]) + Sq @ h
            P = P + self.wq * Sq.T @ Sq
            rhsv = rhsv + self.wq * Sq.T @ rq
        L = np.linalg.cholesky(P)
        x = np.linalg.solve(P, rhsv) + np.linalg.solve(L.T, rng.standard_normal(self.nh))
        if self.singular:                                               # condition the draw on sum(h) = 0 (exact)
            u = np.linalg.solve(P, self.one)
            x = x - u * x.sum() / u.sum()
        return x
