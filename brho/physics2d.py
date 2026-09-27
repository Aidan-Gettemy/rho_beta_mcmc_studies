"""
2D permeameter forward model (cell-centered finite volumes).

    -div(K grad p) = 0 on (0,1)^2
    -K dp/dn = q_in  on the left face  x = 0   (prescribed unit influx)
     p = 0           on the right face x = 1   (outflow)
     no flow         on top and bottom

Harmonic inter-cell transmissibilities; half-cell transmissibility 2 K_cell
at the Dirichlet face; influx enters the RHS as q_in * h per boundary cell.
Cell index m = ix * n + iy; reshape(n, n) gives an [ix, iy] array
(transpose it for imshow with origin="lower").

This is the solver of mcmc_study.py / hier_sampler.py, restructured as a
class with a fixed sparsity pattern (the CSR structure is built once; each
solve only fills in the values).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class Darcy2D:
    dim = 2

    def __init__(self, n=16, q_in=1.0):
        self.n, self.N, self.h, self.q_in = n, n * n, 1.0 / n, q_in
        mid = lambda ix, iy: ix * n + iy
        ix1, iy1 = np.meshgrid(np.arange(n - 1), np.arange(n), indexing="ij")
        self.m1x, self.m2x = mid(ix1, iy1).ravel(), mid(ix1 + 1, iy1).ravel()
        ix2, iy2 = np.meshgrid(np.arange(n), np.arange(n - 1), indexing="ij")
        self.m1y, self.m2y = mid(ix2, iy2).ravel(), mid(ix2, iy2 + 1).ravel()
        self.mL = mid(0, np.arange(n))            # inflow cells
        self.mR = mid(n - 1, np.arange(n))        # outflow cells
        rows = np.concatenate([self.m1x, self.m2x, self.m1x, self.m2x,
                               self.m1y, self.m2y, self.m1y, self.m2y, self.mR])
        cols = np.concatenate([self.m1x, self.m2x, self.m2x, self.m1x,
                               self.m1y, self.m2y, self.m2y, self.m1y, self.mR])
        # fixed CSR pattern: COO entries are summed into unique (row, col)
        key = rows * self.N + cols
        uniq, self._inv = np.unique(key, return_inverse=True)
        self._indices = (uniq % self.N).astype(np.int32)
        self._indptr = np.searchsorted(uniq // self.N,
                                       np.arange(self.N + 1)).astype(np.int32)
        self._nnz = len(uniq)
        self._b = np.zeros(self.N)
        self._b[self.mL] = q_in * self.h

        xc = (np.arange(n) + 0.5) * self.h
        X, Y = np.meshgrid(xc, xc, indexing="ij")
        self.xc = xc
        self.points = np.column_stack([X.ravel(), Y.ravel()])
        ixg, iyg = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        # chessboard "black" cells (ix + iy even): 128 observations for n=16
        self.obs_mask = ((ixg + iyg) % 2 == 0).ravel()

    # ------------------------------------------------------------- solve
    def matrix(self, K):
        k = np.asarray(K, float).ravel()
        Tx = 2 * k[self.m1x] * k[self.m2x] / (k[self.m1x] + k[self.m2x])
        Ty = 2 * k[self.m1y] * k[self.m2y] / (k[self.m1y] + k[self.m2y])
        vals = np.concatenate([Tx, Tx, -Tx, -Tx, Ty, Ty, -Ty, -Ty,
                               2 * k[self.mR]])
        data = np.bincount(self._inv, weights=vals, minlength=self._nnz)
        return sp.csr_matrix((data, self._indices, self._indptr),
                             shape=(self.N, self.N))

    def solve(self, K):
        """Cell pressures p (N,) for permeability K (N,) or (n, n)."""
        return spla.spsolve(self.matrix(K).tocsc(), self._b)

    def keff(self, K, p):
        """Keff = Q L / (A dp), as the laboratory computes it.  Inflow-face
        pressure reconstructed from the influx: p_face = p_cell + q h/2 / K."""
        k = np.asarray(K, float).ravel()
        p_face = p[self.mL] + self.q_in * (self.h / 2) / k[self.mL]
        Q, A, L = self.q_in * 1.0, 1.0, 1.0
        return Q * L / (A * p_face.mean())

    def forward(self, K):
        """One solve -> (p, p_obs, Keff)."""
        p = self.solve(K)
        return p, p[self.obs_mask], self.keff(K, p)

    # ------------------------------------------------------------ checks
    def scaling_check(self, K, c=3.7):
        """Relative errors of p(cK) = p(K)/c and Keff(cK) = c Keff(K)."""
        p1, p2 = self.solve(K), self.solve(c * np.asarray(K))
        e_p = np.abs(p2 - p1 / c).max() / np.abs(p1 / c).max()
        k1, k2 = self.keff(K, p1), self.keff(c * np.asarray(K), p2)
        return dict(p_rel_err=float(e_p), keff_rel_err=float(abs(k2 / (c * k1) - 1)))

    def grid(self, v):
        """(N,) -> (n, n) [ix, iy]."""
        return np.asarray(v).reshape(self.n, self.n)
