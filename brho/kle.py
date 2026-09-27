"""
Global Karhunen-Loeve expansion of the unit-variance squared-exponential
covariance C1 on the cell centers.

    G(theta) = PHI @ (SQ * theta),   theta ~ N(0, I_r)

`r` is the smallest number of modes capturing `energy` of trace(C1)
(r = 56 for n = 16, ell = 0.1, 95%).

Two constructions of the same expansion
---------------------------------------
"dense"  eigendecomposition of the N x N matrix C1.  O(N^3) time, O(N^2)
         memory: fine to a few thousand cells.  Used for the July 16x16
         problem so its runs reproduce exactly.
"kron"   the squared-exponential kernel is separable on a tensor grid,
             exp(-|x-y|^2 / 2l^2) = prod_a exp(-(x_a-y_a)^2 / 2l^2),
         so C1 = C_x (x) C_y [(x) C_z] (cell order m = ix*ny + iy [...]).
         Its eigenpairs are exactly the products of the 1D eigenpairs,
             lambda = lambda_i^x lambda_j^y,  phi = phi_i^x (x) phi_j^y,
         and its Cholesky factor is L_x (x) L_y.  Only 1D matrices are ever
         factorized: 100x100 or 30^3 grids cost milliseconds and no N x N
         storage.  Mathematically identical to "dense"; within a degenerate
         eigenspace (lambda_i lambda_j = lambda_j lambda_i) the basis may
         differ by a rotation, which leaves the law of G unchanged.
"auto"   dense for N <= 4096, kron above.
"""
import numpy as np

DENSE_MAX = 4096
JITTER = 1e-10


def sqexp_cov(points, ell):
    points = np.asarray(points, float)
    if points.ndim == 1:
        points = points[:, None]
    D2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    return np.exp(-D2 / (2.0 * ell ** 2))


class KLE:
    def __init__(self, points, ell, energy=0.95, axes=None, method="auto"):
        self.N = len(points)
        self.ell = ell
        if method == "auto":
            method = "dense" if (self.N <= DENSE_MAX or axes is None) else "kron"
        if method == "kron" and axes is None:
            raise ValueError("kron KLE needs the grid axes")
        self.method = method
        self._chol = None
        if method == "dense":
            self.C = sqexp_cov(points, ell)
            lam, phi = np.linalg.eigh(self.C)
            lam, phi = np.clip(lam[::-1], 0, None), phi[:, ::-1]
            self.lam_all = lam
            self.r = self._truncate(lam, energy)
            self.PHI = phi[:, :self.r]
        elif method == "kron":
            self.C = None
            self._axes = [np.asarray(a, float) for a in axes]
            self._build_kron(energy)
        else:
            raise ValueError(method)
        self.lam = self.lam_all[:self.r]
        self.energy = float(self.lam.sum() / self.lam_all.sum())
        self.SQ = np.sqrt(self.lam)
        # KLE coefficients of the constant function 1 (level direction)
        self.c_one = (self.PHI.T @ np.ones(self.N)) / self.SQ
        # spatial mean of G(theta) is linear in theta: mean_vec @ theta
        self.mean_vec = (self.PHI.T @ np.ones(self.N)) * self.SQ / self.N

    # ------------------------------------------------------------- freeze
    # eigh's basis inside a degenerate eigenspace depends on rounding, i.e.
    # on the BLAS build, CPU and thread count.  A run therefore computes its
    # basis ONCE (at setup), saves it, and every chain and report loads it.
    _FROZEN = ("lam_all", "PHI", "r", "N", "ell", "method")

    def save(self, path):
        np.savez_compressed(path, **{k: np.asarray(getattr(self, k))
                                     for k in self._FROZEN})

    @classmethod
    def load(cls, path, points=None, axes=None):
        """Rebuild a KLE from a saved basis.  `points`/`axes` are only needed
        for full-rank sampling (Cholesky), which the samplers never use."""
        z = np.load(path)
        self = cls.__new__(cls)
        self.lam_all, self.PHI = z["lam_all"], z["PHI"]
        self.r, self.N = int(z["r"]), int(z["N"])
        self.ell, self.method = float(z["ell"]), str(z["method"])
        self._chol = None
        self.C = sqexp_cov(points, self.ell) if (points is not None and
                                                 self.method == "dense") else None
        self._axes = [np.asarray(a, float) for a in axes] if axes is not None else None
        self.lam = self.lam_all[:self.r]
        self.energy = float(self.lam.sum() / self.lam_all.sum())
        self.SQ = np.sqrt(self.lam)
        self.c_one = (self.PHI.T @ np.ones(self.N)) / self.SQ
        self.mean_vec = (self.PHI.T @ np.ones(self.N)) * self.SQ / self.N
        return self

    @staticmethod
    def _truncate(lam, energy):
        return int(np.searchsorted(np.cumsum(lam) / lam.sum(), energy) + 1)

    def _build_kron(self, energy):
        eig1 = []
        for a in self._axes:
            l1, p1 = np.linalg.eigh(sqexp_cov(a, self.ell))
            eig1.append((np.clip(l1[::-1], 0, None), p1[:, ::-1]))
        lam_grid = eig1[0][0]
        for l1, _ in eig1[1:]:
            lam_grid = np.multiply.outer(lam_grid, l1)
        flat = lam_grid.ravel()
        order = np.argsort(-flat, kind="stable")
        self.lam_all = flat[order]
        self.r = self._truncate(self.lam_all, energy)
        idx = np.unravel_index(order[:self.r], lam_grid.shape)
        cols = eig1[0][1][:, idx[0]]                       # (n_x, r)
        for a in range(1, len(eig1)):
            p1 = eig1[a][1][:, idx[a]]                     # (n_a, r)
            cols = (cols[:, None, :] * p1[None, :, :]).reshape(-1, self.r)
        self.PHI = np.ascontiguousarray(cols)

    # --------------------------------------------------------------- maps
    def field(self, theta):
        """theta (r,) or (k, r) -> G (N,) or (k, N)."""
        theta = np.asarray(theta)
        return (theta * self.SQ) @ self.PHI.T

    def project(self, f):
        """L2 projection of a field onto KLE coordinates."""
        return (self.PHI.T @ f) / self.SQ

    def spatial_mean(self, theta):
        return np.asarray(theta) @ self.mean_vec

    # ------------------------------------------------------------ sampling
    def sample_theta(self, rng, size=None):
        shape = (self.r,) if size is None else (size, self.r)
        return rng.standard_normal(shape)

    def cholesky(self):
        """Dense: L with L L^T = C1 + jitter I.  Kron: list of 1D factors."""
        if self._chol is None:
            if self.method == "dense":
                self._chol = np.linalg.cholesky(self.C + JITTER * np.eye(self.N))
            else:
                self._chol = [np.linalg.cholesky(sqexp_cov(a, self.ell)
                                                 + JITTER * np.eye(len(a)))
                              for a in self._axes]
        return self._chol

    def sample_full(self, rng, size=None):
        """Full-rank realization(s) of N(0, C1) by Cholesky (one draw of N
        standard normals per realization, in cell order)."""
        L = self.cholesky()
        k = 1 if size is None else size
        Z = rng.standard_normal((self.N, k)) if size is not None else \
            rng.standard_normal(self.N)[:, None]
        if self.method == "dense":
            G = (L @ Z).T
        else:
            shape = [len(a) for a in self._axes]
            T = Z.T.reshape([k] + shape)
            for ax, La in enumerate(L):                # apply L_a along axis a
                T = np.moveaxis(np.tensordot(La, T, axes=([1], [ax + 1])), 0, ax + 1)
            G = T.reshape(k, self.N)
        return G[0] if size is None else G
