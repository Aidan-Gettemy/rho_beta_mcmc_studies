"""
Global Karhunen-Loeve expansion of the unit-variance squared-exponential
covariance C1 on the cell centers.

    G(theta) = PHI @ (SQ * theta),   theta ~ N(0, I_r)

`r` is the smallest number of modes capturing `energy` of trace(C1)
(r = 56 for n = 16, ell = 0.1, 95%).
"""
import numpy as np


def sqexp_cov(points, ell):
    D2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    return np.exp(-D2 / (2.0 * ell ** 2))


class KLE:
    def __init__(self, points, ell, energy=0.95):
        self.N = len(points)
        self.C = sqexp_cov(points, ell)
        lam, phi = np.linalg.eigh(self.C)
        lam, phi = np.clip(lam[::-1], 0, None), phi[:, ::-1]
        self.lam_all = lam
        self.r = int(np.searchsorted(np.cumsum(lam) / lam.sum(), energy) + 1)
        self.energy = float(lam[:self.r].sum() / lam.sum())
        self.lam = lam[:self.r]
        self.SQ = np.sqrt(self.lam)
        self.PHI = phi[:, :self.r]
        # KLE coefficients of the constant function 1 (level direction)
        self.c_one = (self.PHI.T @ np.ones(self.N)) / self.SQ
        # spatial mean of G(theta) is linear in theta: mean_vec @ theta
        self.mean_vec = (self.PHI.T @ np.ones(self.N)) * self.SQ / self.N
        self._chol = None

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
        if self._chol is None:
            self._chol = np.linalg.cholesky(self.C + 1e-10 * np.eye(self.N))
        return self._chol

    def sample_full(self, rng, size=None):
        """Full-rank realization(s) of N(0, C1) by Cholesky."""
        L = self.cholesky()
        if size is None:
            return L @ rng.standard_normal(self.N)
        return (L @ rng.standard_normal((self.N, size))).T
