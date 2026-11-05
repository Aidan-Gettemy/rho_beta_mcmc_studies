"""
Two-dimensional Nystrom KLE with a reproducible basis (the construction of the MATLAB code):

  1. one 2D Gauss-Legendre rule (nq points per axis, weights w) on the unit square;
  2. the weighted covariance A = sqrt(w) C(X, X) sqrt(w) and its eigenproblem (full 2D, no
     separability used);
  3. the Nystrom extension of the eigenfunctions to ANY set of points P:
         phi_k(P) = C(P, X) (sqrt(w) u_k) / lambda_k,      T = phi sqrt(lambda).
So the modes are solved once and only extended to the 16x16, 32x32, 64x64 cell centres.

Reproducibility.  An eigensolver returns eigenvectors only up to sign, and inside an (exactly) degenerate
eigenspace only up to rotation; on the square the isotropic kernel has such pairs by the x <-> y symmetry.
Different LAPACK builds therefore give different bases, i.e. different fields for the same theta.  Here
the basis of each eigenspace is fixed by the eigenspace PROJECTOR (basis independent): fixed smooth
reference functions g_1, g_2, ... are projected onto the eigenspace and Gram-Schmidt orthonormalised
(sign: positive overlap with the first projected function).  The result depends only on the quadrature
rule and the kernel, not on the eigensolver.
"""
import numpy as np

from . import kernels as KN
from .kle import tensor_rule

TIE_RTOL = 1e-8


def _reference_functions(X, n):
    """n fixed, generic, smooth functions on the quadrature points (no symmetry of the square)."""
    out = []
    for k in range(n):
        a, b, c = 1.3 + 0.71 * k, 0.9 + 1.13 * k, 0.37 + 0.59 * k
        out.append(np.cos(a * X[:, 0] + b * X[:, 1] + c) + 0.3 * X[:, 0] * (k + 1) - 0.2 * X[:, 1] ** 2)
    return np.stack(out, 1)


def canonical_basis(lam, U, sw, X):
    """Replace the eigenvector basis of every (numerically) degenerate group of columns by the canonical one.
    lam descending; U columns orthonormal.  Returns U_canonical, group labels."""
    n = U.shape[1]
    G = _reference_functions(X, 8) * sw[:, None]            # reference functions in the weighted space
    Uc = np.empty_like(U)
    groups, k = np.empty(n, int), 0
    g = 0
    while k < n:
        m = k + 1
        while m < n and abs(lam[m] - lam[k]) <= TIE_RTOL * lam[k]:
            m += 1
        E = U[:, k:m]
        P = E @ (E.T @ G)                                    # projections of the reference functions
        basis = []
        for j in range(P.shape[1]):
            v = P[:, j].copy()
            for b in basis:
                v -= (b @ v) * b
            nv = np.linalg.norm(v)
            if nv > 1e-6 * np.linalg.norm(G[:, j]):
                basis.append(v / nv)
            if len(basis) == m - k:
                break
        if len(basis) != m - k:
            raise ValueError("reference functions do not span the eigenspace of modes %d..%d" % (k, m - 1))
        Uc[:, k:m] = np.stack(basis, 1)
        groups[k:m] = g
        g += 1
        k = m
    return Uc, groups


class Kle2D:
    def __init__(self, nq=8, ell=0.2, sigma2=1.0, r=20, quad="gauss_legendre"):
        self.nq, self.ell, self.sigma2, self.r = nq, ell, sigma2, r
        self.params = dict(ell=ell, sigma2=sigma2)
        self.X, w = tensor_rule(quad, nq, 2)
        self.sw = np.sqrt(w)
        cov = KN.get("sqexp")["cov"]
        A = self.sw[:, None] * cov(self.X, self.X, self.params) * self.sw[None, :]
        A = 0.5 * (A + A.T)
        self.total = float(np.trace(A))
        lam, U = np.linalg.eigh(A)
        lam, U = lam[::-1], U[:, ::-1]
        if lam[r - 1] - lam[r] <= TIE_RTOL * lam[r - 1]:
            raise ValueError("mode r is tied with mode r + 1: truncation would split a degenerate pair")
        U, self.groups = canonical_basis(lam[:r + 8] if len(lam) > r + 8 else lam, U[:, :r + 8], self.sw, self.X)
        self.lam = lam[:r].copy()
        self.energy = float(self.lam.sum() / self.total)
        self.B = (self.sw[:, None] * U[:, :r]) / np.sqrt(self.lam)[None, :]

    def extend(self, points):
        """T (len(points), r): sqrt(lambda_k) phi_k at the points (Nystrom extension)."""
        cov = KN.get("sqexp")["cov"]
        return cov(np.asarray(points, float), self.X, self.params) @ self.B
