"""
Karhunen-Loeve expansion of the unit-variance covariance C1, and exact
full-rank Gaussian draws.

Everything downstream uses one object, the synthesis matrix T (N x r) on the
simulation cells:

    G(theta) = T theta,   theta ~ N(0, I_r),   T[:, k] = sqrt(lam_k) phi_k(x_cells)

Builders (ProblemConfig.kle_method)
-----------------------------------
"dense"        discrete KLE: eigenvectors of the N x N matrix C1 on the cell
               centers (the July method; eigenvalues sum to N).
"kron"         the same discrete KLE built separably, C1 = C_x (x) C_y [(x) C_z].
"nystrom"      weighted Nystrom solution of the Fredholm problem
                   int C(x, y) phi(y) dy = lam phi(x)
               on a tensor quadrature (Gauss-Legendre by default, nq nodes per
               direction): A = W^1/2 K W^1/2, phi(x_j) = u_j / sqrt(w_j),
               extended to the cells by the Nystrom formula
                   phi_k(x) = (1/lam_k) sum_j C(x, x_j) w_j phi_k(x_j).
               Port of KLE_matlab (solve.m / kle3d.m).  Any kernel.
"nystrom_sep"  the same weighted problem for a separable kernel, solved
               exactly as a tensor product of 1D weighted problems
               (A = A_x (x) A_y [(x) A_z]); eigenvalues and eigenfunctions are
               products of 1D ones.  Identical to "nystrom" on the same
               quadrature, at a tiny fraction of the cost; deterministic
               (1D spectra have no ties; eigenvector signs are fixed).
"auto"         dense for N <= 4096 (keeps the July runs exact), else
               nystrom_sep for separable kernels, nystrom otherwise.
The midpoint rule at the cell centers makes "nystrom" coincide with
"dense" (tests/test_kle.py), so the discrete KLE is the special case of the
Nystrom method with Riemann quadrature on the simulation grid.

Truncation: smallest r with sum_{k<=r} lam_k >= energy * total, total = the
trace of the (weighted) operator.  For the Nystrom builders, modes tied with
the last kept one (lam_k = lam_r, e.g. (i, j) and (j, i) on a square) are kept
as well, so the truncated covariance keeps the kernel's symmetry.

A built KLE is saved with its run (save/load) and never recomputed there:
eigh's basis inside degenerate eigenspaces depends on rounding.
"""
import json
import numpy as np
import scipy.linalg as sla

from . import kernels as KN
from .quadrature import tensor_rule, rule_1d

DENSE_MAX = 4096
JITTER = 1e-10
TIE_RTOL = 1e-9


# ====================================================================== KLE
class KLE:
    def __init__(self, T, lam, lam_all, total, method, meta=None):
        self.T = np.ascontiguousarray(T, float)
        self.N, self.r = self.T.shape
        self.lam = np.asarray(lam, float)
        self.lam_all = np.asarray(lam_all, float)       # computed spectrum
        self.total = float(total)                       # trace of the operator
        self.energy = float(self.lam.sum() / self.total)
        self.method = method
        self.meta = dict(meta or {})
        self.mean_vec = self.T.mean(0)                  # spatial mean = mean_vec @ theta
        self._pinv = None

    # --------------------------------------------------------------- maps
    def field(self, theta):
        """theta (r,) or (k, r) -> G (N,) or (k, N)."""
        return np.asarray(theta) @ self.T.T

    def spatial_mean(self, theta):
        return np.asarray(theta) @ self.mean_vec

    def project(self, f):
        """Least-squares coordinates of a field f (N,) in the basis T (valid
        whether or not the columns are orthogonal on the cells)."""
        if self._pinv is None:
            self._pinv = np.linalg.pinv(self.T)
        return self._pinv @ f

    def sample_theta(self, rng, size=None):
        shape = (self.r,) if size is None else (size, self.r)
        return rng.standard_normal(shape)

    def pointwise_variance(self):
        return (self.T ** 2).sum(1)

    # ------------------------------------------------------------- freeze
    def save(self, path):
        np.savez_compressed(path, T=self.T, lam=self.lam, lam_all=self.lam_all,
                            total=self.total, method=self.method,
                            meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        if "T" in z.files:
            return cls(z["T"], z["lam"], z["lam_all"], float(z["total"]),
                       str(z["method"]), json.loads(str(z["meta"])))
        # legacy (first frozen runs): orthonormal PHI + discrete spectrum
        lam_all, r = z["lam_all"], int(z["r"])
        lam = lam_all[:r]
        return cls(z["PHI"] * np.sqrt(lam), lam, lam_all, lam_all.sum(),
                   str(z["method"]), dict(ell=float(z["ell"]), legacy=True))


# ================================================================ builders
def _truncate(lam_sorted, total, energy, ties):
    c = np.cumsum(lam_sorted) / total
    r = int(np.searchsorted(c, energy) + 1)
    r = min(r, len(lam_sorted))
    if ties:
        while r < len(lam_sorted) and \
                abs(lam_sorted[r] - lam_sorted[r - 1]) <= TIE_RTOL * lam_sorted[r - 1]:
            r += 1
    return r


def _fix_signs(U):
    """Deterministic eigenvector signs: largest-|entry| component positive."""
    idx = np.argmax(np.abs(U), 0)
    s = np.sign(U[idx, np.arange(U.shape[1])])
    s[s == 0] = 1
    return U * s


def _fix_signs_projection(U, x):
    """Rounding-independent signs for 1D eigenvectors of a symmetric kernel on [0, 1]: the projection
    onto the coordinate x is positive.  (Odd modes have two equal extremes of opposite sign, so the
    'largest entry' rule of _fix_signs is decided by rounding and can differ between library versions.)
    Raises if a projection is too small to be trusted."""
    proj = x @ U
    scale = np.linalg.norm(x) * np.linalg.norm(U, axis=0)
    bad = np.abs(proj) < 1e-6 * scale
    s = np.where(proj < 0, -1.0, 1.0)
    return U * s, bad


def _discrete_dense(points, kname, params, energy):
    C = KN.get(kname)["cov"](points, points, params)
    lam, phi = np.linalg.eigh(C)
    lam, phi = np.clip(lam[::-1], 0, None), phi[:, ::-1]
    r = _truncate(lam, lam.sum(), energy, ties=False)       # July rule
    return KLE(phi[:, :r] * np.sqrt(lam[:r]), lam[:r], lam, lam.sum(), "dense")


def _tensor_select(spectra, energy, ties):
    lam_grid = spectra[0]
    for l1 in spectra[1:]:
        lam_grid = np.multiply.outer(lam_grid, l1)
    flat = lam_grid.ravel()
    order = np.argsort(-flat, kind="stable")
    lam_all = flat[order]
    total = float(np.prod([s.sum() for s in spectra]))
    r = _truncate(lam_all, total, energy, ties)
    idx = np.unravel_index(order[:r], lam_grid.shape)
    return lam_all, total, r, idx


def _tensor_columns(vecs, idx, r):
    """Column k = outer product over axes of vecs[a][:, idx[a][k]], raveled
    in 'ij' (C) order to match the cell index m = (ix*ny + iy)*nz + iz."""
    cols = vecs[0][:, idx[0]]
    for a in range(1, len(vecs)):
        p1 = vecs[a][:, idx[a]]
        cols = (cols[:, None, :] * p1[None, :, :]).reshape(-1, r)
    return cols


def _discrete_kron(axes, kname, params, energy):
    kern, d = KN.get(kname), len(axes)
    spectra, vecs = [], []
    for a, x in enumerate(axes):
        l1, p1 = np.linalg.eigh(kern["factor"](a, x, x, params, d))
        spectra.append(np.clip(l1[::-1], 0, None))
        vecs.append(p1[:, ::-1])
    lam_all, total, r, idx = _tensor_select(spectra, energy, ties=False)
    PHI = _tensor_columns(vecs, idx, r)
    return KLE(PHI * np.sqrt(lam_all[:r]), lam_all[:r], lam_all, total, "kron")


def _weighted_eig(A, total, energy, ties):
    """Top eigenpairs of the SPD weighted matrix A, enough for `energy`
    (full eigh for small A, growing subset otherwise)."""
    M = A.shape[0]
    if M <= 4000:
        lam, U = sla.eigh(A)
        lam, U = lam[::-1], U[:, ::-1]
    else:
        k = min(M, 512)
        while True:
            lam, U = sla.eigh(A, subset_by_index=[M - k, M - 1])
            lam, U = lam[::-1], U[:, ::-1]
            if k == M or np.cumsum(lam)[-2] / total >= energy:
                break
            k = min(M, 2 * k)
    lam = np.clip(lam, 0, None)
    r = _truncate(lam, total, energy, ties)
    if r >= len(lam) and len(lam) < M:
        raise RuntimeError("energy not reached by the computed spectrum")
    return lam, U, r


def _nystrom_full(points, kname, params, energy, nq, quad):
    cov = KN.get(kname)["cov"]
    d = points.shape[1]
    Xq, wq = tensor_rule(quad, nq, d)
    sw = np.sqrt(wq)
    K = cov(Xq, Xq, params)
    A = sw[:, None] * K * sw[None, :]
    A = 0.5 * (A + A.T)
    total = float(np.trace(A))                     # = sum_j w_j C(x_j, x_j)
    lam, U, r = _weighted_eig(A, total, energy, ties=True)
    lam, U = lam[:r], _fix_signs(U[:, :r])
    # Nystrom extension:  T = K(P, Xq) (sqrt(w) * U) / sqrt(lam), in chunks
    B = (sw[:, None] * U) / np.sqrt(lam)[None, :]
    T = np.empty((len(points), r))
    step = max(1, int(2e7 // max(1, len(Xq))))
    for s in range(0, len(points), step):
        T[s:s + step] = cov(points[s:s + step], Xq, params) @ B
    lam_all = np.linalg.eigvalsh(A)[::-1] if len(A) <= 4000 else lam
    return KLE(T, lam, np.clip(lam_all, 0, None), total, "nystrom",
               dict(nq=nq, quadrature=quad, n_quad=len(Xq)))


def _nystrom_separable(axes, kname, params, energy, nq, quad, sign_rule="argmax"):
    kern, d = KN.get(kname), len(axes)
    spectra, ext = [], []
    for a, x_out in enumerate(axes):
        xq, wq = rule_1d(quad, nq)
        sw = np.sqrt(wq)
        Aa = sw[:, None] * kern["factor"](a, xq, xq, params, d) * sw[None, :]
        la, Ua = np.linalg.eigh(0.5 * (Aa + Aa.T))
        la, Ua = np.clip(la[::-1], 0, None), Ua[:, ::-1]
        if sign_rule == "argmax":
            Ua = _fix_signs(Ua)
        elif sign_rule == "projection":
            Ua, bad = _fix_signs_projection(Ua, xq)
            sel = la > 1e-6 * la[0]         # modes that can be selected
            if np.any(bad & sel):
                raise ValueError("projection sign rule is ambiguous for mode(s) %s"
                                 % np.where(bad & sel)[0].tolist())
        else:
            raise ValueError(sign_rule)
        spectra.append(la)
        # 1D Nystrom extension of every mode (tiny modes are never selected)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = kern["factor"](a, x_out, xq, params, d) @ (sw[:, None] * Ua) / la[None, :]
        ext.append(np.nan_to_num(phi))
    lam_all, total, r, idx = _tensor_select(spectra, energy, ties=True)
    PHI = _tensor_columns(ext, idx, r)
    meta = dict(nq=nq, quadrature=quad,
                nq_check=_nq_check(kern, d, params, energy, nq, quad, lam_all[:r]))
    return KLE(PHI * np.sqrt(lam_all[:r]), lam_all[:r], lam_all, total,
               "nystrom_sep", meta)


NQ_WARN = 1e-4


def _nq_check(kern, d, params, energy, nq, quad, lam_r):
    """Quadrature convergence: relative change of the kept eigenvalues when
    the rule is refined to 1.5 nq (1D problems only: essentially free).
    Rule of thumb from tests: nq ~ 3 / ell gives ~1e-10 for the sq-exp kernel."""
    nq2 = int(np.ceil(1.5 * nq))
    spectra = []
    for a in range(d):
        xq, wq = rule_1d(quad, nq2)
        sw = np.sqrt(wq)
        Aa = sw[:, None] * kern["factor"](a, xq, xq, params, d) * sw[None, :]
        spectra.append(np.clip(np.linalg.eigvalsh(0.5 * (Aa + Aa.T))[::-1], 0, None))
    lam2, _, _, _ = _tensor_select(spectra, energy, ties=True)
    k = len(lam_r)
    rel = float(np.max(np.abs(lam2[:k] / lam_r - 1)))
    if rel > NQ_WARN:
        print(f"WARNING: KLE quadrature not converged: eigenvalues move by "
              f"{rel:.1e} from nq={nq} to nq={nq2}; increase --kle-nq "
              f"(~3/ell is ample for sq-exp)")
    return dict(nq_refined=nq2, max_rel_eig_change=rel, converged=rel <= NQ_WARN)


def resolve_method(method, N, kname):
    if method != "auto":
        return method
    if N <= DENSE_MAX:
        return "dense"
    return "nystrom_sep" if KN.is_separable(kname) else "nystrom"


def build_kle(points, axes, kname, params, energy=0.95, method="auto",
              nq=40, quadrature="gauss_legendre", sign_rule="argmax"):
    """sign_rule (nystrom_sep only): "argmax" (legacy, default) or "projection" (rounding-independent).
    points: (N, d) cell centers in cell order; axes: per-axis 1D cell
    coordinates of the same tensor grid."""
    points = np.asarray(points, float)
    method = resolve_method(method, len(points), kname)
    if method == "dense":
        kle = _discrete_dense(points, kname, params, energy)
    elif method == "kron":
        kle = _discrete_kron(axes, kname, params, energy)
    elif method == "nystrom":
        kle = _nystrom_full(points, kname, params, energy, nq, quadrature)
    elif method == "nystrom_sep":
        if not KN.is_separable(kname):
            raise ValueError(f"kernel {kname} is not separable")
        kle = _nystrom_separable(axes, kname, params, energy, nq, quadrature, sign_rule)
    else:
        raise ValueError(method)
    kle.meta.update(kernel=kname, params={k: np.asarray(v).tolist()
                                          for k, v in params.items()},
                    energy_target=energy)
    return kle


# ===================================================== full-rank draws
class FullRankGaussian:
    """Exact draws of N(0, C1) on the cells by Cholesky (for the synthetic
    truth and the full-rank curve sweep).  Dense Cholesky of C1 + jitter I for
    N <= 4096 (the July construction, same random-number use); for larger
    separable grids the Kronecker factor L_x (x) L_y [(x) L_z]."""

    def __init__(self, points, axes, kname, params, method="auto"):
        self.points, self.axes = np.asarray(points, float), axes
        self.N = len(self.points)
        self.kname, self.params = kname, params
        if method == "auto":
            method = "dense" if self.N <= DENSE_MAX else "kron"
        if method == "kron" and not KN.is_separable(kname):
            raise ValueError("kron draws need a separable kernel")
        self.method = method
        self._L = None

    def cholesky(self):
        if self._L is None:
            k = KN.get(self.kname)
            if self.method == "dense":
                C = k["cov"](self.points, self.points, self.params)
                self._L = np.linalg.cholesky(C + JITTER * np.eye(self.N))
            else:
                d = len(self.axes)
                self._L = [np.linalg.cholesky(k["factor"](a, x, x, self.params, d)
                                              + JITTER * np.eye(len(x)))
                           for a, x in enumerate(self.axes)]
        return self._L

    def sample(self, rng, size=None):
        """One draw of N standard normals per realization, in cell order."""
        L = self.cholesky()
        k = 1 if size is None else size
        Z = rng.standard_normal(self.N)[:, None] if size is None else \
            rng.standard_normal((self.N, k))
        if self.method == "dense":
            G = (L @ Z).T
        else:
            T = Z.T.reshape([k] + [len(a) for a in self.axes])
            for ax, La in enumerate(L):
                T = np.moveaxis(np.tensordot(La, T, axes=([1], [ax + 1])), 0, ax + 1)
            G = T.reshape(k, self.N)
        return G[0] if size is None else G
