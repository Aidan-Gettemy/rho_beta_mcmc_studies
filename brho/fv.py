"""
Permeameter forward model on a uniform structured grid, 2D or 3D
(cell-centered finite volumes, harmonic inter-cell transmissibilities).

    -div(K grad p) = 0 on the unit square / cube
    -K dp/dn = q_in   on the inflow face  x = 0   (prescribed influx)
     p = 0            on the outflow face x = 1
     no flow          on all lateral faces

Discretization (identical to physics2d.Darcy2D in 2D):
    interior face  T = (area / h_axis) * 2 K1 K2 / (K1 + K2)
    outflow face   T = (area / h_x)    * 2 K_cell          (half-cell to p = 0)
    inflow face    RHS += q_in * area                      (flux, no T)
where `area` is the face area normal to the axis (h_y in 2D, h_y h_z in 3D).
The system is symmetric positive definite.

Cells are ordered C-style on an (nx, ny[, nz]) array: m = (ix*ny + iy)*nz + iz
(nz = 1 in 2D), so reshape(shape) gives an [ix, iy(, iz)] array.

Speed
-----
The sparsity pattern (CSR indptr/indices) is built once.  Each solve only
fills the CSR values, with a numba-compiled kernel when numba is installed
(`_fill_csr_numba`, the pattern of simulators_tools._assemble_slab_coo) and
with an equivalent vectorized NumPy path otherwise (`_fill_csr_numpy`).  The
two are checked against each other in tests/test_fv.py.  Linear solvers
(benchmarks/bench_solver.py):
  "splu"     SuperLU with the MMD_AT_PLUS_A ordering, as in
             simulators_tools.darcy_solver_slab_fast; fastest in 2D
             (1.5-1.7x over spsolve at 50^2-100^2)
  "cg"       Jacobi-scaled conjugate gradients; fastest in 3D, where direct
             factorization fill-in explodes (30^3: ~0.14 s vs ~5.6 s)
  "spsolve"  SciPy default (the July configuration)
  "auto"     splu in 2D, cg in 3D (default)

Set BRHO_NO_NUMBA=1 to force the NumPy path.
"""
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    if os.environ.get("BRHO_NO_NUMBA"):
        raise ImportError
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                       # plain-Python stand-in
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


# ---------------------------------------------------------------- kernels
@njit(inline="always", cache=True)
def _harm2(k1, k2):
    """2 k1 k2 / (k1 + k2): the harmonic mean of two cells sharing a face."""
    return 2.0 * k1 * k2 / (k1 + k2)


@njit(cache=True)
def _fill_csr_numba(k, nx, ny, nz, fx, fy, fz, data):
    """Fill CSR values in place.  Row m holds its entries in ascending column
    order: x-, y-, z-, diagonal, z+, y+, x+ (the pattern built by
    `csr_pattern`).  fx, fy, fz are the face-area / distance factors."""
    sx = ny * nz
    sy = nz
    p = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                m = ix * sx + iy * sy + iz
                km = k[m]
                txm = _harm2(k[m - sx], km) * fx if ix > 0 else 0.0
                tym = _harm2(k[m - sy], km) * fy if iy > 0 else 0.0
                tzm = _harm2(k[m - 1], km) * fz if iz > 0 else 0.0
                tzp = _harm2(k[m + 1], km) * fz if iz < nz - 1 else 0.0
                typ = _harm2(k[m + sy], km) * fy if iy < ny - 1 else 0.0
                if ix < nx - 1:
                    txp = _harm2(k[m + sx], km) * fx
                    diag_extra = 0.0
                else:
                    txp = 0.0
                    diag_extra = 2.0 * km * fx          # outflow, p = 0
                if ix > 0:
                    data[p] = -txm; p += 1
                if iy > 0:
                    data[p] = -tym; p += 1
                if iz > 0:
                    data[p] = -tzm; p += 1
                data[p] = txm + tym + tzm + tzp + typ + txp + diag_extra
                p += 1
                if iz < nz - 1:
                    data[p] = -tzp; p += 1
                if iy < ny - 1:
                    data[p] = -typ; p += 1
                if ix < nx - 1:
                    data[p] = -txp; p += 1
    return p


def csr_pattern(nx, ny, nz):
    """indptr, indices of the 5-point (2D) / 7-point (3D) stencil, with the
    column order used by the fill kernels."""
    N = nx * ny * nz
    sx, sy = ny * nz, nz
    ix, iy, iz = np.unravel_index(np.arange(N), (nx, ny, nz))
    m = np.arange(N)
    # (offset, mask) in ascending column order
    nbrs = [(-sx, ix > 0), (-sy, iy > 0), (-1, iz > 0), (0, np.ones(N, bool)),
            (1, iz < nz - 1), (sy, iy < ny - 1), (sx, ix < nx - 1)]
    counts = sum(mask.astype(np.int64) for _, mask in nbrs)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    cols = np.stack([np.where(mask, m + off, -1) for off, mask in nbrs], 1)
    indices = cols[cols >= 0].astype(np.int64)      # row-major -> CSR order
    return indptr, indices


class _NumpyFill:
    """Vectorized fallback producing exactly the kernel's CSR values."""

    def __init__(self, nx, ny, nz, fx, fy, fz, indptr, indices):
        N = nx * ny * nz
        self.N = N
        sx, sy = ny * nz, nz
        ix, iy, iz = np.unravel_index(np.arange(N), (nx, ny, nz))
        row = np.repeat(np.arange(N), np.diff(indptr))
        off = indices - row
        self.is_diag = off == 0
        self.nb_row, self.nb_col = row[~self.is_diag], indices[~self.is_diag]
        o = off[~self.is_diag]
        self.nb_f = np.where(np.abs(o) == sx, fx, np.where(np.abs(o) == sy, fy, fz)) \
            if nz > 1 else np.where(np.abs(o) == sx, fx, fy)
        self.diag_pos = np.flatnonzero(self.is_diag)
        self.out_cells = np.flatnonzero(ix == nx - 1)
        self.fx = fx

    def __call__(self, k, data):
        t = _harm2_np(k[self.nb_row], k[self.nb_col]) * self.nb_f
        data[~self.is_diag] = -t
        diag = np.bincount(self.nb_row, weights=t, minlength=self.N)
        diag[self.out_cells] += 2.0 * k[self.out_cells] * self.fx
        data[self.diag_pos] = diag


def _harm2_np(k1, k2):
    return 2.0 * k1 * k2 / (k1 + k2)


# ---------------------------------------------------------------- model
class Permeameter:
    """Structured-grid permeameter; shape = (nx, ny) or (nx, ny, nz)."""

    def __init__(self, shape, q_in=1.0, solver="auto", backend="auto",
                 cg_rtol=1e-10, cg_maxiter=20000):
        shape = tuple(int(s) for s in shape)
        if len(shape) not in (2, 3):
            raise ValueError(shape)
        self.shape, self.dim = shape, len(shape)
        nx, ny = shape[:2]
        nz = shape[2] if self.dim == 3 else 1
        self._n3 = (nx, ny, nz)
        self.N = nx * ny * nz
        # "auto": direct SuperLU in 2D, Jacobi-scaled CG in 3D (benchmarks/)
        if solver == "auto":
            solver = "splu" if self.dim == 2 else "cg"
        self.q_in, self.solver = q_in, solver
        self.cg_rtol, self.cg_maxiter = cg_rtol, cg_maxiter
        hx, hy = 1.0 / nx, 1.0 / ny
        hz = 1.0 / nz if self.dim == 3 else 1.0
        self.hs = (hx, hy, hz)[:self.dim]
        self.h = hx
        # face-area / distance factors (2D: hy/hx, hx/hy)
        self._f = (hy * hz / hx, hx * hz / hy, hx * hy / hz if self.dim == 3 else 0.0)
        self._area_in = hy * hz

        self.indptr, self.indices = csr_pattern(nx, ny, nz)
        self.nnz = len(self.indices)
        self.backend = ("numba" if HAVE_NUMBA else "numpy") if backend == "auto" else backend
        if self.backend == "numba" and not HAVE_NUMBA:
            raise RuntimeError("numba requested but not installed")
        self._np_fill = _NumpyFill(nx, ny, nz, *self._f, self.indptr, self.indices)

        cells = np.stack(np.unravel_index(np.arange(self.N), shape), 1)
        self.mL = np.flatnonzero(cells[:, 0] == 0)          # inflow cells
        self.mR = np.flatnonzero(cells[:, 0] == nx - 1)     # outflow cells
        self._b = np.zeros(self.N)
        self._b[self.mL] = q_in * self._area_in
        self.axes = [(np.arange(s) + 0.5) / s for s in shape]
        self.xc = self.axes[0]
        self.points = np.stack(np.meshgrid(*self.axes, indexing="ij"), -1).reshape(-1, self.dim)
        # chessboard "black" cells: index sum even
        self.obs_mask = (cells.sum(1) % 2 == 0)

    # ------------------------------------------------------------- solve
    def csr_data(self, K):
        k = np.ascontiguousarray(np.asarray(K, dtype=np.float64).ravel())
        data = np.empty(self.nnz)
        if self.backend == "numba":
            _fill_csr_numba(k, *self._n3, *self._f, data)
        else:
            self._np_fill(k, data)
        return data

    def matrix(self, K):
        """SPD system matrix A (CSR) with A p = b."""
        return sp.csr_matrix((self.csr_data(K), self.indices, self.indptr),
                             shape=(self.N, self.N))

    def solve(self, K):
        """Cell pressures p (N,)."""
        A = self.matrix(K)
        if self.solver == "splu":
            return spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(self._b)
        if self.solver == "spsolve":
            return spla.spsolve(A.tocsc(), self._b)
        if self.solver == "cg":
            return self._cg(A)
        raise ValueError(self.solver)

    def _cg(self, A):
        """Conjugate gradients on the symmetrically Jacobi-scaled system
        D A D y = D b, p = D y, D = diag(A)^(-1/2).  Relative residual
        tolerance `cg_rtol` (default 1e-10; pressure errors ~1e-11 relative,
        far below the data noise)."""
        d = 1.0 / np.sqrt(A.diagonal())
        As = A.multiply(d[:, None]).multiply(d[None, :]).tocsr()
        bs = d * self._b
        try:
            y, info = spla.cg(As, bs, rtol=self.cg_rtol, maxiter=self.cg_maxiter)
        except TypeError:                          # SciPy < 1.12
            y, info = spla.cg(As, bs, tol=self.cg_rtol, maxiter=self.cg_maxiter)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
        return d * y

    def keff(self, K, p):
        """Keff = Q L / (A dp); inflow-face pressure reconstructed from the
        influx, p_face = p_cell + q (h_x/2) / K_cell; Q = q A, A = L = 1."""
        k = np.asarray(K, float).ravel()
        p_face = p[self.mL] + self.q_in * (self.hs[0] / 2) / k[self.mL]
        return self.q_in / p_face.mean()

    def forward(self, K):
        p = self.solve(K)
        return p, p[self.obs_mask], self.keff(K, p)

    def scaling_check(self, K, c=3.7):
        p1, p2 = self.solve(K), self.solve(c * np.asarray(K))
        e_p = np.abs(p2 - p1 / c).max() / np.abs(p1 / c).max()
        k1, k2 = self.keff(K, p1), self.keff(c * np.asarray(K), p2)
        return dict(p_rel_err=float(e_p), keff_rel_err=float(abs(k2 / (c * k1) - 1)))

    def grid(self, v):
        return np.asarray(v).reshape(self.shape)
