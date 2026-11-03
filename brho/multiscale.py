"""
Multiscale (local-KLE) innovation for the theta pCN move, with variance
control IN DISTRIBUTION (whitening), after F. Pereira, "Variance control in
distribution for a multiscale (local-KLE) pCN sampler" (July 2026) and
Sec. 6 of "Two parametrization degeneracies ..." (Sept. 2026).

Offline, no PDE solves (pure Gaussian linear algebra):
  1. partition the grid into blocks x blocks subdomains Omega_k;
  2. local KLE of the unit-variance kernel C1 restricted to Omega_k,
     truncated at `local_energy` (r_k modes; modes tied with the last kept
     one are kept too, so the result does not depend on eigh's basis inside
     degenerate eigenspaces);
  3. assemble-and-project matrix B (r x p), column (k, j) =
         proj( E_k sqrt(mu_j^k) psi_j^k ),   proj = the global-KLE L2 projection
     (E_k extends by zero outside Omega_k);
  4. Sigma_ms = B B^T (exact law of the raw innovation B eta, eta ~ N(0, I_p));
  5. Sigma_ms = V D V^T, r_+ = #{d_i > tol d_1}, V = [V_+ V_0];
     W = V_+ diag(d^-1/2) V_+^T;
  6. whitened innovation  xi = W B eta + V_0 zeta,  zeta ~ N(0, I_{r - r_+}),
     whose law is EXACTLY N(0, I_r):  W Sigma_ms W^T + V_0 V_0^T = I_r.

Innovations (MSInnovation.draw):
  "global"    xi ~ N(0, I_r)                    plain pCN (E5)
  "ms_raw"    xi = B eta   ~ N(0, Sigma_ms)     WRONG law: pCN with a
              likelihood-only acceptance then targets a different measure
  "ms_white"  whitened, ~ N(0, I_r) exactly     correct

Note (Sec. 6 of the degeneracies notes): the whitened innovation has the
same law as the global one, so pCN with "ms_white" has the SAME transition
kernel as plain global pCN.  It is a correctness device (it passes the prior
test), not a speed-up; a run with it must reproduce the global-pCN results
within Monte Carlo error.
"""
import numpy as np

from . import kernels as KN
from .kle import TIE_RTOL

INNOVATIONS = ("global", "ms_raw", "ms_white")


def block_index(points, blocks):
    """Subdomain index of every cell for a blocks x blocks (x blocks) partition
    of the unit square (cube)."""
    P = np.asarray(points, float)
    ij = np.minimum((P * blocks).astype(int), blocks - 1)          # (N, d)
    idx = np.zeros(len(P), int)
    for a in range(P.shape[1]):
        idx = idx * blocks + ij[:, a]
    return idx


def dual_block_index(points, blocks):
    """Subdomain index of every cell for the DUAL decomposition of a
    blocks x blocks (x blocks) primal one: interfaces at (i + 1/2)/blocks, so
    (blocks + 1)^dim blocks, the outer ones half-size (edges) or quarter-size
    (corners).  Same interfaces and lexicographic order as
    blockpcn.boxes(dim, blocks, dual=True)."""
    from .blockpcn import _edges
    inner = _edges(blocks, True)[1:-1]                              # the interior interfaces
    P = np.asarray(points, float)
    m = len(inner) + 1
    idx = np.zeros(len(P), int)
    for a in range(P.shape[1]):
        idx = idx * m + np.searchsorted(inner, P[:, a], side="right")
    return idx


def local_truncation(mu, energy, rtol=TIE_RTOL):
    """Number of local modes kept: the smallest n reaching `energy`, extended
    to include every mode tied with the last kept one (|mu_n - mu_{n-1}| <=
    rtol mu_{n-1}).  A subdomain is a square or cube with a symmetric kernel,
    so its spectrum has exact ties (permutations of the axes); cutting inside
    a tied group would keep an arbitrary part of that eigenspace (eigh's
    choice, which depends on machine and threads), and Sigma_ms = B B^T would
    change from run to run.  With whole groups kept, the kept subspace is
    unique and Sigma_ms is the same everywhere (B changes only by a rotation
    inside the subspace, which B B^T does not see).  Same rule as the global
    KLE (brho/kle.py)."""
    mu = np.asarray(mu, float)
    n = int(np.searchsorted(np.cumsum(mu) / mu.sum(), energy) + 1)
    while n < len(mu) and abs(mu[n] - mu[n - 1]) <= rtol * mu[n - 1]:
        n += 1
    return n


class MSInnovation:
    """Options beyond the original uniform, energy-truncated partition:
      block_idx  explicit subdomain label (0..K-1) of every cell, e.g.
                 dual_block_index(points, 4) for the dual decomposition
                 (`blocks` is then only a label);
      local_dim  keep exactly this many local modes in every block instead of
                 the `local_energy` rule.  Ties are NOT extended here (p is
                 fixed by the caller); `tie_cut[k]` says whether block k's
                 last kept mode is tied with the first dropped one, in which
                 case the kept subspace depends on eigh's basis.
    Also kept per block: cells, local eigenvectors psi and eigenvalues mu, so
    that local_field(c) builds the piecewise field of local coefficients c and
    local_coeffs(f) is its adjoint (c_kj = psi_kj^T f_k / sqrt(mu_kj)); Pi
    (p x r) maps global theta to local coefficients, c = Pi theta."""

    def __init__(self, kle, points, kernel="sqexp", kernel_params=None,
                 blocks=2, local_energy=0.95, tol=1e-10,
                 block_idx=None, local_dim=None):
        kp = dict(kernel_params or {})
        cov = KN.get(kernel)["cov"]
        self.r = kle.r
        self.N = kle.N
        self.blocks, self.local_energy = int(blocks), float(local_energy)
        self.local_dim = None if local_dim is None else int(local_dim)
        self.dim = int(np.asarray(points).shape[1])
        if block_idx is None:
            bidx = block_index(points, self.blocks)
        else:
            bidx = np.asarray(block_idx, int)
            if bidx.shape != (kle.N,):
                raise ValueError(f"block_idx must have one label per cell ({kle.N}), "
                                 f"got shape {bidx.shape}")
        cols, rk = [], []
        self.block_cells, self.block_psi, self.block_mu, self.tie_cut = [], [], [], []
        for k in range(bidx.max() + 1):
            idx = np.flatnonzero(bidx == k)
            if idx.size == 0:
                raise ValueError(f"block {k} has no cells")
            Pk = np.asarray(points)[idx]
            mu, psi = np.linalg.eigh(cov(Pk, Pk, kp))
            mu, psi = np.clip(mu[::-1], 0.0, None), psi[:, ::-1]
            if self.local_dim is None:
                n_k = local_truncation(mu, self.local_energy)
            else:
                n_k = min(self.local_dim, len(mu))
            rk.append(n_k)
            self.tie_cut.append(bool(n_k < len(mu)
                                     and abs(mu[n_k] - mu[n_k - 1]) <= TIE_RTOL * mu[n_k - 1]))
            self.block_cells.append(idx)
            self.block_psi.append(psi[:, :n_k].copy())
            self.block_mu.append(mu[:n_k].copy())
            E = np.zeros((kle.N, n_k))
            E[idx] = psi[:, :n_k] * np.sqrt(mu[:n_k])                # zero-extended
            cols.append(kle.project(E))                               # (r, n_k)
        self.B = np.hstack(cols)                                      # (r, p)
        self.p = self.B.shape[1]
        self.r_k = rk
        self.Pi = self.local_coeffs(kle.T)                            # (p, r)
        self.Sigma = self.B @ self.B.T
        d, V = np.linalg.eigh(self.Sigma)
        d, V = d[::-1], V[:, ::-1]
        self.d = d
        self.r_plus = int((d > tol * d[0]).sum())
        Vp, self.V0 = V[:, :self.r_plus], V[:, self.r_plus:]
        self.Vp = Vp
        self.W = Vp @ np.diag(1.0 / np.sqrt(d[:self.r_plus])) @ Vp.T
        self.WB = self.W @ self.B
        cov_white = self.W @ self.Sigma @ self.W.T + self.V0 @ self.V0.T
        self.operator_check = float(np.linalg.norm(cov_white - np.eye(self.r))
                                    / np.sqrt(self.r))

    def local_field(self, c):
        """Piecewise (discontinuous at the interfaces) field of local
        coefficients c (..., p): block k carries sum_j c_kj sqrt(mu_kj) psi_kj.
        Returns (..., N)."""
        c = np.asarray(c, float)
        lead = c.shape[:-1]
        c2 = c.reshape(-1, self.p)
        f = np.zeros((c2.shape[0], self.N))
        off = 0
        for idx, psi, mu in zip(self.block_cells, self.block_psi, self.block_mu):
            n_k = len(mu)
            f[:, idx] = (c2[:, off:off + n_k] * np.sqrt(mu)) @ psi.T
            off += n_k
        return f.reshape(lead + (self.N,))

    def local_coeffs(self, f):
        """Local coefficients of a cell field f (N,) or (N, m): block by block,
        c_kj = psi_kj^T f_k / sqrt(mu_kj).  Returns (p,) or (p, m).  Lossy: the
        local truncation and the interfaces are not representable."""
        f = np.asarray(f, float)
        f2 = f.reshape(self.N, -1)
        out = np.empty((self.p, f2.shape[1]))
        off = 0
        for idx, psi, mu in zip(self.block_cells, self.block_psi, self.block_mu):
            n_k = len(mu)
            out[off:off + n_k] = (psi.T @ f2[idx]) / np.sqrt(mu)[:, None]
            off += n_k
        return out[:, 0] if f.ndim == 1 else out

    def draw(self, kind, rng):
        if kind == "global":
            return rng.standard_normal(self.r)
        if kind == "ms_raw":
            return self.B @ rng.standard_normal(self.p)
        if kind == "ms_white":
            xi = self.WB @ rng.standard_normal(self.p)
            if self.V0.shape[1]:
                xi = xi + self.V0 @ rng.standard_normal(self.V0.shape[1])
            return xi
        raise ValueError(f"unknown innovation {kind!r}; have {INNOVATIONS}")

    def summary(self):
        return dict(blocks=self.blocks, dim=self.dim, local_energy=self.local_energy,
                    local_dim=self.local_dim, n_blocks=len(self.r_k),
                    tie_cut_blocks=int(sum(self.tie_cut)),
                    r=self.r, p=self.p, r_k=self.r_k, r_plus=self.r_plus,
                    sigma_ms_eig_max=float(self.d[0]),
                    sigma_ms_eig_min=float(self.d[-1]),
                    sigma_ms_trace_over_r=float(self.d.sum() / self.r),
                    operator_check=self.operator_check)


def from_problem(prob, blocks=2, local_energy=0.95):
    from .problem import kernel_params
    return MSInnovation(prob.kle, prob.phys.points, prob.pc.kernel,
                        kernel_params(prob.pc), blocks, local_energy)
