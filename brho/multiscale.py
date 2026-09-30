"""
Multiscale (local-KLE) innovation for the theta pCN move, with variance
control IN DISTRIBUTION (whitening), after F. Pereira, "Variance control in
distribution for a multiscale (local-KLE) pCN sampler" (July 2026) and
Sec. 6 of "Two parametrization degeneracies ..." (Sept. 2026).

Offline, no PDE solves (pure Gaussian linear algebra):
  1. partition the grid into blocks x blocks subdomains Omega_k;
  2. local KLE of the unit-variance kernel C1 restricted to Omega_k,
     truncated at `local_energy` (r_k modes);
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


class MSInnovation:
    def __init__(self, kle, points, kernel="sqexp", kernel_params=None,
                 blocks=2, local_energy=0.95, tol=1e-10):
        kp = dict(kernel_params or {})
        cov = KN.get(kernel)["cov"]
        self.r = kle.r
        self.blocks, self.local_energy = int(blocks), float(local_energy)
        self.dim = int(np.asarray(points).shape[1])
        bidx = block_index(points, self.blocks)
        cols, rk = [], []
        for k in range(bidx.max() + 1):
            idx = np.flatnonzero(bidx == k)
            Pk = np.asarray(points)[idx]
            mu, psi = np.linalg.eigh(cov(Pk, Pk, kp))
            mu, psi = np.clip(mu[::-1], 0.0, None), psi[:, ::-1]
            n_k = int(np.searchsorted(np.cumsum(mu) / mu.sum(), self.local_energy) + 1)
            rk.append(n_k)
            E = np.zeros((kle.N, n_k))
            E[idx] = psi[:, :n_k] * np.sqrt(mu[:n_k])                # zero-extended
            cols.append(kle.project(E))                               # (r, n_k)
        self.B = np.hstack(cols)                                      # (r, p)
        self.p = self.B.shape[1]
        self.r_k = rk
        self.Sigma = self.B @ self.B.T
        d, V = np.linalg.eigh(self.Sigma)
        d, V = d[::-1], V[:, ::-1]
        self.d = d
        self.r_plus = int((d > tol * d[0]).sum())
        Vp, self.V0 = V[:, :self.r_plus], V[:, self.r_plus:]
        self.W = Vp @ np.diag(1.0 / np.sqrt(d[:self.r_plus])) @ Vp.T
        self.WB = self.W @ self.B
        cov_white = self.W @ self.Sigma @ self.W.T + self.V0 @ self.V0.T
        self.operator_check = float(np.linalg.norm(cov_white - np.eye(self.r))
                                    / np.sqrt(self.r))

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
                    r=self.r, p=self.p, r_k=self.r_k, r_plus=self.r_plus,
                    sigma_ms_eig_max=float(self.d[0]),
                    sigma_ms_eig_min=float(self.d[-1]),
                    sigma_ms_trace_over_r=float(self.d.sum() / self.r),
                    operator_check=self.operator_check)


def from_problem(prob, blocks=2, local_energy=0.95):
    from .problem import kernel_params
    return MSInnovation(prob.kle, prob.phys.points, prob.pc.kernel,
                        kernel_params(prob.pc), blocks, local_energy)
