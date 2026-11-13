"""
Overlapping subdomains for the domain-decomposed study (docs/dd_overlap_robin_experiment.md).

Every subdomain of a decomposition `ms` (its CORE) is extended by `halo` cells towards each interior interface (not
beyond the domain).  The extended rectangle carries its own local KLE (same kernel, 95% local energy), its own local
forward problem with the interface condition (Dirichlet spline, or Robin, brho/msm_dd.py) on the EXTENDED boundary, and
only the sensors of the core enter its likelihood (global weight), so the local potentials still add up to the global
one when the interface data are exact.  The global field is assembled from the cores only: the zero-extension of a local
mode is its restriction to the core, projected on the global KLE exactly as before (theta = B c_all).

OverlapDecomp has the attributes of an MSInnovation that DDGrid / assemble_theta / split_blocks use: block_cells (the
extended rectangles, ascending), block_psi, block_mu, r_k, p, B; plus block_core (global cell indices of each core).
"""
import numpy as np

from . import kernels as KN
from .multiscale import local_truncation


class OverlapDecomp:
    def __init__(self, S, ms, halo, local_energy=0.95, kernel="sqexp"):
        n = S.n
        kp = dict(ell=S.ell, sigma2=1.0)
        cov = KN.get(kernel)["cov"]
        self.halo = int(halo)
        self.block_cells, self.block_psi, self.block_mu, self.block_core, self.r_k = [], [], [], [], []
        cols = []
        for core in ms.block_cells:
            core = np.asarray(core)
            ix, iy = core // n, core % n
            ix0, ix1, iy0, iy1 = int(ix.min()), int(ix.max()), int(iy.min()), int(iy.max())
            ix0, ix1 = max(0, ix0 - halo), min(n - 1, ix1 + halo)
            iy0, iy1 = max(0, iy0 - halo), min(n - 1, iy1 + halo)
            gx, gy = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1), indexing="ij")
            cells = (gx * n + gy).ravel()                                    # ascending rectangle
            P = np.asarray(S.points)[cells]
            mu, psi = np.linalg.eigh(cov(P, P, kp))
            mu, psi = np.clip(mu[::-1], 0.0, None), psi[:, ::-1]
            nk = local_truncation(mu, local_energy)
            self.block_cells.append(cells)
            self.block_psi.append(psi[:, :nk].copy())
            self.block_mu.append(mu[:nk].copy())
            self.block_core.append(core)
            self.r_k.append(nk)
            E = np.zeros((S.kle.N, nk))
            in_core = np.isin(cells, core)
            E[cells[in_core]] = psi[in_core, :nk] * np.sqrt(mu[:nk])         # restriction to the core, zero elsewhere
            cols.append(S.kle.project(E))
        self.B = np.hstack(cols)
        self.p = self.B.shape[1]
