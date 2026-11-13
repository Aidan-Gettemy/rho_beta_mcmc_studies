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
    @staticmethod
    def _ramp(i, c0, c1, halo):
        d = np.maximum(c0 - i, 0) + np.maximum(i - c1, 0)           # cells outside the core
        return 1.0 - d / (halo + 1.0)

    def phi(self, n, k):
        """Unnormalised weight of block k on the grid: 1 on the core, linear ramp over the halo, 0 outside the extended rectangle."""
        cx0, cx1, cy0, cy1 = self._core_rects[k]; ex0, ex1, ey0, ey1 = self._ext_rects[k]
        ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        inside = (ix >= ex0) & (ix <= ex1) & (iy >= ey0) & (iy <= ey1)
        return (inside * self._ramp(ix, cx0, cx1, self.halo) * self._ramp(iy, cy0, cy1, self.halo)).ravel()

    def assembly_weights(self, n):
        P = np.stack([self.phi(n, k) for k in range(len(self._core_rects))])
        return P / P.sum(0, keepdims=True)                         # partition of unity (sum over blocks = 1 everywhere)

    def _column(self, S, n, k, c, W):
        cells, core, M = c
        E = np.zeros((S.kle.N, M.shape[1]))
        if W is None:
            in_core = np.isin(cells, core)
            E[cells[in_core]] = M[in_core]                                   # restriction to the core, zero elsewhere
        else:
            E[cells] = W[k][cells][:, None] * M                              # smooth partition-of-unity weights
        return S.kle.project(E)

    def __init__(self, S, ms, halo, local_energy=0.95, kernel="sqexp", halo_sensors=False, blend=False):
        n = S.n
        kp = dict(ell=S.ell, sigma2=1.0)
        cov = KN.get(kernel)["cov"]
        self.halo = int(halo)
        self.blend = bool(blend)                         # assembly: smooth partition of unity over the overlaps instead of the cores only
        self.halo_sensors = bool(halo_sensors)          # True: the block likelihood uses ALL sensors of the extended rectangle
        self.block_cells, self.block_psi, self.block_mu, self.block_core, self.r_k = [], [], [], [], []
        cols, self._core_rects, self._ext_rects = [], [], []
        for core in ms.block_cells:
            core = np.asarray(core)
            ix, iy = core // n, core % n
            ix0, ix1, iy0, iy1 = int(ix.min()), int(ix.max()), int(iy.min()), int(iy.max())
            ix0, ix1 = max(0, ix0 - halo), min(n - 1, ix1 + halo)
            iy0, iy1 = max(0, iy0 - halo), min(n - 1, iy1 + halo)
            gx, gy = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1), indexing="ij")
            cells = (gx * n + gy).ravel()                                    # ascending rectangle
            self._core_rects.append((int(ix.min()), int(ix.max()), int(iy.min()), int(iy.max())))
            self._ext_rects.append((ix0, ix1, iy0, iy1))
            P = np.asarray(S.points)[cells]
            mu, psi = np.linalg.eigh(cov(P, P, kp))
            mu, psi = np.clip(mu[::-1], 0.0, None), psi[:, ::-1]
            nk = local_truncation(mu, local_energy)
            self.block_cells.append(cells)
            self.block_psi.append(psi[:, :nk].copy())
            self.block_mu.append(mu[:nk].copy())
            self.block_core.append(core)
            self.r_k.append(nk)
            cols.append((cells, core, psi[:, :nk] * np.sqrt(mu[:nk])))
        W = self.assembly_weights(n) if self.blend else None
        cols = [self._column(S, n, k, c, W) for k, c in enumerate(cols)]
        self.B = np.hstack(cols)
        lines = set()                                   # faces (cell b-1 | b) on the extended interior boundaries
        for cells in self.block_cells:
            ix, iy = cells // n, cells % n
            for lo, hi in ((ix.min(), ix.max()), (iy.min(), iy.max())):
                if lo > 0:
                    lines.add(int(lo))
                if hi < n - 1:
                    lines.add(int(hi) + 1)
        self.boundary_lines = sorted(lines)
        self.p = self.B.shape[1]
