"""
One object bundling physics + KLE + synthetic data + likelihood, built from a
ProblemConfig.  Every runner starts with `prob = Problem(pc)`.

Frozen runs
-----------
`Problem(pc, frozen=<results dir>)` loads the KLE basis (kle.npz) and the
synthetic data (truth.npz) saved by that run's setup step instead of
recomputing them.  The eigenvectors of C1 inside degenerate eigenspaces
depend on floating-point rounding (BLAS build, CPU, thread count), so a basis
recomputed on another node or in another process can differ by a rotation:
the same theta would then describe a different field.  Freezing makes every
chain and every report of a run use the one basis the run was set up with.
`Problem.freeze(dir)` writes both files.

Likelihood (pressure data; optional log-Keff term):
    Phi(K) = ||d_p - O p(K)||^2 / (2 sigma_p^2)
           [+ (log K* - log Keff(K))^2 / (2 sigma_e^2)]
"""
import os
import numpy as np
from .physics2d import Darcy2D
from .kle import KLE
from .truth import make_synthetic, Synthetic


class Problem:
    def __init__(self, pc, frozen=None):
        self.pc = pc
        self.phys = Darcy2D(pc.n, pc.q_in, solver=pc.solver, backend=pc.backend)
        kle_path = os.path.join(frozen, "kle.npz") if frozen else None
        if kle_path and os.path.exists(kle_path):
            self.kle = KLE.load(kle_path, self.phys.points, self.phys.axes)
            self.syn = Synthetic.load(os.path.join(frozen, "truth.npz"))
            if self.kle.N != self.phys.N or abs(self.kle.ell - pc.ell) > 0:
                raise ValueError(f"frozen KLE in {frozen} does not match the config")
            self.frozen = True
        else:
            if frozen:
                raise FileNotFoundError(f"{kle_path} missing: run the setup step")
            self.kle = KLE(self.phys.points, pc.ell, pc.energy,
                           axes=self.phys.axes, method=pc.kle_method)
            self.syn = make_synthetic(self.phys, self.kle, pc)
            self.frozen = False
        self.r = self.kle.r

    def freeze(self, directory):
        self.kle.save(os.path.join(directory, "kle.npz"))
        self.syn.save(os.path.join(directory, "truth.npz"))

    def potential(self, K, use_keff=False):
        """Returns (Phi, p_obs, Keff) for a permeability field K (N,)."""
        p, pobs, keff = self.phys.forward(K)
        res = (self.syn.d_p - pobs) / self.syn.sigma_p
        phi = 0.5 * res @ res
        if use_keff:
            phi += 0.5 * ((np.log(self.syn.keff_obs) - np.log(keff))
                          / self.syn.sigma_e) ** 2
        return float(phi), pobs, float(keff)

    def noise_floor(self, use_keff=False):
        return (self.syn.n_obs + int(use_keff)) / 2.0
