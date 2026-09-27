"""
One object bundling physics + KLE + synthetic data + likelihood, built from a
ProblemConfig.  Every runner starts with `prob = Problem(pc)`.

Likelihood (pressure data; optional log-Keff term):
    Phi(K) = ||d_p - O p(K)||^2 / (2 sigma_p^2)
           [+ (log K* - log Keff(K))^2 / (2 sigma_e^2)]
"""
import numpy as np
from .physics2d import Darcy2D
from .kle import KLE
from .truth import make_synthetic


class Problem:
    def __init__(self, pc):
        self.pc = pc
        self.phys = Darcy2D(pc.n, pc.q_in)
        self.kle = KLE(self.phys.points, pc.ell, pc.energy)
        self.syn = make_synthetic(self.phys, self.kle, pc)
        self.r = self.kle.r

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
