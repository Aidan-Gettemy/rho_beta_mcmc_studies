"""Cheap forward models with the GlamModel interface, for tests and dress rehearsals (no solver, no tracer).

LinearGaussianModel   Phi(x) = ||y - A x||^2 / (2 s^2) on x = (theta, w_rho), prior N(0, I) on everything, no rho
                      truncation: the posterior is Gaussian with precision I + A^T A / s^2 (exact moments), which tests the
                      pCN moves, acceptance, adaptation and exchanges against a known answer.
ToyModel              same state/field structure as the real model (assemble, log K = log beta(rho) + rho YP, PCHIP beta)
                      with a cheap surrogate in place of the solver and tracer: dp ~ 1/Keff and
                      conc(cell, t) = 1 - exp(-t K / Kexp).  Likelihood is the real two-sigma normalized error.
"""
import numpy as np

from ..glam_likelihood import GlamData, phi_like
from .anchors import BuiltField
from .model import BetaPchip, GlamModel, SimData


class LinearGaussianModel(GlamModel):
    def __init__(self, n_sub=3, m=4, n_data=12, s=0.5, seed=0, x_true=None):
        rng = np.random.default_rng(seed)
        self.n_sub, self.m = n_sub, m
        self.block_dims = [m] * n_sub
        self.d = n_sub * m + 1
        self.A = rng.standard_normal((n_data, self.d)) / np.sqrt(self.d)
        x_true = rng.standard_normal(self.d) if x_true is None else x_true
        self.s = s
        self.y = self.A @ x_true + s * rng.standard_normal(n_data)
        self.rho_range = (0.0, np.inf)
        self.mu_rho, self.s_rho = 0.0, 1.0

    @staticmethod
    def _x(state):
        return np.concatenate([np.asarray(b, float).ravel() for b in state])

    def build(self, state):
        return BuiltField(simulation_field=self._x(state), prior_field=None)

    def simulate(self, field, rank=0, want_pressure=False):
        return SimData(0.0, None)

    def phi(self, state, field, sim):
        r = self.y - self.A @ self._x(state)
        return float(r @ r / (2 * self.s ** 2))

    def posterior(self):
        prec = np.eye(self.d) + self.A.T @ self.A / self.s ** 2
        cov = np.linalg.inv(prec)
        return cov @ self.A.T @ self.y / self.s ** 2, cov


class ToyModel(GlamModel):
    def __init__(self, beta: BetaPchip, n_sub=4, m=6, n_cells=120, n_times=4, truth=None, seed=0,
                 sigma_conc=0.02, sigma_dp=0.02, rho_range=(0.05, 5.0), mu_rho=0.0, s_rho=0.75):
        rng = np.random.default_rng(seed)
        self.beta, self.n_sub, self.m, self.N = beta, n_sub, m, n_cells
        self.block_dims = [m] * n_sub
        self.T = [rng.standard_normal((n_cells, m)) / np.sqrt(n_sub * m) for _ in range(n_sub)]
        self.times = np.linspace(0.2, 2.0, n_times)
        self.sigma_conc, self.sigma_dp = sigma_conc, sigma_dp
        self.rho_range = (rho_range[0], min(rho_range[1], beta.rho_max))
        self.mu_rho, self.s_rho = mu_rho, s_rho
        st = truth if truth is not None else self.sample_state(rng, self.block_dims)
        self.truth = st
        sim = self.simulate(self.build(st))
        self.data = GlamData(sim.conc, sim.dp_pa)

    def build(self, state):
        rho = self.rho_of(state[self.n_sub][0])
        Y = sum(T @ np.asarray(b, float) for T, b in zip(self.T, state[:self.n_sub]))
        logK = float(self.beta.log_beta(rho)) + rho * Y
        return BuiltField(simulation_field=np.exp(logK), prior_field=dict(YP=Y, rho=rho, logK=logK))

    def simulate(self, field, rank=0, want_pressure=False):
        K = field.simulation_field if isinstance(field, BuiltField) else field
        keff = np.exp(np.mean(np.log(K)))
        conc = 1.0 - np.exp(-np.outer(K / self.beta.Kexp, self.times) * 0.5)
        return SimData(3314.6 * self.beta.Kexp / keff, conc,
                       pressure=np.log(K) if want_pressure else None)

    def phi(self, state, field, sim):
        if not self.rho_ok(state) or sim is None or sim.conc is None:
            return np.inf
        return phi_like(self.data, sim.conc, sim.dp_pa, self.sigma_conc, self.sigma_dp)[0]
