"""Forward models for GLAM.

State = list of n_sub subdomain blocks theta_i (each N(0, I)) followed by the one-element block [w_rho],
w_rho ~ N(0, 1), log rho = mu_rho + s_rho * w_rho, restricted to the PCHIP range (proposals outside are rejected,
i.e. the prior is truncated).  rho is treated as one more coordinate of the state.

  build(state)                   -> BuiltField   Y = sum T_i theta_i ; YP = whitened projection ;
                                                  log K = log beta(rho) + rho * YP ; K = exp(log K)   [m^2]
  simulate(field, rank, ...)     -> SimData      solver (+ tracer): pressure drop [Pa], concentration (N x n_times)
  phi(state, field, sim)         -> float        normalized squared error, 2 sigmas (brho/glam_likelihood.py);
                                                  +inf when rho is outside the range or the forward model failed

`ExternalModel` is the production backend (external Darcy solver + GPU tracer); `brho/glam/toy.py` has cheap
backends for tests.
"""
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..glam_likelihood import GlamData, phi_like
from .anchors import BuiltField


@dataclass
class SimData:
    dp_pa: float
    conc: Any                      # (n_cells, n_times) or None
    pressure: Any = None           # (n_cells,) when requested (post-processing)


class BetaPchip:
    """beta(rho) = Kexp * exp(-h(rho)), h = PCHIP through the MC means (anchored h(0) = 0); valid on [0, rho_max]."""

    def __init__(self, rho_nodes, h_nodes, Kexp):
        self.f = PchipInterpolator(np.asarray(rho_nodes, float), np.asarray(h_nodes, float), extrapolate=False)
        self.Kexp = float(Kexp)
        self.rho_max = float(np.max(rho_nodes))

    @classmethod
    def from_fits_json(cls, path):
        d = json.load(open(path))
        return cls(d["pchip"]["rho"], d["pchip"]["h"], d["Kexp"])

    def h(self, rho):
        return self.f(rho)

    def log_beta(self, rho):
        return np.log(self.Kexp) - self.f(rho)

    def beta(self, rho):
        return np.exp(self.log_beta(rho))


class GlamModel:
    """Base: prior bookkeeping shared by all backends."""
    n_sub = 12
    mu_rho, s_rho = 0.0, 0.75
    rho_range = (0.05, 5.0)

    def rho_of(self, w):
        return float(np.exp(self.mu_rho + self.s_rho * float(w)))

    def rho_ok(self, state):
        r = self.rho_of(state[self.n_sub][0])
        return self.rho_range[0] <= r <= self.rho_range[1]

    def sample_state(self, rng, block_dims):
        for _ in range(10000):
            st = [rng.standard_normal(d) for d in block_dims] + [rng.standard_normal(1)]
            if self.rho_ok(st):
                return st
        raise RuntimeError("could not draw rho inside the PCHIP range from the prior")

    # --- interface used by the sampler and the manager callbacks -------------------------------
    def build(self, state):                       # pragma: no cover - abstract
        raise NotImplementedError

    def simulate(self, field, rank=0, want_pressure=False):   # pragma: no cover - abstract
        raise NotImplementedError

    def phi(self, state, field, sim):             # pragma: no cover - abstract
        raise NotImplementedError

    def phi_of_state(self, state, rank=0):
        """Convenience: full evaluation of one state (skips the solver when rho is out of range)."""
        if not self.rho_ok(state):
            return np.inf, None, None
        field = self.build(state)
        sim = self.simulate(field, rank)
        return self.phi(state, field, sim), field, sim


def read_petsc_vec(path, n=None):
    """Pure-NumPy reader for one PETSc binary Vec (big-endian: int32 classid, int32 n, n float64)."""
    with open(path, "rb") as f:
        head = np.fromfile(f, dtype=">i4", count=2)
        if head[0] != 1211214:
            raise ValueError(f"{path}: not a PETSc Vec file (classid {head[0]})")
        v = np.fromfile(f, dtype=">f8", count=int(head[1]))
    if n is not None and v.size != n:
        raise ValueError(f"{path}: expected {n} values, found {v.size}")
    return v.astype(np.float64)


def assign_gpu(rank):
    """Many ranks share the node's GPUs: rank r uses device r % n_gpus (round robin).  Sets CUDA_VISIBLE_DEVICES for
    this process, so it must run before CuPy is first imported; a no-op when no GPU list is visible.  Returns the
    physical device id (or None)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devs = [d for d in vis.split(",") if d.strip()]
    if not devs:
        n = int(os.environ.get("SLURM_GPUS_ON_NODE", "0") or 0)
        devs = [str(i) for i in range(n)]
    if not devs:
        return None
    d = devs[rank % len(devs)]
    os.environ["CUDA_VISIBLE_DEVICES"] = d
    return d


class TracerRunner:
    """Calls the group's GPU tracer in-process: transpTracer.set_problem_and_call_solver(cfg, velocity_file).
    The tracer is proprietary and lives outside the repository: give its folder (the `transport/` directory of the
    solver repository) as tracer_dir or $TRACER_DIR.  Needs CuPy + petsc4py on the machine that calls it."""

    def __init__(self, cfg, tracer_dir=None, rank=0):
        self.cfg = os.path.abspath(cfg)
        self.gpu = assign_gpu(rank)
        self.tracer_dir = tracer_dir or os.environ.get("TRACER_DIR")
        self._fn = None

    def __call__(self, velocity_file):
        if self._fn is None:
            import sys
            if not self.tracer_dir:
                raise RuntimeError("set tracer_dir or $TRACER_DIR to the solver repository's transport/ folder")
            if self.tracer_dir not in sys.path:
                sys.path.insert(0, self.tracer_dir)
            from transpTracer import set_problem_and_call_solver
            self._fn = set_problem_and_call_solver
        sat = self._fn(self.cfg, velocity_file)
        if hasattr(sat, "get"):                  # CuPy array
            sat = sat.get()
        return np.asarray(sat, dtype=np.float64)


class ExternalModel(GlamModel):
    def __init__(self, field, beta, runner, tracer, data: GlamData, sigma_conc=1e-3, sigma_dp=2e-3,
                 mu_rho=0.0, s_rho=0.75, rho_range=(0.05, 5.0)):
        self.field, self.beta, self.runner, self.tracer, self.data = field, beta, runner, tracer, data
        self.sigma_conc, self.sigma_dp = sigma_conc, sigma_dp
        self.mu_rho, self.s_rho = mu_rho, s_rho
        self.rho_range = (max(rho_range[0], 0.0), min(rho_range[1], beta.rho_max))
        self.n_sub = len(field.T)
        self.block_dims = [T.shape[1] for T in field.T]

    def build(self, state):
        rho = self.rho_of(state[self.n_sub][0])
        Y = self.field.assemble([np.asarray(b, np.float32 if self.field.dtype == np.float32 else float) for b in state[:self.n_sub]])
        YP = self.field.whiten_project(Y).astype(np.float64)
        logK = float(self.beta.log_beta(rho)) + rho * YP
        return BuiltField(simulation_field=np.exp(logK), prior_field=dict(YP=YP, rho=rho, logK=logK))

    def simulate(self, field, rank=0, want_pressure=False):
        K = field.simulation_field if isinstance(field, BuiltField) else field
        res = self.runner.solve(K, tag=f"glam_r{rank}", pressure=want_pressure, velocity=True)
        if res is None:
            return SimData(np.nan, None)
        try:
            conc = self.tracer(res["velocity_file"])
            p = read_petsc_vec(res["pressure_file"], K.size) if want_pressure else None
            return SimData(res["dp"], conc, p)
        finally:
            self.runner.cleanup(res)

    def phi(self, state, field, sim):
        if not self.rho_ok(state):
            return np.inf
        if sim is None or sim.conc is None:
            return np.inf
        return phi_like(self.data, sim.conc, sim.dp_pa, self.sigma_conc, self.sigma_dp)[0]
