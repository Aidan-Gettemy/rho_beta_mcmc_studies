"""GLAM likelihood: normalized squared error with one precision (sigma) per data type.

    Phi = sum_d  ||ref_d - sim_d||^2 / ||ref_d||^2 / (2 sigma_d^2)          (sigma, not variance)

    concentration : the full tracer field (n_cells x n_times) compared against Syn_2mm_TC.dat, sigma = 0.001
    pressure drop : the scalar solver pressure drop [Pa] against Syn_2mm_pDrop.dat (psi -> Pa), sigma = 0.002

log-likelihood = -Phi.  A failed forward model (NaN/inf anywhere) gives Phi = +inf, i.e. always rejected.
"""
from dataclasses import dataclass
import numpy as np

PSI_TO_PA = 6894.757293168


@dataclass
class GlamData:
    conc: np.ndarray          # (n_cells, n_times)
    dp_pa: float              # reference pressure drop [Pa]

    @staticmethod
    def load(conc_path, dp_path, dp_unit="psi"):
        conc = np.loadtxt(conc_path)
        if conc.ndim == 1:
            conc = conc[:, None]
        dp = float(np.loadtxt(dp_path))
        return GlamData(conc, dp * (PSI_TO_PA if dp_unit == "psi" else 1.0))


def normalized_sq_error(ref, sim):
    ref = np.asarray(ref, float); sim = np.asarray(sim, float)
    return float(np.sum((ref - sim) ** 2) / np.sum(ref ** 2))


def phi_like(data: GlamData, conc_sim, dp_sim_pa, sigma_conc=1e-3, sigma_dp=2e-3):
    """Returns (Phi, (Phi_conc, Phi_dp))."""
    if conc_sim is None or not np.isfinite(dp_sim_pa) or not np.all(np.isfinite(conc_sim)):
        return np.inf, (np.inf, np.inf)
    if np.shape(conc_sim) != data.conc.shape:
        raise ValueError(f"simulated concentration has shape {np.shape(conc_sim)}, data has {data.conc.shape}")
    pc = normalized_sq_error(data.conc, conc_sim) / (2.0 * sigma_conc ** 2)
    pd = normalized_sq_error(data.dp_pa, dp_sim_pa) / (2.0 * sigma_dp ** 2)
    return pc + pd, (pc, pd)
