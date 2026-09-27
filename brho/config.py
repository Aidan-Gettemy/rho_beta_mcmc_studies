"""
All experiment settings in one place.

Every runner builds its config from these dataclasses, applies optional
QUICK overrides (tiny budgets for smoke tests) and CLI overrides, and writes
the final config into the run's manifest.json so every figure can be traced
back to the exact settings that produced it.
"""
from dataclasses import dataclass, asdict, replace, fields
import json


# ------------------------------------------------------------------ problem
@dataclass(frozen=True)
class ProblemConfig:
    """Physics, prior covariance and synthetic truth (shared by E2-E4)."""
    n: int = 16                    # cells per side (16x16 = 256 cells)
    ell: float = 0.1               # squared-exponential correlation length
    energy: float = 0.95           # KLE energy fraction  (-> r = 56 in 2D)
    q_in: float = 1.0              # unit influx on the left face
    beta_true: float = 2.3
    rho_true: float = 1.5
    # "standardized": G_in centered and scaled to unit sample variance, so
    #                 (beta, rho) = (2.3, 1.5) are exactly what the data see
    #                 (short-report decision).
    # "raw":          the uncentered Cholesky draw used in the July reports
    #                 (reproduces their data exactly: same seed, same order).
    truth_mode: str = "standardized"
    truth_seed: int = 2026
    noise_rel: float = 0.01        # sigma_p = noise_rel * (max p - min p)
    sigma_e: float = 0.05          # log-scale error of the Keff measurement


# -------------------------------------------------------- E2: 5x5 grid study
@dataclass(frozen=True)
class GridStudyConfig:
    betas: tuple = (0.5, 1.0, 2.3, 5.0, 10.0)
    rhos: tuple = (0.5, 1.0, 1.5, 2.25, 3.0)
    n_iter: int = 4000
    n_burn: int = 2000
    s0: float = 0.15               # initial pCN step size
    acc_lo: float = 0.18           # adapt step toward [acc_lo, acc_hi]
    acc_hi: float = 0.30
    adapt_every: int = 100
    adapt_factor: float = 1.15
    seed0: int = 1000              # chain k uses seed0 + 17 k
    n_workers: int = 8


# ------------------------------------------------------ E3: offline beta curve
@dataclass(frozen=True)
class CurveConfig:
    rho_min: float = 0.05
    rho_max: float = 5.0
    n_rho: int = 25
    n_real: int = 128              # Monte Carlo realizations (shared across rho)
    # "kle": truncated global KLE (same r modes as the sampler; decision)
    # "cholesky": full-rank fields (for the consistency comparison only)
    field: str = "kle"
    # beta_bar = K* / f(rho) with f one of:
    #   "ratio_of_mean": f = mean_i Keff_i           (short-report decision)
    #   "mean_of_ratio": f = 1 / mean_i (1/Keff_i)   (method-report eq. 8)
    #   "median":        f = median_i Keff_i
    statistic: str = "ratio_of_mean"
    seed: int = 99


# ---------------------------------------------- E4: 3-step curve sampler
@dataclass(frozen=True)
class SamplerConfig:
    n_chains: int = 4
    n_total: int = 51000           # iterations per chain (incl. burn-in)
    n_burn: int = 15000
    thin: int = 30
    chunk: int = 5000              # iterations per checkpointed call
    mu_rho: float = 0.0            # log rho = mu_rho + s_rho * w
    s_rho: float = 0.75
    s_init: tuple = (0.15, 0.20, 0.20)     # (s_theta, s_w, s_ridge)
    targets: tuple = (0.25, 0.25, 0.45)    # acceptance targets A, B, C
    adapt_every: int = 100
    adapt_factor: float = 1.12
    use_keff_term: bool = False    # method report: pressure-only likelihood
    init: str = "prior"            # "prior" draws or "zero"
    seed0: int = 101
    n_workers: int = 4
    prior_test: bool = False       # likelihood switched off (Phi = 0)


# ------------------------------------------------------------ quick budgets
QUICK = {
    GridStudyConfig: dict(n_iter=300, n_burn=150, n_workers=2),
    CurveConfig: dict(n_rho=8, n_real=16),
    SamplerConfig: dict(n_total=1200, n_burn=400, thin=10, chunk=600,
                        n_workers=2),
}


def build(cls, quick=False, **overrides):
    """Instantiate `cls` with QUICK overrides (if quick) then explicit ones."""
    cfg = cls()
    if quick:
        cfg = replace(cfg, **QUICK.get(cls, {}))
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **overrides) if overrides else cfg


def add_cli(parser, cls, prefix=""):
    """Expose every scalar field of `cls` as an optional --flag."""
    for f in fields(cls):
        if f.type in (int, float, str, bool, "int", "float", "str", "bool"):
            typ = {"int": int, "float": float, "str": str,
                   "bool": _bool}.get(f.type, f.type)
            if typ is bool:
                typ = _bool
            parser.add_argument(f"--{prefix}{f.name.replace('_', '-')}",
                                dest=f"{prefix}{f.name}", type=typ,
                                default=None)


def from_cli(args, cls, prefix="", quick=False):
    vals = {f.name: getattr(args, f"{prefix}{f.name}", None)
            for f in fields(cls)}
    return build(cls, quick=quick, **vals)


def to_dict(*cfgs):
    return {type(c).__name__: asdict(c) for c in cfgs}


def from_dict(cls, d):
    d = dict(d)
    for f in fields(cls):
        if isinstance(d.get(f.name), list):
            d[f.name] = tuple(d[f.name])
    return cls(**{k: v for k, v in d.items()
                  if k in {f.name for f in fields(cls)}})


def _bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def dumps(*cfgs):
    return json.dumps(to_dict(*cfgs), indent=1)
