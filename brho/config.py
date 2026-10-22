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
    n: int = 32                    # cells per side (32x32 = 1024 cells; >= 3 per
                                   # correlation length: n * ell >= 3; July runs: 16)
    dim: int = 2                   # 2: unit square (n x n); 3: unit cube (n x n x n).
                                   # 3D study (2026-09-29): --dim 3 --n 16 --ell 0.2
                                   # --kle-method nystrom_sep --kle-nq 40 (r = 78)
    ell: float = 0.1               # squared-exponential correlation length
    energy: float = 0.95           # KLE energy fraction  (-> r = 56 in 2D)
    kernel: str = "sqexp"          # covariance kernel (brho/kernels.py)
    # KLE builder (brho/kle.py): "auto" (dense for N<=4096, else nystrom_sep /
    # nystrom) | "dense" | "kron" | "nystrom" | "nystrom_sep"
    kle_method: str = "auto"
    kle_nq: int = 40               # Nystrom quadrature nodes per direction
    kle_quadrature: str = "gauss_legendre"   # | "midpoint"
    q_in: float = 1.0              # unit influx on the left face
    solver: str = "auto"           # "auto" (splu 2D, cg 3D) | "splu" | "cg" | "spsolve"
    backend: str = "auto"          # CSR fill: "auto" (numba if installed) | "numba" | "numpy"
    beta_true: float = 2.3
    rho_true: float = 1.5
    # "standardized": G_in centered and scaled to unit sample variance, so
    #                 (beta, rho) = (2.3, 1.5) are exactly what the data see
    #                 (short-report decision).
    # "raw":          the uncentered Cholesky draw used in the July reports
    #                 (reproduces their data exactly: same seed, same order).
    # "kle":          (2026-09-28 decision) the "standardized" field projected
    #                 onto the span of the truncated KLE the samplers use
    #                 (theta_true = T^+ G, G <- T theta_true): the E1-E4 truth
    #                 minus its small scales, exactly in the sampler's space, so
    #                 no truncation floor.  "kle_raw": the raw draw projected
    #                 (theta_true ~ N(0, I_r) exactly, own spatial mean).  Noise
    #                 is added to the pressures and to Keff as in the other modes.
    #                 The paper must state this (the inverse crime is deliberate;
    #                 the full-rank "standardized" runs are the robustness check).
    truth_mode: str = "standardized"
    truth_seed: int = 2026
    noise_rel: float = 0.01        # sigma_p = noise_rel * (max p - min p)
    sigma_e: float = 0.05          # log-scale error of the Keff measurement
    # Pressure likelihood (brho/problem.py):
    #   "relative" (default): Phi = ||d - G||^2 / (2 sigma_rel^2 ||d||^2)
    #                             = relerr^2 / (2 sigma_rel^2)
    #   "absolute":           Phi = ||d - G||^2 / (2 sigma_p^2)   (July runs)
    likelihood: str = "relative"
    # sigma_rel > 0 is used as given; sigma_rel = 0 derives it from the pair
    # (rel_eps, rel_n_eff): sigma_rel = rel_eps / sqrt(rel_n_eff), i.e.
    #   Phi = (rel_n_eff / 2) (relerr / rel_eps)^2,
    # rel_eps = relative error expected at the truth, rel_n_eff = number of
    # independent observations credited.  Only the ratio matters.  Same form
    # as MCMCpar.py / mcmctools (sigma = s, Phi = relerr^2 / (2 s^2)).
    sigma_rel: float = 0.0
    rel_eps: float = 0.02
    rel_n_eff: float = 128.0


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
    seed0: int = 1000
    seed_scheme: str = "spawn"     # see chain_seed(); "legacy" = July runs
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
    seed_scheme: str = "spawn"     # see chain_seed(); "legacy" = first runs
    n_workers: int = 4
    prior_test: bool = False       # likelihood switched off (Phi = 0)
    trace_every: int = 1           # record per-iteration traces every k its
    moves: str = "ABC"             # "AB": ablation without the ridge move C
    exp_tag: str = ""              # run folder e4_sampler_<tag> (keeps runs apart)


def chain_seed(seed0, k, scheme="spawn", legacy_stride=13):
    """Seed for chain / grid pair k.
    "spawn":  numpy SeedSequence([seed0, k]) -- statistically independent
              streams for any number of chains, reproducible from (seed0, k)
              alone, so array job k needs nothing but its index.
    "legacy": seed0 + stride * k (the July scripts: stride 17 for the grid,
              13 for the sampler); fine for a handful of chains."""
    import numpy as np
    if scheme == "spawn":
        return np.random.SeedSequence([seed0, k])
    if scheme == "legacy":
        return seed0 + legacy_stride * k
    raise ValueError(scheme)


# ------------------------- E5: soft-curve sampler (beta free, 4 moves)
@dataclass(frozen=True)
class SoftSamplerConfig:
    """E5 (2026-09-28): beta is a free unknown with the SOFT curve prior of
    report.pdf, Remark 3, instead of E4's hard beta = beta_bar(rho):
        log beta | rho ~ N( log beta_bar(rho),  s(rho)^2 ),
        s(rho)^2 = var_i log(Keff_i(rho)) [E3 scatter] + sigma_e^2.
    (Equivalently log beta = log beta_bar(rho) + s(rho) xi_b, xi_b ~ N(0, 1).)
    Likelihood: pressures only (K* enters once, through the curve prior).
    Moves per iteration (step sizes adapted in burn-in, then frozen):
      A  pCN on theta                                       1 solve
      B  pCN on w, beta fixed                               1 solve
      C  ridge: rho' = rho e^eta, theta' = e^-eta theta,
         beta fixed -> log K unchanged                      0 solves
      D  level swap: log beta' = log beta + delta,
         theta' = theta - (delta / rho) c, c = KLE coords
         of the constant field -> log K nearly unchanged    1 solve
    theta_innovation (move A's pCN innovation, brho/multiscale.py):
      "global"    N(0, I_r)                              plain pCN (default)
      "ms_raw"    local-KLE assemble-and-project, B eta  law N(0, Sigma_ms):
                  WRONG (fails the prior test by design)
      "ms_white"  the same, whitened: W B eta + V0 zeta  law N(0, I_r) exactly
      with ms_blocks x ms_blocks subdomains, local KLEs at ms_local_energy.
      "ms_white" has the same transition kernel as "global" (a correctness
      device, not a speed-up).
      "block"     E8 (brho/blockpcn.py): move A becomes Metropolis-within-
                  Gibbs over theta subspaces -- a coarse move, then one pCN
                  move per local block of a block_n^dim primal decomposition
                  and (block_dual) of the (block_n+1)^dim dual one shifted
                  by half a block.  Local subspace = theta-directions with at
                  least block_energy of their field energy inside the block
                  (widened by block_halo); coarse = complement of the primal
                  locals.  Each move preserves N(0, I_r) exactly; one solve
                  each.  All blocks every iteration (2D, 2x2: 1 + 4 + 9 = 14
                  theta solves).
      "block_random"  the same moves, but each iteration proposes the coarse
                  move and ONE local block chosen uniformly (2 theta solves,
                  4 per iteration in all; equal-cost comparisons).  Every
                  block adapts its own step (init block_s0, target
                  block_target; <= 0 means: use s_init[0] / targets[0])."""
    n_chains: int = 64
    n_total: int = 150000          # same budget as the E4 run (3 solves / it)
    n_burn: int = 30000
    thin: int = 100
    chunk: int = 5000
    mu_rho: float = 0.0
    s_rho: float = 0.75
    s_init: tuple = (0.15, 0.20, 0.20, 0.20)   # (s_theta, s_w, s_ridge, s_swap)
    targets: tuple = (0.25, 0.25, 0.45, 0.45)
    adapt_every: int = 100
    adapt_factor: float = 1.12
    curve_sigma_e: bool = True     # add sigma_e^2 to the curve scatter
    init: str = "prior"
    seed0: int = 505
    seed_scheme: str = "spawn"
    n_workers: int = 4
    prior_test: bool = False
    trace_every: int = 1
    theta_innovation: str = "global"
    ms_blocks: int = 2
    ms_local_energy: float = 0.95
    block_n: int = 2
    block_energy: float = 0.9
    block_dual: bool = True
    block_halo: float = 0.0
    block_s0: float = 0.0
    block_target: float = 0.0
    block_coarse: bool = True      # False: no coarse move (primal + dual must span)
    block_dual_energy: float = 0.0  # <= 0: same as block_energy
    block_dual_halo: float = -1.0   # < 0: same as block_halo
    exp_tag: str = ""               # suffix of the run folder (variants of one innovation)
    learn_noise: bool = False       # sigma_rel unknown: s2 = sigma_rel^2 ~ IG(noise_a0, noise_b0),
                                    # exact Gibbs draw each iteration (no solve); relative
                                    # likelihood only.  noise_b0 <= 0: a0 * sigma_rel^2
                                    # (prior mean of s2 near the configured sigma_rel^2)
    noise_a0: float = 2.0
    noise_b0: float = 0.0
    scale_edges: str = "20,60"     # theta_innovation="scale": KLE-mode cuts (1-based last
                                    # mode of each block but the last); edges >= r dropped
    scale_probs: str = ""          # visit probabilities per scale block ("" = equal)
    collapse_beta: bool = False    # moves A and B propose log beta jointly from its exact
                                    # closed-form conditional (p(beta K) = p(K)/beta): no
                                    # extra solve; MH-corrected, so exact
    theta_repeat: int = 1           # move A k times per iteration (global / ms / scale):
                                    # 2 + k solves; the E8 control (same theta share,
                                    # global moves).  E8 2x2 spends 14 of 16 on theta.


# ------------------------------------ E1: naive global pCN on (theta, beta, rho)
@dataclass(frozen=True)
class NaiveConfig:
    """Joint model log K = log beta + rho G(theta), pressure data only.
    Whitened: u = (theta, xi_b, xi_r) ~ N(0, I_{r+2}),
        log beta = mu_logbeta + s_logbeta xi_b,  log rho = mu_logrho + s_logrho xi_r.
    move = "blocked" (default): per iteration A) pCN on theta, B) pCN on
           (xi_b, xi_r), each with its own adapted step (2 solves / iteration)
    move = "global": ONE pCN move on all of u with a single step (1 solve).
    theta_repeat = k (blocked only): move A k times per iteration, then B once
           (k + 1 solves / iteration); the E1 counterpart of E5's --theta-repeat,
           so both samplers get the same knob in the efficiency comparison."""
    move: str = "blocked"
    theta_repeat: int = 1          # blocked: theta moves per (xi_b, xi_r) move
    exp_tag: str = ""              # run folder suffix: e1_<move>_<tag>
    n_iter: int = 100000
    n_burn: int = 10000            # step-size adaptation window (frozen after)
    thin: int = 100                # theta / p_obs saved every `thin` iterations
    s0: float = 0.10               # initial pCN step (theta block, or global)
    s0_h: float = 0.10             # initial pCN step of the (xi_b, xi_r) block
    target: float = 0.25           # acceptance target during adaptation
    target_h: float = 0.25         # acceptance target of the hyper block
    adapt_every: int = 100
    adapt_factor: float = 1.12
    mu_logbeta: float = 0.0        # beta prior: lognormal, median 1
    s_logbeta: float = 1.0
    mu_logrho: float = 0.0         # rho prior: lognormal, median 1 (as in E4)
    s_logrho: float = 0.75
    init: str = "prior"            # "prior" draws or "zero"
    seed0: int = 2024
    seed_scheme: str = "spawn"
    checkpoint_every: int = 5000   # iterations between checkpoint writes
    log_every: int = 5000          # progress line every k iterations


@dataclass(frozen=True)
class GlamConfig:
    """GLAM (global-local anchored multistage sampling) on top of the PCHIP beta(rho) curve.
    State = 12 subdomain blocks theta_i and w_rho (log rho = mu_logrho + s_logrho w_rho, rho in [rho_min, rho_max]),
    plain pCN (no preconditioning), two step sizes (beta_g global stages, beta_l local stages).
    Window 1 = [0, window1): both betas adapted (Robbins-Monro on the logit scale, pooled over chains), then
    frozen; the tail used for post-processing starts at window1.  Window 2 = [window1, window1 + window2): betas
    frozen, local stages still allowed; local stages stop at window1 + window2 (global-only stages and exchanges
    continue).  Exchanges stop after anchor_stop_iteration (-1 = never)."""
    n_iter: int = 20000
    window1: int = 5000
    window2: int = 5000
    burnin_length: int = 500        # first (global) stage before the first exchange
    global_stage_length: int = 200
    local_stage_length: int = 200
    global_stage_decay: float = 1.0
    local_stage_decay: float = 1.0
    min_global_stage_length: int = 50
    min_local_stage_length: int = 50
    anchor_stop_iteration: int = -1
    beta_g0: float = 0.05
    beta_l0: float = 0.10
    target: float = 0.25
    adapt_interval: int = 100
    mu_logrho: float = 0.0
    s_logrho: float = 0.75
    rho_min: float = 0.05
    rho_max: float = 5.0
    sigma_conc: float = 0.001
    sigma_dp: float = 0.002
    quality_tolerance: float = 20.0
    minimum_anchor_distance: float = 5.0
    restart_scale: float = 0.2
    anchor_current_only: bool = False
    rho_distance_weight: float = 1.0
    max_anchors: int = 0            # archive size cap: 0 = 2 x number of chains (the manager's recommended setting), <0 = unlimited
    save_every: int = 200           # chain records per output part
    log_every: int = 100
    seed0: int = 2026
    # ---- anchor-manager settings (names and defaults of GlobalLocalAnchorManager; a negative value of a setting
    # marked "None" in its comment means None).  Recorded in the manifest like everything else.
    num_cycles: int = 1000000000
    archive_selection_mode: str = "diversity"      # diversity | logdet | farthest
    covariance_regularization: float = 1.0e-6
    pressure_covariance_regularization: float = 1.0e-6
    permeability_logdet_weight: float = 1.0        # (the manager's own default is 0.5)
    pressure_logdet_weight: float = 0.0            # no simulation data in the archive geometry (manager: 0.5)
    normalize_logdet_by_dimension: bool = True
    distance_score_weight: float = 0.02
    quality_score_weight: float = 0.50
    minimum_logdet_gain: float = -1.0              # None
    diversity_alpha: float = 0.70
    logdet_gain_scale: float = 1.0
    minimum_diversity_score: float = 0.20
    archive_selection_temperature: float = 0.10
    exploit_fraction: float = 0.25
    frontier_fraction: float = 0.40
    explore_fraction: float = 0.35
    exploit_restart_scale: float = 0.20
    frontier_restart_scale: float = 0.35
    exploration_restart_scale: float = 0.35
    frontier_target_phi: float = -1.0              # None
    frontier_minimum_distance: float = -1.0        # None
    successful_region_distance: float = -1.0       # None
    successful_region_merge_distance: float = -1.0  # None
    frontier_max_rounds: int = 3
    frontier_minimum_improvement: float = 0.0
    frontier_cooldown_rounds: int = 3
    frontier_relative_quality_tolerance: float = -1.0   # None
    frontier_reseed_interval: int = 0
    frontier_quality_weight: float = 0.60
    frontier_distance_weight: float = 0.40
    preserve_frontier_exact: bool = True
    minimum_anchor_occupancy: int = 1
    anchor_temperature: float = 1.0
    uniform_allocation_weight: float = 0.30
    equilibrium_allocation_enabled: bool = True
    equilibrium_positive_fraction_threshold: float = 0.60
    equilibrium_drift_tolerance: float = 0.0
    equilibrium_min_observations: int = 8
    equilibrium_dispersion_strength: float = 0.75
    equilibrium_distance_bias_max: float = 0.85
    preserve_exact_anchor: bool = True
    max_exact_anchor_copies: int = -1              # None = one exact copy per active anchor
    restart_phi_tolerance: float = 5.0
    max_restart_attempts: int = 3
    restart_scale_decay: float = 0.8
    random_reach_attempts: int = 10
    random_reach_initial_scale: float = -1.0       # None
    random_reach_decay: float = 0.95
    duplicate_tolerance: float = 1.0e-12


# ------------------------------------------------------------ quick budgets
QUICK = {
    GridStudyConfig: dict(n_iter=300, n_burn=150, n_workers=2),
    CurveConfig: dict(n_rho=8, n_real=16),
    SamplerConfig: dict(n_total=1200, n_burn=400, thin=10, chunk=600,
                        n_workers=2),
    SoftSamplerConfig: dict(n_chains=4, n_total=1200, n_burn=400, thin=10,
                            chunk=600, n_workers=2),
    NaiveConfig: dict(n_iter=1500, n_burn=500, thin=10, checkpoint_every=500,
                      log_every=500),
    GlamConfig: dict(n_iter=60, window1=20, window2=20, burnin_length=10, global_stage_length=10,
                     local_stage_length=10, min_global_stage_length=5, min_local_stage_length=5,
                     adapt_interval=5, save_every=20, log_every=10),
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
            dashed = f"--{prefix}{f.name.replace('_', '-')}"
            under = f"--{prefix}{f.name}"
            names = [dashed] + ([under] if under != dashed else [])
            # both spellings work: --sigma-rel and --sigma_rel
            parser.add_argument(*names, dest=f"{prefix}{f.name}", type=typ,
                                default=None)


def from_cli(args, cls, prefix="", quick=False):
    vals = {f.name: getattr(args, f"{prefix}{f.name}", None)
            for f in fields(cls)}
    return build(cls, quick=quick, **vals)


def to_dict(*cfgs):
    return {type(c).__name__: asdict(c) for c in cfgs}


# Settings added after runs already existed: a manifest without the key was
# made under the old behaviour, which these values reproduce.
LEGACY_DEFAULTS = {"ProblemConfig": dict(likelihood="absolute")}


def sigma_rel_from(eps, n_eff):
    """(epsilon, n_eff) -> sigma_rel = epsilon / sqrt(n_eff)."""
    return float(eps) / float(n_eff) ** 0.5


def eps_from(sigma_rel, n_eff):
    """sigma_rel -> epsilon for a chosen n_eff (the pair is not unique)."""
    return float(sigma_rel) * float(n_eff) ** 0.5


def resolve_sigma_rel(pc):
    return pc.sigma_rel if pc.sigma_rel > 0 else sigma_rel_from(pc.rel_eps, pc.rel_n_eff)


def from_dict(cls, d):
    d = dict(d)
    for k, v in LEGACY_DEFAULTS.get(cls.__name__, {}).items():
        d.setdefault(k, v)
    for f in fields(cls):
        if isinstance(d.get(f.name), list):
            d[f.name] = tuple(d[f.name])
    return cls(**{k: v for k, v in d.items()
                  if k in {f.name for f in fields(cls)}})


def _bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def dumps(*cfgs):
    return json.dumps(to_dict(*cfgs), indent=1)
