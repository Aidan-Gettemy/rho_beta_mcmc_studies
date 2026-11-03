"""
Building blocks of the local-KLE (MSM) pCN experiment, docs/msm_local_kle_experiment.md.

Setup (`MSMSetup`): the 32 x 32 unit square, squared-exponential kernel with
ell = 0.1, global KLE with r = 64 modes, the primal 4 x 4 decomposition (4 local
modes per block, p = 64) and the dual 5 x 5 decomposition (interfaces at
(i + 1/2)/4; local dimension of every block from the 95% local-energy rule).
Both are `brho.multiscale.MSInnovation` objects: B (r x p) maps local
coefficients c to global theta, Pi (p x r) maps global theta to local
coefficients, Sigma_ms = B B^T.

Moves (all pCN, step s; every block proposed simultaneously, one step size per grid):
  state c (experiments 2, 3, 5):   c' = sqrt(1 - s^2) c + s eta,  eta ~ N(0, I_p),
                                   theta = B c'   (raw)   or   theta = W B c'  (whitened)
  state theta (experiment 4):      c = Pi_g theta,  c' = sqrt(1 - s^2) c + s eta,
                                   theta' = B_g c'  (grid g chosen with probability 1/2)
Acceptance is on the likelihood ratio only.  With the likelihood off (Phi = 0)
every proposal is accepted, which is the prior test (`prior_chain`).

The implied prior on theta is N(0, Sigma_ms) for the raw state-c chain and
N(0, I_r) for the whitened one (W Sigma_ms W^T = I_r when Sigma_ms has full rank).
"""
import numpy as np

from .kle import build_kle
from .multiscale import MSInnovation, dual_block_index
from .physics2d import Darcy2D

KLE_ENERGY_R64 = 0.966     # tie-aware truncation gives r = 64 (modes 63, 64 tied; 65 separate)
RANK_TOL = 1e-10           # Pereira's tolerance for the effective rank of Sigma_ms


class MSMSetup:
    """KLE, primal and dual decompositions of the experiment."""

    def __init__(self, n=32, ell=0.1, kernel="sqexp", kle_energy=KLE_ENERGY_R64,
                 r_expected=64, blocks=4, primal_local_dim=4, dual_energy=0.95):
        self.n, self.ell, self.blocks = n, ell, blocks
        self.phys = Darcy2D(n, bc="dirichlet")          # grid, cell centers, observation mask
        self.points = self.phys.points
        kp = dict(ell=ell, sigma2=1.0)
        self.kle = build_kle(self.points, self.phys.axes, kernel, kp, energy=kle_energy)
        if r_expected is not None and self.kle.r != r_expected:
            raise ValueError(f"KLE has r = {self.kle.r}, expected {r_expected} "
                             f"(energy {kle_energy})")
        self.r = self.kle.r
        self.primal = MSInnovation(self.kle, self.points, kernel, kp, blocks=blocks,
                                   local_dim=primal_local_dim)
        self.dual = MSInnovation(self.kle, self.points, kernel, kp, blocks=blocks,
                                 local_energy=dual_energy,
                                 block_idx=dual_block_index(self.points, blocks))


# ------------------------------------------------------------ rank test
def rank_report(ms, tol=RANK_TOL):
    """The Sigma_ms rank test for one decomposition (spec Section 6, step 2).
    Sigma_ms = B B^T = V D V^T (descending d); r_plus = #{d_i > tol d_1}.
    Pass: r_plus == r (full rank).  Everything needed to judge it is returned."""
    B = ms.B
    sv = np.linalg.svd(B, compute_uv=False)               # (min(r, p),)
    Sigma = B @ B.T
    d = np.linalg.eigvalsh(Sigma)[::-1]                   # recomputed here, independently of ms
    r_plus = int((d > tol * d[0]).sum())
    out = dict(
        grid_blocks=len(ms.r_k), r=ms.r, p=ms.p, r_k=list(ms.r_k),
        tie_cut_blocks=int(sum(ms.tie_cut)),
        B_shape=list(B.shape), singular_values_B=sv,
        sv_max=float(sv[0]), sv_min=float(sv[-1]), cond_B=float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        eig_Sigma=d, d1=float(d[0]), d_min=float(d[-1]),
        cond_Sigma=float(d[0] / d[-1]) if d[-1] > 0 else float("inf"),
        # rank of B with the singular-value threshold equivalent to tau on Sigma_ms
        # (d_i > tau d_1  <=>  sigma_i > sqrt(tau) sigma_1), so it matches r_plus
        tol=tol, r_plus=r_plus,
        matrix_rank_B=int(np.linalg.matrix_rank(B, tol=np.sqrt(tol) * sv[0])),
        operator_check=float(ms.operator_check),
        trace_over_r=float(d.sum() / ms.r),
        passed=bool(r_plus == ms.r and ms.p >= ms.r))
    return out


# ------------------------------------------------------------ moves
def step_c(c, s, rng):
    """pCN step on local coefficients (any leading shape)."""
    return np.sqrt(1.0 - s * s) * c + s * rng.standard_normal(np.shape(c))


def theta_from_c(ms, c, whiten=False):
    """Global theta (..., r) from local coefficients c (..., p)."""
    if whiten:
        if ms.V0.shape[1]:
            raise ValueError("Sigma_ms is rank-deficient: the whitened map needs the "
                             "null-space term V0 zeta, which is not part of this experiment")
        return np.asarray(c) @ ms.WB.T
    return np.asarray(c) @ ms.B.T


def theta_move_via_grid(ms, theta, s, rng):
    """Experiment-4 move on the global state theta (..., r): project onto the
    local coefficients of grid `ms`, pCN step, project back."""
    c = np.asarray(theta) @ ms.Pi.T
    return step_c(c, s, rng) @ ms.B.T


# ------------------------------------------------------------ prior chains (Phi = 0)
KINDS = ("global", "primal_raw", "primal_white", "mixture", "sequence")


def prior_chain(kind, setup, m, n_iter, thin, burn, s, rng, p_primal=0.5):
    """m independent chains of the experiment's sampler with the likelihood off
    (every proposal accepted), vectorised over chains.  Returns a dict with
    thinned global theta samples after `burn` iterations, shape (m, n, r):
        "theta"                    (all kinds; for "sequence" the two phases are
                                    "theta1", "theta2", n_iter // 2 iterations each)
        "c" (raw/white primal), "c1", "c2" ("sequence": primal / dual coefficients)
        "switch_field_change"      ("sequence": mean relative L2 change of the field
                                    caused by the lossy projection at the phase switch)
    """
    if kind not in KINDS:
        raise ValueError(kind)
    r, kle = setup.r, setup.kle
    P, D = setup.primal, setup.dual
    out = {}

    def record(i, keep_t, keep_c, th, c=None):
        if i >= burn and (i - burn) % thin == 0:
            keep_t.append(th.copy())
            if c is not None:
                keep_c.append(c.copy())

    if kind == "global":
        th = rng.standard_normal((m, r))
        kt = []
        for i in range(n_iter):
            th = step_c(th, s, rng)
            record(i, kt, [], th)
        out["theta"] = np.stack(kt, 1)
    elif kind in ("primal_raw", "primal_white"):
        white = kind == "primal_white"
        c = rng.standard_normal((m, P.p))                     # exact stationary start of c
        kt, kc = [], []
        for i in range(n_iter):
            c = step_c(c, s, rng)
            if i >= burn and (i - burn) % thin == 0:
                kt.append(theta_from_c(P, c, white)); kc.append(c.copy())
        out["theta"], out["c"] = np.stack(kt, 1), np.stack(kc, 1)
    elif kind == "mixture":
        th = rng.standard_normal((m, r))
        kt = []
        for i in range(n_iter):
            use_p = rng.random(m) < p_primal
            th_p = theta_move_via_grid(P, th, s, rng)
            th_d = theta_move_via_grid(D, th, s, rng)
            th = np.where(use_p[:, None], th_p, th_d)
            record(i, kt, [], th)
        out["theta"] = np.stack(kt, 1)
    else:                                                      # "sequence"
        n1 = n_iter // 2
        c = rng.standard_normal((m, P.p))
        kt1, kc1, kt2, kc2 = [], [], [], []
        for i in range(n1):
            c = step_c(c, s, rng)
            if i >= burn and (i - burn) % thin == 0:
                kt1.append(c @ P.B.T); kc1.append(c.copy())
        th_end = c @ P.B.T
        c2 = th_end @ D.Pi.T                                   # project onto the dual basis
        T = kle.T                                              # (N, r); field = theta @ T.T
        f_before, f_after = th_end @ T.T, (c2 @ D.B.T) @ T.T
        out["switch_field_change"] = float(np.mean(
            np.linalg.norm(f_after - f_before, axis=1) / np.linalg.norm(f_before, axis=1)))
        c = c2
        for i in range(n_iter - n1):
            c = step_c(c, s, rng)
            if i >= burn and (i - burn) % thin == 0:
                kt2.append(c @ D.B.T); kc2.append(c.copy())
        out["theta1"], out["c1"] = np.stack(kt1, 1), np.stack(kc1, 1)
        out["theta2"], out["c2"] = np.stack(kt2, 1), np.stack(kc2, 1)
    return out
