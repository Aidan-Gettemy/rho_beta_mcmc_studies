"""
Production sampler of the local-KLE (MSM) experiment, docs/msm_local_kle_experiment.md.
One chain per call (`run_chain`); the six samplers are the `kind`s of brho.msm_local.KINDS
that carry a likelihood:
    global        state theta                       (experiment 1)
    primal_raw    state c (primal), theta = B c     (2)
    primal_pinv   state c, theta = W_+ B c          (3a)
    primal_aug    state (c, zeta), theta = W_+ B c + V0 zeta   (3b)
    mixture       state theta; w.p. 1/2 each: c = Pi_g theta, pCN, theta' = B_g c'   (4)
    sequence      primal_raw for K iterations, then c = Pi_dual theta_final and
                  dual raw for n_iter - K iterations                                 (5)
Every proposal: pCN with ONE step size per grid, all blocks together; acceptance on the
likelihood ratio only; one forward solve per iteration.  Step sizes adapt toward 25% for
the first M iterations of each phase (fresh adapter at the phase switch), then freeze.

New (2026-11-03, docs/msm_experiment_plan_2026-11-03.md); state = global theta, field-norm least-squares
lifts Lp, Ld (brho.msm_local.field_lift; c = L theta):
    alt            L iterations on the primal grid, L on the dual, alternating.  Move on grid g with
                   c0 = L_g theta, c' = sqrt(1-s^2) c0 + s xi:  theta' = theta + B_g (c' - c0)  (the part of
                   theta the grid cannot represent is an untouched offset: s = 0 gives theta' = theta
                   exactly).  alt_offset=False (6d): primal theta' = B_p c' (lift only, lossy).
    mixture_inn    random grid per iteration, theta' = sqrt(1-s^2) theta + s B_g xi (never projects the state)
    sequence_exact as `sequence`, but the switch is the exact dual lift c = Ld theta (B_d c = theta)
`mixture` keeps the original Pi lift (experiment 4, also with 8 modes per block).

Data (`make_data`): theta_true ~ N(0, I_r) with a fixed seed, forward solve with the
Dirichlet permeameter, observation on the checkerboard cells, additive Gaussian noise
with sigma_p = data_relerr ||p_obs|| / sqrt(n_obs) (default data_relerr = sigma_rel), so that the
expected GLOBAL relative error at the truth is data_relerr and the expected Phi at the truth is
0.5 (data_relerr / sigma_rel)^2.  NOTE: with the global-norm likelihood this floor is ~0.5, not
n_obs/2; the repo's own runs temper it through rel_n_eff (sigma_rel = rel_eps / sqrt(rel_n_eff)).
Likelihood: Phi = ||d - O p||^2 / (2 sigma_rel^2 ||d||^2).
"""
import numpy as np

from .samplers import StepAdapter
from . import msm_local as ML

CHAIN_KINDS = ("global", "primal_raw", "primal_pinv", "primal_aug", "mixture", "sequence",
               "alt", "mixture_inn", "sequence_exact")
TWO_GRID = ("mixture", "alt", "mixture_inn")      # kinds with one step size per grid
SEQ = ("sequence", "sequence_exact")


class Data:
    def __init__(self, S, sigma_rel=0.003, seed=2026, theta_true=None, data_relerr=None):
        self.S, self.sigma_rel, self.seed = S, sigma_rel, seed
        self.data_relerr = sigma_rel if data_relerr is None else data_relerr
        rng = np.random.default_rng(seed)
        self.theta_true = rng.standard_normal(S.r) if theta_true is None else theta_true
        K = np.exp(S.kle.field(self.theta_true))
        _, pobs, _ = S.phys.forward(K)
        self.n_obs = int(pobs.size)
        self.sigma_p = self.data_relerr * np.linalg.norm(pobs) / np.sqrt(self.n_obs)
        self.d = pobs + self.sigma_p * rng.standard_normal(self.n_obs)
        self.dnorm = float(np.linalg.norm(self.d))
        self.w = 1.0 / (sigma_rel ** 2 * self.dnorm ** 2)
        self.relerr_truth = float(np.linalg.norm(self.d - pobs) / self.dnorm)
        self.phi_truth = 0.5 * self.w * float(np.sum((self.d - pobs) ** 2))
        self.noise_floor = 0.5 * (self.data_relerr / sigma_rel) ** 2   # expectation; realised value is phi_truth
        self.obs_clean = pobs

    def potential(self, theta):
        """(Phi, relative error) of a global theta."""
        K = np.exp(self.S.kle.field(theta))
        _, pobs, _ = self.S.phys.forward(K)
        r2 = float(np.sum((self.d - pobs) ** 2))
        return 0.5 * self.w * r2, np.sqrt(r2) / self.dnorm

    def summary(self):
        return dict(seed=self.seed, sigma_rel=self.sigma_rel, data_relerr=self.data_relerr, n_obs=self.n_obs,
                    sigma_p=float(self.sigma_p), data_norm=self.dnorm,
                    relerr_at_truth=self.relerr_truth, phi_at_truth=self.phi_truth,
                    phi_noise_floor_expected=self.noise_floor)


def _pcn(x, s, rng):
    return np.sqrt(1.0 - s * s) * x + s * rng.standard_normal(np.shape(x))


def run_chain(kind, S, data, seed0, chain, n_iter, n_adapt, k_phase=None, s0=0.2,
              save_every=100, target=0.25, alt_L=1, alt_offset=True, init_scale=1.0):
    """One chain.  Returns a dict of arrays (see docstring of the module for the state).
    relerr (n_iter,) state's relative error after every iteration; acc (n_iter,) accepted
    flags; move (n_iter,) grid used (0 primal, 1 dual; all 0 for the single-grid kinds);
    theta (n_iter // save_every, r) global theta at iterations 0, 100, ... (before the step);
    c / c1 / c2 / zeta local states at the same iterations; s_hist (n_iter//save_every, ngrid);
    phi0 / phi1 (sequence) Phi just before / after the phase switch."""
    if kind not in CHAIN_KINDS:
        raise ValueError(kind)
    P, D, r = S.primal, S.dual, S.r
    rng = np.random.default_rng(np.random.SeedSequence([seed0, chain]))
    K = k_phase if kind in SEQ else None
    if kind in SEQ and (K is None or 2 * K != n_iter):
        raise ValueError("sequence needs k_phase with n_iter == 2 * k_phase")
    n_g = 2 if kind in TWO_GRID else 1
    if kind in ("alt", "mixture_inn", "sequence_exact"):
        Lp, Ld = ML.field_lift(P, S.kle.T), ML.field_lift(D, S.kle.T)
    ad = StepAdapter([s0] * n_g, targets=[target] * n_g, every=100)
    # state and the map state -> theta
    th0 = init_scale * rng.standard_normal(r)    # same first draw for every kind (paired seeds); init_scale 1 = prior draw
    c = z = None
    ms = P
    if kind == "global" or kind in TWO_GRID:
        theta = th0
    else:
        c = rng.standard_normal(P.p)
        if kind == "primal_aug":
            z = rng.standard_normal(P.V0.shape[1])
        theta = _theta(kind, ms, c, z)
    phi, rel = data.potential(theta)
    n_save = n_iter // save_every
    out = dict(relerr=np.empty(n_iter, np.float32), acc=np.zeros(n_iter, np.uint8),
               move=np.zeros(n_iter, np.uint8), theta=np.empty((n_save, r), np.float32),
               s_hist=np.empty((n_save, n_g), np.float32), phi=np.empty(n_iter, np.float32))
    if kind in ("primal_raw", "primal_pinv", "primal_aug"):
        out["c"] = np.empty((n_save, P.p), np.float32)
    if kind == "primal_aug":
        out["zeta"] = np.empty((n_save, z.size), np.float32)
    if kind in TWO_GRID:
        out["c_primal"] = np.empty((n_save, P.p), np.float32)
        out["c_dual"] = np.empty((n_save, D.p), np.float32)
    if kind in SEQ:
        out["c1"] = np.empty((n_save // 2, P.p), np.float32)
        out["c2"] = np.empty((n_save // 2, D.p), np.float32)
    for it in range(n_iter):
        if kind in SEQ and it == K:
            out["phi0"], out["rel0"] = phi, rel
            ms = D
            if kind == "sequence_exact":
                c = Ld @ theta                     # exact lift: B_d c = theta, the field is unchanged
                theta = D.B @ c
            else:
                c = D.Pi @ theta                   # project the final global field onto the dual basis
                theta = D.B @ c
            phi, rel = data.potential(theta)       # (sequence) the lossy projection changes the field
            out["phi1"], out["rel1"] = phi, rel
            ad = StepAdapter([s0], targets=[target], every=100)
        k_it = it - K if (kind in SEQ and it >= K) else it
        if it % save_every == 0:
            j = it // save_every
            out["theta"][j] = theta
            out["s_hist"][j] = ad.s
            if kind in ("primal_raw", "primal_pinv", "primal_aug"):
                out["c"][j] = c
            if kind == "primal_aug":
                out["zeta"][j] = z
            if kind in ("mixture",):
                out["c_primal"][j], out["c_dual"][j] = P.Pi @ theta, D.Pi @ theta
            if kind in ("alt", "mixture_inn"):
                out["c_primal"][j], out["c_dual"][j] = Lp @ theta, Ld @ theta
            if kind in SEQ:
                (out["c1"] if it < K else out["c2"])[(j if it < K else j - K // save_every)] = c
        # --- proposal
        if kind == "global":
            th_new = _pcn(theta, ad.s[0], rng); g = 0
        elif kind == "mixture":
            g = 0 if rng.random() < 0.5 else 1
            m = P if g == 0 else D
            th_new = m.B @ _pcn(m.Pi @ theta, ad.s[g], rng)
        elif kind == "mixture_inn":
            g = 0 if rng.random() < 0.5 else 1
            m = P if g == 0 else D
            sg = ad.s[g]
            th_new = np.sqrt(1.0 - sg * sg) * theta + sg * (m.B @ rng.standard_normal(m.p))
        elif kind == "alt":
            g = (it // alt_L) % 2
            m, Lg = (P, Lp) if g == 0 else (D, Ld)
            c0 = Lg @ theta
            c_new = _pcn(c0, ad.s[g], rng)
            th_new = (m.B @ c_new) if (g == 0 and not alt_offset) else theta + m.B @ (c_new - c0)
        else:
            g = 0
            c_new = _pcn(c, ad.s[0], rng)
            z_new = _pcn(z, ad.s[0], rng) if z is not None else None
            th_new = _theta(kind, ms, c_new, z_new)
        phi_new, rel_new = data.potential(th_new)
        accepted = np.log(rng.random()) < phi - phi_new
        if accepted:
            theta, phi, rel = th_new, phi_new, rel_new
            if kind not in ("global",) + TWO_GRID:
                c, z = c_new, z_new
        ad.record(g, accepted)
        ad.tick(k_it, n_adapt)
        out["relerr"][it], out["acc"][it], out["move"][it], out["phi"][it] = rel, accepted, g, phi
    return out


def _theta(kind, ms, c, z):
    if kind in ("primal_raw",) + SEQ:
        return ML.theta_from_c(ms, c)
    if kind == "primal_pinv":
        return ML.theta_from_c(ms, c, True, pinv=True)
    return ML.theta_from_c(ms, c, True, zeta=z)
