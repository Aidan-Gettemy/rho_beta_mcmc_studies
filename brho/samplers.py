"""
MCMC kernels.

E2  pcn_fixed_pair      pCN on theta with (beta, rho) held fixed
                        (one chain per grid pair; mcmc_study.py logic).

E4  ThreeStepSampler    unknowns (theta, w), log rho = mu_rho + s_rho w,
                        beta eliminated by the offline curve beta_bar(rho):
                            log K = log beta_bar(rho) + rho * G(theta)
    one iteration = A, B, C  (hierarchical-model report, Sec. 5, with the
    hyperparameter block reduced to w as in the method report, Sec. 4):
      A  pCN on theta                         1 solve, accept e^{Phi-Phi'}
      B  pCN on w                             1 solve, accept e^{Phi-Phi'}
      C  ridge move  rho' = rho e^eta,  theta' = e^{-eta} theta
         rho*theta is preserved, but beta_bar(rho') != beta_bar(rho), so the
         field level changes and Phi MUST be re-evaluated (1 solve) -- unlike
         the hierarchical model, where C is free.  Acceptance (method eq. 11):
         log a = Phi - Phi' - (|theta'|^2 - |theta|^2)/2 - (w'^2 - w^2)/2 - r eta
    Step sizes adapt every `adapt_every` iterations during burn-in only, then
    freeze.  Chains are checkpointed in chunks and resumable.  Chain k is
    seeded by config.chain_seed(seed0, k, seed_scheme), so array job k can
    start (or restart) its chain knowing only its index.
"""
import json
import os
import numpy as np

from .config import chain_seed


def trace_burn_index(n_burn, trace_every):
    """Index of the first recorded trace entry at or after iteration n_burn
    (traces are recorded at iterations 0, k, 2k, ...)."""
    return -(-n_burn // trace_every)


# ---------------------------------------------------------------- adaptation
class StepAdapter:
    """Multiplicative step-size adaptation toward acceptance targets."""

    def __init__(self, sizes, targets=None, band=None, every=100,
                 factor=1.12, s_min=1e-3, s_max=0.95):
        self.s = np.array(sizes, float)
        self.targets, self.band = targets, band
        self.every, self.factor = every, factor
        self.s_min, self.s_max = s_min, s_max
        self.acc = np.zeros(len(self.s))
        self.win = 0

    def record(self, j, accepted):
        self.acc[j] += bool(accepted)

    def tick(self, it, n_burn):
        self.win += 1
        if self.win < self.every:
            return
        if it < n_burn:
            rate = self.acc / self.win
            for j in range(len(self.s)):
                if self.band is not None:           # stay inside [lo, hi]
                    lo, hi = self.band
                    if rate[j] > hi:
                        self.s[j] *= self.factor
                    elif rate[j] < lo:
                        self.s[j] /= self.factor
                else:                                # steer toward target
                    self.s[j] = (self.s[j] * self.factor if rate[j] > self.targets[j]
                                 else self.s[j] / self.factor)
            self.s = np.clip(self.s, self.s_min, self.s_max)
        self.acc[:] = 0
        self.win = 0


def pcn(x, s, rng):
    return np.sqrt(1.0 - s * s) * x + s * rng.standard_normal(np.shape(x))


# ================================================================ E2 kernel
def pcn_fixed_pair(prob, beta, rho, gcfg, seed):
    """pCN in theta-space for fixed (beta, rho).  Returns summary dict."""
    rng = np.random.default_rng(seed)
    kle, r = prob.kle, prob.r
    logK = lambda th: np.log(beta) + rho * kle.field(th)

    theta = np.zeros(r)
    phi, pobs, _ = prob.potential(np.exp(logK(theta)))
    ad = StepAdapter([gcfg.s0], band=(gcfg.acc_lo, gcfg.acc_hi),
                     every=gcfg.adapt_every, factor=gcfg.adapt_factor,
                     s_min=0.005, s_max=0.9)
    trace = np.empty(gcfg.n_iter)
    n_acc, best = 0, phi
    sum_pobs = np.zeros_like(pobs)
    sum_G = np.zeros(prob.phys.N)
    sum_th, sum_th2 = np.zeros(r), np.zeros(r)
    n_post = 0
    for it in range(gcfg.n_iter):
        prop = pcn(theta, ad.s[0], rng)
        phi_p, pobs_p, _ = prob.potential(np.exp(logK(prop)))
        acc = np.log(rng.random()) < phi - phi_p
        if acc:
            theta, phi, pobs = prop, phi_p, pobs_p
            n_acc += 1
            best = min(best, phi)
        ad.record(0, acc)
        ad.tick(it, gcfg.n_burn)
        trace[it] = phi
        if it >= gcfg.n_burn:
            sum_pobs += pobs
            sum_G += kle.field(theta)
            sum_th += theta
            sum_th2 += theta ** 2
            n_post += 1
    mean_pobs = sum_pobs / n_post
    mean_th = sum_th / n_post
    return dict(beta=beta, rho=rho,
                rms=float(np.sqrt(np.mean((prob.syn.d_p - mean_pobs) ** 2))),
                trace=trace, mean_pobs=mean_pobs, mean_G=sum_G / n_post,
                theta_norm2_mean=float(sum_th2.sum() / n_post),
                mean_theta=mean_th, acc=n_acc / gcfg.n_iter,
                best_phi=float(best), mean_phi=float(trace[gcfg.n_burn:].mean()),
                s_final=float(ad.s[0]))


# ================================================================ E4 kernel
class ThreeStepSampler:
    N_TRACK = 5                      # leading theta modes traced every iteration

    def __init__(self, prob, curve, scfg):
        self.prob, self.curve, self.cfg = prob, curve, scfg
        self.r = prob.r
        self.Kstar = prob.syn.keff_obs

    # ------------------------------------------------------------- model
    def rho(self, w):
        return float(np.exp(self.cfg.mu_rho + self.cfg.s_rho * w))

    def beta(self, rho):
        return float(self.curve.beta_bar(rho, self.Kstar))

    def logK(self, theta, w):
        rho = self.rho(w)
        return np.log(self.beta(rho)) + rho * self.prob.kle.field(theta)

    def level(self, theta, w):
        rho = self.rho(w)
        return np.log(self.beta(rho)) + rho * self.prob.kle.spatial_mean(theta)

    def potential(self, theta, w):
        """(Phi, p_obs, Keff); Phi = inf outside the curve's rho range."""
        rho = self.rho(w)
        b = self.beta(rho)
        if not np.isfinite(b):
            return np.inf, None, np.nan
        if self.cfg.prior_test:
            return 0.0, None, np.nan
        K = b * np.exp(rho * self.prob.kle.field(theta))
        return self.prob.potential(K, self.cfg.use_keff_term)

    # ------------------------------------------------------------- state
    def init_state(self, rng):
        if self.cfg.init == "prior":
            theta, w = rng.standard_normal(self.r), float(rng.standard_normal())
        else:
            theta, w = np.zeros(self.r), 0.0
        phi, pobs, keff = self.potential(theta, w)
        return dict(theta=theta, w=w, phi=phi, pobs=pobs, keff=keff)

    # ------------------------------------------------------------ one sweep
    def step(self, st, ad, rng):
        s_th, s_w, s_rg = ad.s
        r = self.r
        # A: pCN on theta (w fixed)
        th_p = pcn(st["theta"], s_th, rng)
        phi_p, pobs_p, keff_p = self.potential(th_p, st["w"])
        a = np.log(rng.random()) < st["phi"] - phi_p
        if a:
            st.update(theta=th_p, phi=phi_p, pobs=pobs_p, keff=keff_p)
        ad.record(0, a)
        # B: pCN on w (theta fixed)
        w_p = float(pcn(st["w"], s_w, rng))
        phi_p, pobs_p, keff_p = self.potential(st["theta"], w_p)
        a = np.log(rng.random()) < st["phi"] - phi_p
        if a:
            st.update(w=w_p, phi=phi_p, pobs=pobs_p, keff=keff_p)
        ad.record(1, a)
        # C: ridge move, rho*theta preserved; beta_bar(rho) changes -> solve
        eta = s_rg * rng.standard_normal()
        th_p = np.exp(-eta) * st["theta"]
        w_p = st["w"] + eta / self.cfg.s_rho
        phi_p, pobs_p, keff_p = self.potential(th_p, w_p)
        logA = (st["phi"] - phi_p
                - 0.5 * (th_p @ th_p - st["theta"] @ st["theta"])
                - 0.5 * (w_p ** 2 - st["w"] ** 2) - r * eta)
        a = np.log(rng.random()) < logA
        if a:
            st.update(theta=th_p, w=w_p, phi=phi_p, pobs=pobs_p, keff=keff_p)
        ad.record(2, a)
        return st


# ------------------------------------------------------ checkpointed chains
TRACE_KEYS = ("rho", "beta", "phi", "level", "w", "theta_norm2")
KEPT_KEYS = ("theta", "rho", "beta", "pobs", "keff")


def run_chunk(sampler, chain_id, ckpt_path):
    """Advance one chain by cfg.chunk iterations (or to n_total) from its
    checkpoint; returns the new iteration count."""
    cfg = sampler.cfg
    if os.path.exists(ckpt_path):
        ck = dict(np.load(ckpt_path, allow_pickle=False))
        it0 = int(ck["it"])
        if it0 >= cfg.n_total:
            return it0
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(ck["rng_state"]))
        st = dict(theta=ck["theta_cur"], w=float(ck["w_cur"]))
        st["phi"], st["pobs"], st["keff"] = sampler.potential(st["theta"], st["w"])
        s = ck["s"]
        acc_tot = ck["acc_tot"]
        traces = {k: list(ck["tr_" + k]) for k in TRACE_KEYS}
        traces["theta_lead"] = list(ck["tr_theta_lead"])
        kept = {k: list(ck["kept_" + k]) for k in KEPT_KEYS}
        s_hist = list(ck["s_hist"])
    else:
        it0 = 0
        rng = np.random.default_rng(chain_seed(cfg.seed0, chain_id,
                                               cfg.seed_scheme, 13))
        st = sampler.init_state(rng)
        s = np.array(cfg.s_init, float)
        acc_tot = np.zeros(3)
        traces = {k: [] for k in TRACE_KEYS}
        traces["theta_lead"] = []
        kept = {k: [] for k in KEPT_KEYS}
        s_hist = []
    ad = StepAdapter(s, targets=cfg.targets, every=cfg.adapt_every,
                     factor=cfg.adapt_factor)
    n_steps = min(cfg.chunk, cfg.n_total - it0)
    k = ThreeStepSampler.N_TRACK
    for loc in range(n_steps):
        it = it0 + loc
        before = ad.acc.copy()
        sampler.step(st, ad, rng)
        acc_tot += ad.acc - before
        ad.tick(it, cfg.n_burn)
        if it < cfg.n_burn and (it + 1) % cfg.adapt_every == 0:
            s_hist.append(ad.s.copy())
        rho = sampler.rho(st["w"])
        if it % cfg.trace_every == 0:
            traces["rho"].append(rho)
            traces["beta"].append(sampler.beta(rho))
            traces["phi"].append(st["phi"])
            traces["level"].append(sampler.level(st["theta"], st["w"]))
            traces["w"].append(st["w"])
            traces["theta_norm2"].append(float(st["theta"] @ st["theta"]))
            traces["theta_lead"].append(st["theta"][:k].copy())
        if it >= cfg.n_burn and (it - cfg.n_burn) % cfg.thin == 0:
            kept["theta"].append(st["theta"].copy())
            kept["rho"].append(rho)
            kept["beta"].append(sampler.beta(rho))
            kept["pobs"].append(st["pobs"] if st["pobs"] is not None
                                else np.full(sampler.prob.syn.n_obs, np.nan))
            kept["keff"].append(st["keff"])
    it_new = it0 + n_steps
    out = dict(it=it_new, s=ad.s, acc_tot=acc_tot, theta_cur=st["theta"],
               w_cur=st["w"], rng_state=json.dumps(rng.bit_generator.state),
               s_hist=np.array(s_hist).reshape(-1, 3))
    for key, v in traces.items():
        out["tr_" + key] = np.array(v)
    for key, v in kept.items():
        out["kept_" + key] = np.array(v)
    tmp = ckpt_path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, ckpt_path)
    return it_new


def load_chains(paths):
    """Load completed chain checkpoints into a dict of stacked arrays."""
    zs = [np.load(p) for p in paths]
    out = {}
    for key in zs[0].files:
        if key.startswith(("tr_", "kept_", "acc_tot", "s_hist")):
            arrs = [z[key] for z in zs]
            L = min(len(a) for a in arrs)
            out[key] = np.stack([a[:L] for a in arrs])
    out["it"] = np.array([int(z["it"]) for z in zs])
    out["s_final"] = np.stack([z["s"] for z in zs])
    return out
