"""
MCMC kernels.

E2  pcn_fixed_pair      pCN on theta with (beta, rho) held fixed
                        (one chain per grid pair; mcmc_study.py logic).

E5  SoftCurveSampler    beta free with the soft curve prior; moves A-D
                        (see the class docstring and config.SoftSamplerConfig)
E7  ... theta_innovation="ms_white": whitened multiscale innovation in A
E8  ... theta_innovation="block": A = block (subdomain) pCN sweep, coarse +
                        primal + dual local subspaces (brho/blockpcn.py), one
                        adapted step per block

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
    """Multiplicative step-size adaptation toward acceptance targets.
    Acceptance rates are per PROPOSAL of each move (a move not proposed in a
    window, e.g. a block skipped by the random block schedule, is left
    alone).  When every move is proposed once per iteration -- all samplers
    before E8 -- this is exactly the old rate acc / iterations."""

    def __init__(self, sizes, targets=None, band=None, every=100,
                 factor=1.12, s_min=1e-3, s_max=0.95):
        self.s = np.array(sizes, float)
        self.targets, self.band = targets, band
        self.every, self.factor = every, factor
        self.s_min, self.s_max = s_min, s_max
        self.acc = np.zeros(len(self.s))
        self.n = np.zeros(len(self.s))
        self.win = 0

    def record(self, j, accepted):
        self.acc[j] += bool(accepted)
        self.n[j] += 1

    def tick(self, it, n_burn):
        self.win += 1
        if self.win < self.every:
            return
        if it < n_burn:
            rate = self.acc / np.maximum(self.n, 1)
            for j in range(len(self.s)):
                if self.n[j] == 0:
                    continue
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
        self.n[:] = 0
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
    STATE = ("w",)                   # checkpointed state besides theta

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

    # ---------------------------------------------- run_chunk interface
    def potential_st(self, st):
        return self.potential(st["theta"], st["w"])

    def beta_st(self, st):
        return self.beta(self.rho(st["w"]))

    def level_st(self, st):
        return self.level(st["theta"], st["w"])

    def extra_traces(self, st):
        return {}

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


# ================================================================ E5 kernel
class SoftCurveSampler:
    """E5: beta free with the soft curve prior (report.pdf, Remark 3).

    Unknowns (theta, w, lb = log beta), prior
        theta ~ N(0, I_r),  w ~ N(0, 1) [rho = exp(mu_rho + s_rho w) <= rho_max],
        lb | rho ~ N(m(rho), s(rho)^2),  m = log beta_bar(rho; K*),
        s^2 = (E3 scatter of log Keff_i)^2 + sigma_e^2.
    Likelihood: pressures only; K* enters once, through m(rho).
    One iteration = A, B, C, D (see config.SoftSamplerConfig):
      A pCN on theta                                     accept e^{Phi - Phi'}
      B pCN on w, beta fixed                             1 solve,
        log a = Phi - Phi' + log p(lb|rho') - log p(lb|rho)
        (moving w at fixed xi_b would drag the level along with beta_bar(rho)
        and force tiny steps; with beta fixed only rho * Gbar moves)
      C ridge: rho' = rho e^eta, theta' = e^-eta theta, lb fixed.  log K is
        EXACTLY unchanged (no solve); only priors + Jacobian e^{-r eta}:
        log a = -(|th'|^2-|th|^2)/2 - (w'^2-w^2)/2 + log p(lb|rho') - log p(lb|rho) - r eta
      D level swap: lb' = lb + delta, theta' = theta - (delta/rho) c, where
        T c ~ 1 (least squares; the constant field is in the KLE span to a few
        percent), so log K moves only by delta (1 - Tc): one solve,
        log a = Phi - Phi' - (|th'|^2-|th|^2)/2 + log p(lb'|rho) - log p(lb|rho)
      C and D are deterministic maps given eta / delta with symmetric
      N(0, s^2) draws, so the ratios above are the full MH ratios.
    Solves per iteration: A 1, B 1, C 0, D 1 = 3.  (A pure level move --
    random walk on log beta alone -- was tried and dropped: the data pin the
    level to ~0.006 given theta, so it adapts to tiny steps and does not help;
    the level's marginal spread comes through theta.)"""
    N_TRACK = 5
    STATE = ("w", "lb")

    def __init__(self, prob, curve, scfg):
        self.prob, self.curve, self.cfg = prob, curve, scfg
        self.r = prob.r
        self.Kstar = prob.syn.keff_obs
        self.sig_e2 = prob.syn.sigma_e ** 2 if scfg.curve_sigma_e else 0.0
        T = prob.kle.T
        ones = np.ones(T.shape[0])
        self.c, *_ = np.linalg.lstsq(T, ones, rcond=None)
        self.c_resid = float(np.linalg.norm(T @ self.c - ones) / np.sqrt(len(ones)))
        self.innovation = getattr(scfg, "theta_innovation", "global")
        self.ms = self.bp = None
        self.s_init, self.targets = tuple(scfg.s_init), tuple(scfg.targets)
        self.move_names = ("A", "B", "C", "D")
        self.solves_per_iteration = 3
        if self.innovation in ("block", "block_random"):
            # E8: move A becomes block (subdomain) pCN, brho/blockpcn.py.  Step
            # sizes: index 0 = coarse move (A's slot), 1-3 = B, C, D, then one
            # per local block (primal P1.., dual Q1..), each adapted on its own.
            from .blockpcn import from_problem as block_from_problem
            self.bp = block_from_problem(
                prob, scfg.block_n, scfg.block_energy, scfg.block_dual, scfg.block_halo,
                getattr(scfg, "block_coarse", True),
                scfg.block_dual_energy if getattr(scfg, "block_dual_energy", 0) > 0 else None,
                scfg.block_dual_halo if getattr(scfg, "block_dual_halo", -1) >= 0 else None)
            self.schedule = "random" if self.innovation == "block_random" else "sweep"
            nl = self.bp.n_local
            s_loc = scfg.block_s0 if scfg.block_s0 > 0 else scfg.s_init[0]
            t_loc = scfg.block_target if scfg.block_target > 0 else scfg.targets[0]
            self.s_init += (s_loc,) * nl
            self.targets += (t_loc,) * nl
            self.move_names = ("A0 coarse" if self.bp.coarse else "A0 (off)",
                               "B", "C", "D") + tuple(
                mv["name"] for mv in self.bp.moves[1:])
            self.solves_per_iteration = self.bp.solves(self.schedule) + 2
        elif self.innovation != "global":
            from .multiscale import from_problem
            self.ms = from_problem(prob, scfg.ms_blocks, scfg.ms_local_energy)

    def move_index(self, j):
        """Step / acceptance slot of block move j (0 = coarse)."""
        return 0 if j == 0 else 3 + j

    # ------------------------------------------------------------- model
    def rho(self, w):
        return float(np.exp(self.cfg.mu_rho + self.cfg.s_rho * w))

    def m(self, rho):
        return float(np.log(self.curve.beta_bar(rho, self.Kstar)))

    def s(self, rho):
        return float(np.sqrt(self.curve.log_scatter(rho) ** 2 + self.sig_e2))

    def log_p_lb(self, lb, rho):
        m, s = self.m(rho), self.s(rho)
        if not (np.isfinite(m) and np.isfinite(s)) or s <= 0:
            return -np.inf
        return -0.5 * ((lb - m) / s) ** 2 - np.log(s)

    def potential(self, theta, w, lb):
        rho = self.rho(w)
        if not np.isfinite(self.m(rho)):
            return np.inf, None, np.nan
        if self.cfg.prior_test:
            return 0.0, None, np.nan
        return self.prob.potential(np.exp(lb + rho * self.prob.kle.field(theta)))

    # ---------------------------------------------- run_chunk interface
    def potential_st(self, st):
        return self.potential(st["theta"], st["w"], st["lb"])

    def beta_st(self, st):
        return float(np.exp(st["lb"]))

    def level_st(self, st):
        return float(st["lb"] + self.rho(st["w"]) * self.prob.kle.spatial_mean(st["theta"]))

    def extra_traces(self, st):
        if st is None:
            return {"xi_b": None}
        rho = self.rho(st["w"])
        return {"xi_b": (st["lb"] - self.m(rho)) / self.s(rho)}

    def init_state(self, rng):
        if self.cfg.init == "prior":
            theta = rng.standard_normal(self.r)
            while True:                                  # w truncated at rho_max
                w = float(rng.standard_normal())
                if np.isfinite(self.m(self.rho(w))):
                    break
            rho = self.rho(w)
            lb = self.m(rho) + self.s(rho) * float(rng.standard_normal())
        else:
            theta, w = np.zeros(self.r), 0.0
            lb = self.m(self.rho(w))
        phi, pobs, keff = self.potential(theta, w, lb)
        return dict(theta=theta, w=w, lb=lb, phi=phi, pobs=pobs, keff=keff)

    # ------------------------------------------------------------ one sweep
    def step(self, st, ad, rng):
        s_th, s_h, s_rg, s_lv = ad.s[:4]
        th, w, lb = st["theta"], st["w"], st["lb"]
        rho = self.rho(w)
        # A: pCN on theta (global innovation: unchanged code path, so existing
        #    runs resume bit-identically; multiscale: brho/multiscale.py)
        if self.bp is not None:
            # E8: coarse + local blocks, Metropolis-within-Gibbs (blockpcn.py)
            for j in self.bp.order(self.schedule, rng):
                k = self.move_index(j)
                th_p = self.bp.propose(st["theta"], j, ad.s[k], rng)
                if th_p is None:
                    continue
                phi_p, pobs_p, keff_p = self.potential(th_p, w, lb)
                a = np.log(rng.random()) < st["phi"] - phi_p
                if a:
                    st.update(theta=th_p, phi=phi_p, pobs=pobs_p, keff=keff_p)
                ad.record(k, a)
        else:
            if self.ms is None:
                th_p = pcn(th, s_th, rng)
            else:
                th_p = (np.sqrt(1.0 - s_th * s_th) * th
                        + s_th * self.ms.draw(self.innovation, rng))
            phi_p, pobs_p, keff_p = self.potential(th_p, w, lb)
            a = np.log(rng.random()) < st["phi"] - phi_p
            if a:
                st.update(theta=th_p, phi=phi_p, pobs=pobs_p, keff=keff_p)
            ad.record(0, a)
        # B: pCN on w with beta fixed (the level barely moves: beta mixes
        #    through D, rho through B and C).  pCN preserves N(0,1) for w;
        #    the curve prior of lb enters the ratio.
        th, lb = st["theta"], st["lb"]
        w_p = float(pcn(w, s_h, rng))
        lp_new = self.log_p_lb(lb, self.rho(w_p))
        a = False
        if np.isfinite(lp_new):
            phi_p, pobs_p, keff_p = self.potential(th, w_p, lb)
            logA = st["phi"] - phi_p + lp_new - self.log_p_lb(lb, rho)
            a = np.log(rng.random()) < logA
            if a:
                st.update(w=w_p, phi=phi_p, pobs=pobs_p, keff=keff_p)
        ad.record(1, a)
        # C: ridge (rho, theta) with beta fixed -- log K unchanged, no solve
        th, w, lb = st["theta"], st["w"], st["lb"]
        eta = s_rg * rng.standard_normal()
        w_p = w + eta / self.cfg.s_rho
        th_p = np.exp(-eta) * th
        lp_new = self.log_p_lb(lb, self.rho(w_p))
        a = False
        if np.isfinite(lp_new):
            logA = (-0.5 * (th_p @ th_p - th @ th) - 0.5 * (w_p ** 2 - w ** 2)
                    + lp_new - self.log_p_lb(lb, self.rho(w)) - self.r * eta)
            a = np.log(rng.random()) < logA
            if a:
                st.update(theta=th_p, w=float(w_p))    # phi, pobs, keff unchanged
        ad.record(2, a)
        # D: level swap (beta, theta-mean) -- log K nearly unchanged, 1 solve
        th, w, lb = st["theta"], st["w"], st["lb"]
        rho = self.rho(w)
        delta = s_lv * rng.standard_normal()
        lb_p = lb + delta
        th_p = th - (delta / rho) * self.c
        phi_p, pobs_p, keff_p = self.potential(th_p, w, lb_p)
        logA = (st["phi"] - phi_p - 0.5 * (th_p @ th_p - th @ th)
                + self.log_p_lb(lb_p, rho) - self.log_p_lb(lb, rho))
        a = np.log(rng.random()) < logA
        if a:
            st.update(theta=th_p, lb=float(lb_p), phi=phi_p, pobs=pobs_p, keff=keff_p)
        ad.record(3, a)
        return st


# ------------------------------------------------------ checkpointed chains
TRACE_KEYS = ("rho", "beta", "phi", "level", "w", "theta_norm2")
KEPT_KEYS = ("theta", "rho", "beta", "pobs", "keff")


def run_chunk(sampler, chain_id, ckpt_path):
    """Advance one chain by cfg.chunk iterations (or to n_total) from its
    checkpoint; returns the new iteration count.  Works for any sampler with
    STATE (state variables besides theta), step(), potential_st(), beta_st(),
    level_st(), rho(), extra_traces() -- ThreeStepSampler (E4) and
    SoftCurveSampler (E5)."""
    cfg = sampler.cfg
    s_init = getattr(sampler, "s_init", cfg.s_init)
    targets = getattr(sampler, "targets", cfg.targets)
    n_moves = len(s_init)
    extra_keys = tuple(sampler.extra_traces(None))
    trace_keys = TRACE_KEYS + extra_keys
    if os.path.exists(ckpt_path):
        ck = dict(np.load(ckpt_path, allow_pickle=False))
        it0 = int(ck["it"])
        if it0 >= cfg.n_total:
            return it0
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(ck["rng_state"]))
        st = dict(theta=ck["theta_cur"])
        for key in sampler.STATE:
            st[key] = float(ck[key + "_cur"])
        st["phi"], st["pobs"], st["keff"] = sampler.potential_st(st)
        s = ck["s"]
        acc_tot = ck["acc_tot"]
        # proposals per move (E8's random block schedule skips moves); runs
        # before E8 proposed every move once per iteration
        prop_tot = ck["acc_tot_prop"] if "acc_tot_prop" in ck else np.full(n_moves, float(it0))
        traces = {k: list(ck["tr_" + k]) for k in trace_keys}
        traces["theta_lead"] = list(ck["tr_theta_lead"])
        kept = {k: list(ck["kept_" + k]) for k in KEPT_KEYS}
        s_hist = list(ck["s_hist"])
    else:
        it0 = 0
        rng = np.random.default_rng(chain_seed(cfg.seed0, chain_id,
                                               cfg.seed_scheme, 13))
        st = sampler.init_state(rng)
        s = np.array(s_init, float)
        acc_tot = np.zeros(n_moves)
        prop_tot = np.zeros(n_moves)
        traces = {k: [] for k in trace_keys}
        traces["theta_lead"] = []
        kept = {k: [] for k in KEPT_KEYS}
        s_hist = []
    ad = StepAdapter(s, targets=targets, every=cfg.adapt_every,
                     factor=cfg.adapt_factor)
    n_steps = min(cfg.chunk, cfg.n_total - it0)
    k = sampler.N_TRACK
    for loc in range(n_steps):
        it = it0 + loc
        before, before_n = ad.acc.copy(), ad.n.copy()
        sampler.step(st, ad, rng)
        acc_tot += ad.acc - before
        prop_tot += ad.n - before_n
        ad.tick(it, cfg.n_burn)
        if it < cfg.n_burn and (it + 1) % cfg.adapt_every == 0:
            s_hist.append(ad.s.copy())
        rho = sampler.rho(st["w"])
        beta = sampler.beta_st(st)
        if it % cfg.trace_every == 0:
            traces["rho"].append(rho)
            traces["beta"].append(beta)
            traces["phi"].append(st["phi"])
            traces["level"].append(sampler.level_st(st))
            traces["w"].append(st["w"])
            traces["theta_norm2"].append(float(st["theta"] @ st["theta"]))
            traces["theta_lead"].append(st["theta"][:k].copy())
            for key, v in sampler.extra_traces(st).items():
                traces[key].append(v)
        if it >= cfg.n_burn and (it - cfg.n_burn) % cfg.thin == 0:
            kept["theta"].append(st["theta"].copy())
            kept["rho"].append(rho)
            kept["beta"].append(beta)
            kept["pobs"].append(st["pobs"] if st["pobs"] is not None
                                else np.full(sampler.prob.syn.n_obs, np.nan))
            kept["keff"].append(st["keff"])
    it_new = it0 + n_steps
    out = dict(it=it_new, s=ad.s, acc_tot=acc_tot, acc_tot_prop=prop_tot,
               theta_cur=st["theta"],
               rng_state=json.dumps(rng.bit_generator.state),
               s_hist=np.array(s_hist).reshape(-1, n_moves))
    for key in sampler.STATE:
        out[key + "_cur"] = st[key]
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
