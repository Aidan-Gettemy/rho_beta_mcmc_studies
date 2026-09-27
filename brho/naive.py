"""
E1 - the naive joint model sampled by pCN (the failure case).

    log K(x) = log beta + rho * G(x; theta),   G(theta) = T theta   (global KLE)
    u = (theta, xi_b, xi_r) ~ N(0, I_{r+2})
    log beta = mu_logbeta + s_logbeta xi_b,     log rho = mu_logrho + s_logrho xi_r
    Phi(u) = ||d_p - O p(K(u))||^2 / (2 sigma_p^2)          (pressure data only)

Two moves (cfg.move):
  "blocked" (default, the standard): per iteration two Metropolis-Hastings
      steps, each a pCN proposal that preserves the N(0, I) prior of its block,
        A  theta' = sqrt(1 - s_t^2) theta + s_t zeta      (xi fixed)      1 solve
        B  xi'    = sqrt(1 - s_h^2) xi    + s_h zeta_2    (theta fixed)   1 solve
      accept each w.p. min(1, e^{Phi(u) - Phi(u')}); separate step sizes s_t,
      s_h, so the stiff level direction (xi_b: beta scales the whole field,
      which the influx data measure very precisely) does not throttle theta.
  "global": ONE pCN move on the whole vector u with a single step s (1 solve):
        u' = sqrt(1 - s^2) u + s zeta; the step is set by the stiffest
      direction (~0.005 here, against 0.009-0.05 for theta alone).
Step sizes adapt toward their targets during burn-in, then freeze.

Why it fails (non-uniqueness report, Props. 1-2): the likelihood depends on
(beta, rho, theta) only through log K; rho is traded against |theta| and
beta against the constant component of G.  The data pin log K (its level and
amplitude) but not beta and rho separately, so chains fit the data equally
well while beta and rho wander over a prior-shaped plateau and different
chains settle in different places.

Saved per chain (chain<k>.npz), everything needed by reports/report_e1.py:
  per iteration (float32): acc (0/1; block A for "blocked"), acc_h (0/1,
      block B; zeros for "global"), phi, relerr = ||d - p_obs|| / ||d||,
      logbeta, rho, level (spatial mean of log K), amp (rho * spatial std of G),
      theta_norm2
  every `thin` iterations (float64): kept_it, kept_theta (r), kept_logbeta,
      kept_rho, kept_phi, kept_pobs  -> exact reconstruction of log K and of
      the full pressure field of those accepted states (reconstruct_fields)
  adaptation: s_hist (k, n_steps), s_final (n_steps,); resume: u, rng_state, it
  solves: PDE solves used (1 or 2 per iteration)
"""
import json
import os
import time

import numpy as np

from .samplers import StepAdapter
from .config import chain_seed

PER_IT = ("acc", "acc_h", "phi", "relerr", "logbeta", "rho", "level", "amp", "theta_norm2")
KEPT = ("kept_it", "kept_theta", "kept_logbeta", "kept_rho", "kept_phi", "kept_pobs")


class NaiveModel:
    def __init__(self, prob, cfg):
        self.prob, self.cfg = prob, cfg
        self.r = prob.r
        self.d = prob.syn.d_p
        self.dnorm = float(np.linalg.norm(self.d))

    def unpack(self, u):
        c = self.cfg
        theta = u[:self.r]
        logbeta = c.mu_logbeta + c.s_logbeta * u[self.r]
        rho = float(np.exp(c.mu_logrho + c.s_logrho * u[self.r + 1]))
        return theta, float(logbeta), rho

    def logK(self, u):
        theta, lb, rho = self.unpack(u)
        return lb + rho * self.prob.kle.field(theta)

    def evaluate(self, u):
        """(Phi, p_obs, rel. error); Phi = inf if the field is not solvable."""
        with np.errstate(over="ignore", invalid="ignore"):
            lk = self.logK(u)
            if not np.all(np.isfinite(lk)) or lk.max() - lk.min() > 700:
                return np.inf, None, np.inf
            try:
                phi, pobs, _ = self.prob.potential(np.exp(lk))
            except Exception:                        # singular / failed solve
                return np.inf, None, np.inf
        if not np.isfinite(phi):
            return np.inf, None, np.inf
        return phi, pobs, float(np.linalg.norm(self.d - pobs) / self.dnorm)

    def summaries(self, u):
        theta, lb, rho = self.unpack(u)
        G = self.prob.kle.field(theta)
        return dict(logbeta=lb, rho=rho, level=lb + rho * G.mean(),
                    amp=rho * G.std(), theta_norm2=float(theta @ theta))


def _save(path, out):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def _adapter(cfg, s_init):
    blocked = cfg.move == "blocked"
    targets = (cfg.target, cfg.target_h) if blocked else (cfg.target,)
    return StepAdapter(s_init, targets=targets, every=cfg.adapt_every,
                       factor=cfg.adapt_factor)


def run_chain(prob, cfg, chain_id, path, log=print):
    """Run (or resume) chain `chain_id` to cfg.n_iter; returns a summary."""
    if cfg.move not in ("blocked", "global"):
        raise ValueError(cfg.move)
    blocked = cfg.move == "blocked"
    model = NaiveModel(prob, cfg)
    r, n = model.r, cfg.n_iter
    ns = 2 if blocked else 1                          # step sizes / solves per it
    t0 = time.time()
    if os.path.exists(path):
        z = dict(np.load(path, allow_pickle=False))
        it0 = int(z["it"])
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(z["rng_state"]))
        u = z["u"].copy()
        hist = {k: np.zeros(n, np.float32) for k in PER_IT}
        for k in PER_IT:
            if k in z:
                hist[k][:it0] = z[k][:it0]
        kept = {k: list(z[k]) for k in KEPT}
        s_hist = [np.atleast_1d(x) for x in z["s_hist"]]
        ad = _adapter(cfg, np.atleast_1d(z["s_final"]).astype(float))
        log(f"chain {chain_id}: resuming at {it0}/{n}")
    else:
        it0 = 0
        rng = np.random.default_rng(chain_seed(cfg.seed0, chain_id, cfg.seed_scheme, 1))
        u = rng.standard_normal(r + 2) if cfg.init == "prior" else np.zeros(r + 2)
        hist = {k: np.zeros(n, np.float32) for k in PER_IT}
        kept = {k: [] for k in KEPT}
        s_hist = []
        ad = _adapter(cfg, [cfg.s0, cfg.s0_h] if blocked else [cfg.s0])
    phi, pobs, rel = model.evaluate(u)
    tries = 0
    while not np.isfinite(phi):                      # unsolvable prior draw: redraw
        tries += 1
        if tries > 100:
            raise RuntimeError("no solvable initial state in 100 prior draws")
        u = rng.standard_normal(r + 2)
        phi, pobs, rel = model.evaluate(u)
    summ = model.summaries(u)

    def checkpoint(it):
        out = {k: hist[k][:it] for k in PER_IT}
        out.update({k: np.asarray(v) for k, v in kept.items()})
        if len(kept["kept_theta"]) == 0:
            out["kept_theta"] = np.zeros((0, r))
            out["kept_pobs"] = np.zeros((0, len(model.d)), np.float32)
        out.update(it=it, n_iter=n, chain_id=chain_id, u=u, s_final=ad.s.copy(),
                   s_hist=np.asarray(s_hist).reshape(-1, ns), move=cfg.move,
                   solves=it * ns, rng_state=json.dumps(rng.bit_generator.state),
                   seconds=time.time() - t0)
        _save(path, out)

    def mh(up):
        """One Metropolis-Hastings step to up; returns accepted."""
        nonlocal u, phi, pobs, rel, summ
        phi_p, pobs_p, rel_p = model.evaluate(up)
        if np.log(rng.random()) < phi - phi_p:
            u, phi, pobs, rel = up, phi_p, pobs_p, rel_p
            summ = model.summaries(u)
            return True
        return False

    for it in range(it0, n):
        if blocked:
            st, sh = ad.s
            up = u.copy()                                          # A: theta block
            up[:r] = np.sqrt(1.0 - st * st) * u[:r] + st * rng.standard_normal(r)
            acc = mh(up)
            ad.record(0, acc)
            up = u.copy()                                          # B: (xi_b, xi_r)
            up[r:] = np.sqrt(1.0 - sh * sh) * u[r:] + sh * rng.standard_normal(2)
            acc_h = mh(up)
            ad.record(1, acc_h)
            hist["acc_h"][it] = acc_h
        else:
            s = ad.s[0]
            acc = mh(np.sqrt(1.0 - s * s) * u + s * rng.standard_normal(r + 2))
            ad.record(0, acc)
        ad.tick(it, cfg.n_burn)
        if it < cfg.n_burn and (it + 1) % cfg.adapt_every == 0:
            s_hist.append(ad.s.copy())
        hist["acc"][it] = acc
        hist["phi"][it] = phi
        hist["relerr"][it] = rel
        for k in ("logbeta", "rho", "level", "amp", "theta_norm2"):
            hist[k][it] = summ[k]
        if it % cfg.thin == 0:
            theta, lb, rho = model.unpack(u)
            kept["kept_it"].append(it)
            kept["kept_theta"].append(theta.copy())
            kept["kept_logbeta"].append(lb)
            kept["kept_rho"].append(rho)
            kept["kept_phi"].append(phi)
            kept["kept_pobs"].append(pobs.astype(np.float32))
        if (it + 1) % cfg.log_every == 0:
            w = slice(max(0, it + 1 - cfg.log_every), it + 1)
            accs = f"acc={hist['acc'][w].mean():.2f}" + (
                f"/{hist['acc_h'][w].mean():.2f}" if blocked else "")
            steps = "s=" + "/".join(f"{x:.3g}" for x in ad.s)
            log(f"chain {chain_id} [{cfg.move}]: {it + 1}/{n}  {accs}  {steps}  "
                f"phi={phi:.1f}  relerr={rel:.2e}  beta={np.exp(summ['logbeta']):.3g}  "
                f"rho={summ['rho']:.3g}  ({(time.time() - t0) / 60:.1f} min)")
        if (it + 1) % cfg.checkpoint_every == 0 or it + 1 == n:
            checkpoint(it + 1)
    post = slice(cfg.n_burn, None) if n > cfg.n_burn else slice(None)
    out = dict(chain_id=chain_id, move=cfg.move, acc=float(hist["acc"].mean()),
               acc_post=float(hist["acc"][post].mean()),
               s_final=[float(x) for x in ad.s],
               beta_final=float(np.exp(summ["logbeta"])), rho_final=summ["rho"],
               relerr_final=float(rel), minutes=(time.time() - t0) / 60)
    if blocked:
        out.update(acc_h=float(hist["acc_h"].mean()),
                   acc_h_post=float(hist["acc_h"][post].mean()))
    return out


# ----------------------------------------------------------- post-processing
def reconstruct_fields(prob, theta, logbeta, rho, full_pressure=True):
    """Exact log K (and full cell pressure fields) of saved samples.
    theta (k, r), logbeta (k,), rho (k,) -> logK (k, N)[, p (k, N)]."""
    theta = np.atleast_2d(theta)
    lk = np.asarray(logbeta)[:, None] + np.asarray(rho)[:, None] * prob.kle.field(theta)
    if not full_pressure:
        return lk
    p = np.stack([prob.phys.solve(np.exp(row)) for row in lk])
    return lk, p


def load_chains(paths):
    """Stack chain files; per-iteration arrays are cut to the shortest chain."""
    zs = [np.load(p) for p in paths]
    L = min(int(z["it"]) for z in zs)
    out = {k: np.stack([z[k][:L] if k in z.files else np.zeros(L, np.float32)
                        for z in zs]) for k in PER_IT}
    for k in KEPT:
        out[k] = [z[k] for z in zs]                   # ragged across chains
    out["it"] = np.array([int(z["it"]) for z in zs])
    out["s_final"] = np.stack([np.atleast_1d(z["s_final"]).astype(float) for z in zs])
    out["s_hist"] = [np.asarray(z["s_hist"]).reshape(len(z["s_hist"]), -1) for z in zs]
    out["move"] = str(zs[0]["move"]) if "move" in zs[0].files else "global"
    out["chain_id"] = np.array([int(z["chain_id"]) for z in zs])
    out["n_common"] = L
    return out
