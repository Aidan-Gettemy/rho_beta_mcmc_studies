"""
E1 - the naive joint model sampled by standard global pCN (the failure case).

    log K(x) = log beta + rho * G(x; theta),   G(theta) = T theta   (global KLE)
    u = (theta, xi_b, xi_r) ~ N(0, I_{r+2})
    log beta = mu_logbeta + s_logbeta xi_b,     log rho = mu_logrho + s_logrho xi_r
    Phi(u) = ||d_p - O p(K(u))||^2 / (2 sigma_p^2)          (pressure data only)

One pCN move on the WHOLE vector u per iteration (one PDE solve):
    u' = sqrt(1 - s^2) u + s zeta,  zeta ~ N(0, I),   accept w.p. min(1, e^{Phi(u) - Phi(u')})
The step size s adapts toward `target` acceptance during burn-in, then freezes.

Why it fails (non-uniqueness report, Props. 1-2): the likelihood depends on
(beta, rho, theta) only through log K; rho is traded against |theta| and
beta against the constant component of G.  The data pin log K (its level and
amplitude) but not beta and rho separately, so chains fit the data equally
well while beta and rho wander over a prior-shaped plateau and different
chains settle in different places.

Saved per chain (chain<k>.npz), everything needed by reports/report_e1.py:
  per iteration (float32): acc (0/1), phi, relerr = ||d - p_obs|| / ||d||,
      logbeta, rho, level (spatial mean of log K), amp (rho * spatial std of G),
      theta_norm2
  every `thin` iterations (float64): kept_it, kept_theta (r), kept_logbeta,
      kept_rho, kept_phi, kept_pobs  -> exact reconstruction of log K and of
      the full pressure field of those accepted states (reconstruct_fields)
  adaptation: s_hist, s_final; resume state: u, rng_state, it
"""
import json
import os
import time

import numpy as np

from .samplers import StepAdapter
from .config import chain_seed

PER_IT = ("acc", "phi", "relerr", "logbeta", "rho", "level", "amp", "theta_norm2")
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


def run_chain(prob, cfg, chain_id, path, log=print):
    """Run (or resume) chain `chain_id` to cfg.n_iter; returns a summary."""
    model = NaiveModel(prob, cfg)
    r, n = model.r, cfg.n_iter
    t0 = time.time()
    if os.path.exists(path):
        z = dict(np.load(path, allow_pickle=False))
        it0 = int(z["it"])
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(z["rng_state"]))
        u = z["u"].copy()
        hist = {k: np.zeros(n, np.float32) for k in PER_IT}
        for k in PER_IT:
            hist[k][:it0] = z[k][:it0]
        kept = {k: list(z[k]) for k in KEPT}
        s_hist = list(z["s_hist"])
        ad = StepAdapter([float(z["s_final"])], targets=(cfg.target,),
                         every=cfg.adapt_every, factor=cfg.adapt_factor)
        log(f"chain {chain_id}: resuming at {it0}/{n}")
    else:
        it0 = 0
        rng = np.random.default_rng(chain_seed(cfg.seed0, chain_id, cfg.seed_scheme, 1))
        u = rng.standard_normal(r + 2) if cfg.init == "prior" else np.zeros(r + 2)
        hist = {k: np.zeros(n, np.float32) for k in PER_IT}
        kept = {k: [] for k in KEPT}
        s_hist = []
        ad = StepAdapter([cfg.s0], targets=(cfg.target,), every=cfg.adapt_every,
                         factor=cfg.adapt_factor)
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
        out.update(it=it, n_iter=n, chain_id=chain_id, u=u, s_final=ad.s[0],
                   s_hist=np.asarray(s_hist), rng_state=json.dumps(rng.bit_generator.state),
                   seconds=time.time() - t0)
        _save(path, out)

    for it in range(it0, n):
        s = ad.s[0]
        up = np.sqrt(1.0 - s * s) * u + s * rng.standard_normal(r + 2)
        phi_p, pobs_p, rel_p = model.evaluate(up)
        acc = np.log(rng.random()) < phi - phi_p
        if acc:
            u, phi, pobs, rel = up, phi_p, pobs_p, rel_p
            summ = model.summaries(u)
        ad.record(0, acc)
        ad.tick(it, cfg.n_burn)
        if it < cfg.n_burn and (it + 1) % cfg.adapt_every == 0:
            s_hist.append(ad.s[0])
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
            a = hist["acc"][max(0, it + 1 - cfg.log_every):it + 1].mean()
            log(f"chain {chain_id}: {it + 1}/{n}  acc(last {cfg.log_every})={a:.2f}  "
                f"s={ad.s[0]:.3g}  phi={phi:.1f}  relerr={rel:.2e}  "
                f"beta={np.exp(summ['logbeta']):.3g}  rho={summ['rho']:.3g}  "
                f"({(time.time() - t0) / 60:.1f} min)")
        if (it + 1) % cfg.checkpoint_every == 0 or it + 1 == n:
            checkpoint(it + 1)
    acc_all = float(hist["acc"].mean())
    acc_post = float(hist["acc"][cfg.n_burn:].mean()) if n > cfg.n_burn else acc_all
    return dict(chain_id=chain_id, acc=acc_all, acc_post=acc_post, s_final=float(ad.s[0]),
                beta_final=float(np.exp(summ["logbeta"])), rho_final=summ["rho"],
                relerr_final=float(rel), minutes=(time.time() - t0) / 60)


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
    out = {k: np.stack([z[k][:L] for z in zs]) for k in PER_IT}
    for k in KEPT:
        out[k] = [z[k] for z in zs]                   # ragged across chains
    out["it"] = np.array([int(z["it"]) for z in zs])
    out["s_final"] = np.array([float(z["s_final"]) for z in zs])
    out["s_hist"] = [z["s_hist"] for z in zs]
    out["chain_id"] = np.array([int(z["chain_id"]) for z in zs])
    out["n_common"] = L
    return out
