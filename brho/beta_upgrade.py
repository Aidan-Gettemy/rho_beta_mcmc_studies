"""Level-swap post-processing of hard-curve (E4) output: honest beta uncertainty
with no new PDE solves.

The 3-move sampler sets log beta = m(rho) = log beta_bar(rho) exactly.  The
pressure data fix the level, log beta + rho * mean(T theta), not how it splits
between beta and the field's mean, so theta carries whatever offset this field
has along c, the KLE coordinates of a constant field (T c ~ 1), and beta's
interval is far too narrow.

Restore the curve scatter, log beta | rho ~ N(m(rho), s(rho)^2), with s^2 the
scatter of log K_eff over the curve realizations plus sigma_e^2.  For each
hard-curve sample (rho, theta) move along the direction that leaves log K
unchanged:

    log beta' = m(rho) + delta,      theta' = theta - (delta / rho) c.

Writing the restored posterior in (theta, rho, delta), the likelihood depends
on log K only and the shear has Jacobian 1, so

    pi(theta', rho, log beta') = [hard-curve posterior of (theta, rho)]
                                 x exp(delta a / rho - delta^2 |c|^2 / (2 rho^2))
                                 x N(delta; 0, s(rho)^2),     a = theta . c,

using |theta - (delta/rho) c|^2 = |theta|^2 - 2 delta a / rho + delta^2 |c|^2 / rho^2.
The delta factor is Gaussian:

    delta | theta, rho ~ N(mu, 1/P),   P = |c|^2 / rho^2 + 1 / s(rho)^2,
                                       mu = (a / rho) / P,

and integrating it out leaves the importance weight of the hard-curve sample

    w(theta, rho) = exp(a^2 / (2 rho^2 P)) / (s(rho) sqrt(P)).

So: draw delta, shift beta and theta, weight by w.  Exact up to the part of the
constant field outside the KLE span (|T c - 1| / |1|: 3.4% at r = 57), which
changes log K by delta times that residual; one PDE solve per kept sample would
correct it (weight exp(Phi_old - Phi_new)), but is optional.
"""
import numpy as np


def constant_coords(T):
    """c = argmin |T c - 1| and the relative residual |T c - 1| / |1|."""
    ones = np.ones(T.shape[0])
    c, *_ = np.linalg.lstsq(T, ones, rcond=None)
    return c, float(np.linalg.norm(T @ c - ones) / np.sqrt(len(ones)))


def level_swap(theta, rho, c, m, s, rng, return_theta=True):
    """Upgrade hard-curve samples.

    theta: (n, r) samples; rho: (n,); c: (r,); m, s: arrays (n,) of
    log beta_bar(rho) and the curve scatter s(rho) at each sample.
    Returns dict with lb (new log beta), delta, mu, sd (= 1/sqrt(P)), logw,
    w (normalized weights), ess, and theta (shifted) if return_theta."""
    theta = np.asarray(theta, float)
    rho = np.asarray(rho, float)
    m = np.asarray(m, float)
    s = np.asarray(s, float)
    a = theta @ c
    cc = float(c @ c)
    P = cc / rho ** 2 + 1.0 / s ** 2
    mu = (a / rho) / P
    delta = mu + rng.standard_normal(len(rho)) / np.sqrt(P)
    logw = a ** 2 / (2.0 * rho ** 2 * P) - np.log(s) - 0.5 * np.log(P)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    out = dict(lb=m + delta, delta=delta, mu=mu, sd=1.0 / np.sqrt(P),
               logw=logw, w=w, ess=float(1.0 / np.sum(w ** 2)), a=a)
    if return_theta:
        out["theta"] = theta - (delta / rho)[:, None] * c[None, :]
    return out


def wquantile(x, q, w=None):
    """Weighted quantiles (w None: plain)."""
    x = np.asarray(x, float)
    if w is None:
        return np.quantile(x, q)
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw = (cw - 0.5 * w[o]) / cw[-1]
    return np.interp(q, cw, x[o])


def summarize(x, truth=None, w=None, level=0.90):
    """median, central interval, mean, sd (weighted if w), and coverage."""
    x = np.asarray(x, float)
    lo, med, hi = wquantile(x, [(1 - level) / 2, 0.5, (1 + level) / 2], w)
    ww = np.full(len(x), 1.0 / len(x)) if w is None else w
    mean = float(ww @ x)
    sd = float(np.sqrt(ww @ (x - mean) ** 2))
    out = dict(median=float(med), lo=float(lo), hi=float(hi), mean=mean, sd=sd)
    if truth is not None:
        out.update(truth=float(truth), covered=bool(lo <= truth <= hi))
    return out
