"""
Convergence and summary statistics.

psrf / mpsrf follow Brooks & Gelman (1998) as in plots_mpsrf.py:
    R^2   = (n-1)/n + (m+1)/m * (B/n)/W
    R_p^2 = (n-1)/n + (m+1)/m * lambda_max(W^{-1} B/n)
Growing-window curves use [t/2, t] (first half of each window discarded).
"""
import numpy as np


def psrf(chains):
    """chains: (m, n)."""
    chains = np.asarray(chains, float)
    m, n = chains.shape
    W = chains.var(1, ddof=1).mean()
    Bn = chains.mean(1).var(ddof=1)
    return float(np.sqrt((n - 1) / n + (m + 1) / m * Bn / W))


def split_rhat(chains):
    """Split-R-hat: each chain halved, then psrf over 2m half-chains."""
    chains = np.asarray(chains, float)
    h = chains.shape[1] // 2
    return psrf(np.concatenate([chains[:, :h], chains[:, h:2 * h]]))


def mpsrf(chains):
    """chains: (m, n, p)."""
    chains = np.asarray(chains, float)
    m, n, p = chains.shape
    W = np.zeros((p, p))
    mu = chains.mean(1)
    for c in range(m):
        X = chains[c] - mu[c]
        W += X.T @ X / (n - 1)
    W /= m
    D = mu - mu.mean(0)
    Bn = D.T @ D / (m - 1)
    lam = np.linalg.eigvals(np.linalg.solve(W, Bn)).real.max()
    return float(np.sqrt((n - 1) / n + (m + 1) / m * lam))


def ess(x):
    """Effective sample size of one chain (Geyer initial positive sequence)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or np.allclose(x, 0):
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    ac /= ac[0]
    s, t = 0.0, 1
    while t < n - 1 and ac[t] + ac[t + 1] > 0:
        s += ac[t] + ac[t + 1]
        t += 2
    return float(n / (1 + 2 * s))


def ess_multi(chains):
    return float(sum(ess(c) for c in np.asarray(chains)))


def window_curve(fn, arr, n_points=40, t_min=None):
    """Evaluate fn(arr[:, t//2:t]) on growing windows. arr: (m, n[, p])."""
    n = arr.shape[1]
    t_min = t_min or max(40, n // 50)
    ts = np.unique(np.linspace(t_min, n, n_points).astype(int))
    return ts, np.array([fn(arr[:, t // 2:t]) for t in ts])


def summarize(samples, truth=None, prob=(5, 95)):
    """Median, central interval, mean, sd, and truth-in-interval / z-score."""
    s = np.asarray(samples, float).ravel()
    lo, hi = np.percentile(s, prob)
    out = dict(median=float(np.median(s)), lo=float(lo), hi=float(hi),
               mean=float(s.mean()), sd=float(s.std(ddof=1)))
    if truth is not None:
        out.update(truth=float(truth), covered=bool(lo <= truth <= hi),
                   z=float((out["mean"] - truth) / out["sd"]) if out["sd"] > 0 else np.nan)
    return out


def pointwise_coverage(samples, truth, prob=(5, 95)):
    """Fraction of cells whose truth lies in the pointwise interval."""
    lo, hi = np.percentile(samples, prob, axis=0)
    return float(np.mean((truth >= lo) & (truth <= hi)))


def pointwise_moments(theta, T):
    """Pointwise mean and variance over samples of the field G = T theta,
    exactly from the sample mean and (population) covariance of theta:
        mean = T mu,   var_i = (T C T^T)_ii.
    No (n_samples x cells) matrix is formed.  theta: (..., r)."""
    th = np.asarray(theta, float).reshape(-1, np.shape(theta)[-1])
    mu = th.mean(0)
    C0 = np.cov(th.T, bias=True)
    return T @ mu, np.einsum("ij,jk,ik->i", T, C0, T)
