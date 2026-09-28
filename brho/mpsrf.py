"""
MPSRF / PSRF curves in the group's MATLAB convention (getMSPRF +
computeMPSRFcurve + the crossing-iteration read-out).

getMSPRF (one window, theta: (dim, n, m) in MATLAB, here (m, n, dim)):
    W      = 1/m sum_j  cov_j                    (within, (n-1) normalised)
    B/n    = cov of the m chain means            ((m-1) normalised)
    MPSRF  = sqrt((n-1)/n + (m+1)/m * lambda_max(W^{-1} B/n))
    V      = (n-1)/n W + (m+1)/m B/n
    PSRF   = max_i sqrt(V_ii / W_ii)             (worst single component)

Curve (computeMPSRFcurve): at thinned-sample counts
    k = step + iter_burn, step + iter_burn + step, ..., <= n  (1-based)
the window is samples iter_burn, iter_burn + skip, ..., k: it grows, always
starts at sample iter_burn (the burn-in), and uses every skip-th sample.
The x axis is k * thinning (MCMC iterations).

Convergence read-out: first curve point (after the first 10) with
MPSRF < 1.2, linearly interpolated between it and the previous point.
"""
import numpy as np


def get_mpsrf(theta):
    """theta: (m chains, n samples, dim).  Returns (MPSRF, PSRF_max, PSRF_i)."""
    th = np.asarray(theta, float)
    m, n, dim = th.shape
    mu = th.mean(1)                                   # (m, dim)
    W = np.zeros((dim, dim))
    for j in range(m):
        X = th[j] - mu[j]
        W += (X.T @ X) / (n - 1) / m
    D = mu - mu.mean(0)
    Bn = (D.T @ D) / (m - 1)
    lam = np.linalg.eigvals(np.linalg.solve(W, Bn))
    lam1 = float(np.abs(lam[np.argmax(lam.real)]))    # MATLAB: eigs(.,1), abs()
    mpsrf = np.sqrt((n - 1) / n + (m + 1) / m * lam1)
    V = (n - 1) / n * W + (m + 1) / m * Bn
    psrf_i = np.sqrt(np.diag(V) / np.diag(W))
    return float(mpsrf), float(psrf_i.max()), psrf_i


def mpsrf_curve(samples, skip=2, step=15, iter_burn=10, chains=None,
                dispiter=0, log=print):
    """computeMPSRFcurve.  samples: (m, n, dim) thinned samples.

    MATLAB (1-based):  iter_step = (step + iter_burn):step:maxIter
                       idx       = iter_burn:skip:iter_step(u)
    i.e. a growing window that always starts at sample iter_burn and uses
    every skip-th sample.  Returns k (= iter_step, 1-based thinned count),
    mpsrf, psrf (max over components), psrf_i (len(k), dim).  A point whose
    window is too short for W to be invertible is NaN (kept, so that the
    'ignore the first 10 points' rule counts the same points as MATLAB)."""
    s = np.asarray(samples, float)
    if chains is not None:
        s = s[list(chains)]
    m, n, dim = s.shape
    if m < 2:
        raise ValueError("MPSRF needs at least 2 chains")
    ks = np.arange(step + iter_burn, n + 1, step)
    out = dict(k=[], mpsrf=[], psrf=[], psrf_i=[])
    for u, k in enumerate(ks, start=1):
        sub = s[:, iter_burn - 1:k:skip]            # MATLAB iter_burn:skip:k
        nw = sub.shape[1]
        if nw < 2 or m * (nw - 1) <= dim:           # W singular
            mp, pm, pi = np.nan, np.nan, np.full(dim, np.nan)
        else:
            mp, pm, pi = get_mpsrf(sub)
        out["k"].append(k)
        out["mpsrf"].append(mp)
        out["psrf"].append(pm)
        out["psrf_i"].append(pi)
        if dispiter and u % dispiter == 0:
            log(f"Step {u} / {len(ks)} | MPSRF = {mp:.3f}")
    return {key: np.asarray(v) for key, v in out.items()}


def crossing_iteration(t1, v1, t2, v2, eps):
    return t1 + (t2 - t1) * (eps - v1) / (v2 - v1)


def convergence_iteration(iters, curve, level=1.2, ignore_first=10):
    """First iteration where `curve` < level (ignoring the first points),
    interpolated as in the MATLAB script.  None if it never crosses."""
    c = np.abs(np.asarray(curve, float)).copy()
    c[:ignore_first] = np.inf
    c[np.isnan(c)] = np.inf
    idx = np.flatnonzero(c < level)
    if len(idx) == 0:
        return None
    j = int(idx[0])
    if j == 0:
        return float(iters[0])
    return float(np.ceil(crossing_iteration(iters[j - 1], curve[j - 1],
                                            iters[j], curve[j], level)))
