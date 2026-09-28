"""
MPSRF / PSRF curves in the group's MATLAB convention (getMSPRF +
computeMPSRFcurve + the crossing-iteration read-out).

getMSPRF (one window, theta: (dim, n, m) in MATLAB, here (m, n, dim)):
    W      = 1/m sum_j  cov_j                    (within, (n-1) normalised)
    B/n    = cov of the m chain means            ((m-1) normalised)
    MPSRF  = sqrt((n-1)/n + (m+1)/m * lambda_max(W^{-1} B/n))
    V      = (n-1)/n W + (m+1)/m B/n
    PSRF   = max_i sqrt(V_ii / W_ii)             (worst single component)

Curve (computeMPSRFcurve): evaluated at thinned-sample counts
    k = iter_burn, iter_burn + step, ...,  on the window of samples
    (k - k // skip, k]  -> skip = 2 keeps the second half of the first k
    samples (Gelman-Rubin "discard the first half"), skip = 5 the last 4/5.
The x axis is the MCMC iteration of sample k (thinned index * thinning).

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
    """samples: (m, n, dim) thinned samples.  Returns dict of arrays:
    k (thinned count), mpsrf, psrf (max over components), psrf_i (len(k), dim)."""
    s = np.asarray(samples, float)
    if chains is not None:
        s = s[list(chains)]
    m, n, dim = s.shape
    if m < 2:
        raise ValueError("MPSRF needs at least 2 chains")
    ks = np.arange(iter_burn, n + 1, step)
    if len(ks) == 0 or ks[-1] != n:
        ks = np.append(ks, n)
    out = dict(k=[], mpsrf=[], psrf=[], psrf_i=[])
    for i, k in enumerate(ks):
        a = k // skip                       # discard the first 1/skip
        if k - a < 2:
            continue
        if m * (k - a - 1) <= dim:          # W singular: not enough samples yet
            continue
        mp, pm, pi = get_mpsrf(s[:, a:k])
        out["k"].append(k)
        out["mpsrf"].append(mp)
        out["psrf"].append(pm)
        out["psrf_i"].append(pi)
        if dispiter and i % dispiter == 0:
            log(f"  k = {k}/{n}  window {k - a}  MPSRF {mp:.3f}  max PSRF {pm:.3f}")
    return {key: np.asarray(v) for key, v in out.items()}


def crossing_iteration(t1, v1, t2, v2, eps):
    return t1 + (t2 - t1) * (eps - v1) / (v2 - v1)


def convergence_iteration(iters, curve, level=1.2, ignore_first=10):
    """First iteration where `curve` < level (ignoring the first points),
    interpolated as in the MATLAB script.  None if it never crosses."""
    c = np.abs(np.asarray(curve, float)).copy()
    c[:ignore_first] = np.inf
    idx = np.flatnonzero(c < level)
    if len(idx) == 0:
        return None
    j = int(idx[0])
    if j == 0:
        return float(iters[0])
    return float(np.ceil(crossing_iteration(iters[j - 1], curve[j - 1],
                                            iters[j], curve[j], level)))
