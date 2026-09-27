"""
Covariance kernels (the subset of KLE_matlab/+kle/kernels.m used so far).

A kernel is registered with
    cov(X, Y, p)   -> (n, m) covariance between point sets X (n, d), Y (m, d)
    factor(a, x, y, p) -> 1D factor along axis a, if the kernel is separable,
                          i.e. cov(X, Y) = prod_a factor(a, X[:, a], Y[:, a]);
                          None otherwise.
Parameters p: dict with "ell" (scalar, or one value per axis) and "sigma2".

Adding a kernel from the MATLAB registry = one entry in KERNELS.  A
non-separable kernel automatically uses the full (non-tensor) Nystrom solver.
"""
import numpy as np


def _ells(p, d):
    ell = np.atleast_1d(np.asarray(p["ell"], float))
    return np.full(d, ell[0]) if ell.size == 1 else ell


def sqexp_cov(X, Y, p):
    """sigma2 * exp(-sum_a (x_a - y_a)^2 / (2 l_a^2))."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    X = X[:, None] if X.ndim == 1 else X
    Y = Y[:, None] if Y.ndim == 1 else Y
    ell = _ells(p, X.shape[1])
    s2 = p.get("sigma2", 1.0)
    if X.shape[0] * Y.shape[0] * X.shape[1] <= 2e7 and np.all(ell == ell[0]):
        # explicit differences: bit-identical to the July scripts
        D2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        return s2 * np.exp(-D2 / (2.0 * ell[0] ** 2))
    Xs, Ys = X / ell, Y / ell                       # large sets: low memory
    d2 = (Xs ** 2).sum(1)[:, None] + (Ys ** 2).sum(1)[None, :] - 2 * Xs @ Ys.T
    return s2 * np.exp(-0.5 * np.maximum(d2, 0.0))


def sqexp_factor(axis, x, y, p, dim):
    ell = _ells(p, dim)[axis]
    x = np.asarray(x, float)[:, None]
    y = np.asarray(y, float)[None, :]
    s2 = p.get("sigma2", 1.0) if axis == 0 else 1.0      # sigma2 on one axis
    return s2 * np.exp(-0.5 * ((x - y) / ell) ** 2)


KERNELS = {
    "sqexp": dict(cov=sqexp_cov, factor=sqexp_factor,
                  doc="squared exponential (isotropic or per-axis ell)"),
}


def get(name):
    if name not in KERNELS:
        raise KeyError(f"unknown kernel {name!r}; have {sorted(KERNELS)}")
    return KERNELS[name]


def is_separable(name):
    return get(name)["factor"] is not None
