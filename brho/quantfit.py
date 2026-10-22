"""
Quantile-regression version of the offline beta(rho) curve fits.

The mean-based fits in curvefit.py (and their bootstrap bands) describe the uncertainty of the fitted MEAN
curve, which shrinks like 1/sqrt(N).  Here the target is the conditional distribution of a SINGLE realization:
for tau in (0.05, 0.5, 0.95) we fit the tau-quantile of y = log(keff/K0) given rho, using ALL points
(n_rho x n_realizations), i.e. minimise  sum rho_tau(y - f(rho)),  rho_tau(u) = u (tau - 1[u<0]).

  polynomials  f(rho) = c0 + a rho^2 [+ b rho^4 [+ c rho^6]]   exact LP solution (dual form, k equality rows)
  PCHIP        the node-wise quantile (saturated model: one free parameter per rho node, which is the exact
               quantile-regression solution for a model with a dummy per rho) interpolated by PCHIP anchored at
               f(0) = 0 (all quantiles vanish with no heterogeneity), like the mean-based PCHIP.

beta(rho) = Kexp exp(-h): the interval reverses (the 95% quantile of h is the 5% quantile of beta).
"""
import numpy as np
from scipy.optimize import linprog

from .curvefit import design, evaluate  # noqa: F401  (re-exported for convenience)
from scipy.interpolate import PchipInterpolator

TAUS = (0.05, 0.5, 0.95)


def quantile_poly(rhos, logk, degree, tau):
    """Even-polynomial quantile regression on all points. Returns the coefficient vector (c0, a, b, ...).

    Dual LP: max y.a  s.t.  X^T a = (1 - tau) X^T 1,  0 <= a <= 1  (HiGHS; n variables, k equality rows)."""
    n_rho, n = logk.shape
    X1, _ = design(np.asarray(rhos, float), degree)
    X = np.repeat(X1, n, axis=0)                      # (n_rho*n, k), same order as logk.ravel()
    y = logk.ravel()
    rhs = (1.0 - tau) * X.sum(axis=0)
    res = linprog(-y, A_eq=X.T, b_eq=rhs, bounds=(0.0, 1.0), method="highs")
    if not res.success:
        raise RuntimeError(f"quantile LP failed (degree {degree}, tau {tau}): {res.message}")
    # primal coefficients = equality-constraint duals
    coef = -np.asarray(res.eqlin.marginals, float)   # objective is -y.a, so beta = -marginals
    return coef


def pinball(y, f, tau):
    u = y - f
    return float(np.sum(u * (tau - (u < 0))))


def quantile_fits(rhos, keff, K0, degrees=(2, 4, 6), taus=TAUS):
    """Return dict name -> {tau: callable h_q(rho)} plus the node quantiles and the pinball losses."""
    rhos = np.asarray(rhos, float)
    logk = np.log(np.asarray(keff) / K0)
    out = {"taus": list(taus), "rhos": rhos, "node_quantiles": {}, "models": {}, "loss": {}}
    for tau in taus:
        out["node_quantiles"][tau] = np.quantile(logk, tau, axis=1)
    y = logk.ravel()
    for d in degrees:
        nm = f"poly{d}"
        out["models"][nm], out["loss"][nm] = {}, {}
        for tau in taus:
            coef = quantile_poly(rhos, logk, d, tau)
            out["models"][nm][tau] = dict(coef=coef.tolist(), powers=design(rhos, d)[1].tolist())
            f = np.repeat(design(rhos, d)[0] @ coef, logk.shape[1])
            out["loss"][nm][tau] = pinball(y, f, tau)
    out["models"]["pchip"], out["loss"]["pchip"] = {}, {}
    for tau in taus:
        q = out["node_quantiles"][tau]
        out["models"]["pchip"][tau] = PchipInterpolator(np.r_[0.0, rhos], np.r_[0.0, q], extrapolate=False)
        out["loss"]["pchip"][tau] = pinball(y, np.repeat(q, logk.shape[1]), tau)
    return out


def eval_quantile(fit, name, tau, rho):
    m = fit["models"][name][tau]
    if name == "pchip":
        return m(rho)
    X, _ = design(np.atleast_1d(rho), int(name[4:]))
    return X @ np.array(m["coef"])


def coverage(fit, keff, K0, name):
    """Fraction of ALL points below each quantile curve (should be ~tau)."""
    logk = np.log(np.asarray(keff) / K0)
    return {tau: float(np.mean(logk < eval_quantile(fit, name, tau, fit["rhos"])[:, None]))
            for tau in fit["taus"]}
