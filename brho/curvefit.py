"""
Fits of the offline beta(rho) curve from Monte Carlo effective permeabilities.

    h(rho) = mean_i log( keff_i(rho) / K0 )         (keff from fields K = K0 exp(rho YP_i))
    even polynomials   c0 + a rho^2 [+ b rho^4 [+ c rho^6]]        (weighted by 1/se^2 by default)
    PCHIP of h through the MC means, anchored at h(0) = 0           (no parametric form)
    beta(rho) = Kexp * exp(-h(rho))               (Kexp = measured effective permeability)

K0 cancels (scaling lemma Keff(cK) = c Keff(K)).  With no heterogeneity h(0) = 0, so a fitted c0
far from 0 means a problem in the solver or the fields.  Leave-one-out errors (in units of the MC
standard error) compare the four models on prediction, not just on fit.

Relation to brho/curve.py: BetaCurve works with keff1 = keff / K0 (beta = 1 solves) and the
statistic f = mean Keff ("ratio_of_mean"), PCHIP in log f.  Here the statistic is the mean of
log Keff (the geometric mean), which is much less dominated by single realizations at large rho
(log Keff has sd 1.7 at rho = 5 for the 38x38x45 problem).  Both are computed from the same data.
Polynomials are valid only inside [rho_min, rho_max]; do not extrapolate.
"""
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator


def mc_statistics(rhos, keff, K0):
    """Mean and standard error of log(keff/K0) at each rho."""
    logk = np.log(keff / K0)
    n = logk.shape[1]
    h = logk.mean(axis=1)
    sd = logk.std(axis=1, ddof=1)
    return h, sd / np.sqrt(n), sd, n


def synthetic_data(seed=0, n_rho=40, n=1024):
    """Fake keff with a known answer: h(rho) = 0.15 rho^2 - 0.004 rho^4, scatter growing with rho."""
    rng = np.random.default_rng(seed)
    rhos = np.linspace(0.05, 5.0, n_rho)
    K0 = 1.0e-14
    h_true = 0.15 * rhos**2 - 0.004 * rhos**4
    s = 0.05 + 0.12 * rhos
    z = rng.standard_normal(n)                      # common random numbers across rho
    keff = K0 * np.exp(h_true[:, None] + s[:, None] * z[None, :])
    return rhos, keff, K0, h_true


# --------------------------------------------------------------------------- fits
def design(rhos, degree):
    """Columns rho^0, rho^2, ..., rho^degree (even powers only)."""
    powers = np.arange(0, degree + 1, 2)
    return np.stack([rhos**p for p in powers], axis=1), powers


def fit_model(rhos, h, se, degree, weighted=True):
    X, powers = design(rhos, degree)
    w = 1.0 / se if weighted else np.ones_like(se)
    Xw, hw = X * w[:, None], h * w
    coef, *_ = np.linalg.lstsq(Xw, hw, rcond=None)
    dof = len(rhos) - X.shape[1]
    resid = h - X @ coef
    # covariance of the coefficients; if unweighted, scale by the residual variance
    cov = np.linalg.inv(Xw.T @ Xw)
    if not weighted:
        cov *= (resid @ resid) / dof
    chi2 = float(np.sum((resid / se) ** 2))
    n = len(rhos)
    k = X.shape[1]
    return dict(
        degree=degree, powers=powers.tolist(), coef=coef.tolist(),
        coef_se=np.sqrt(np.diag(cov)).tolist(), cov=cov.tolist(),
        rms_resid=float(np.sqrt(np.mean(resid**2))),
        max_abs_resid_over_se=float(np.max(np.abs(resid / se))),
        chi2=chi2, dof=int(dof), chi2_per_dof=chi2 / dof,
        aic=chi2 + 2 * k, bic=chi2 + k * np.log(n),
    )


def evaluate(model, rho):
    X, _ = design(np.atleast_1d(rho), model["degree"])
    return X @ np.array(model["coef"])


def pchip_model(rhos, h):
    """PCHIP of h(rho) through the MC means, anchored at h(0) = 0 (Keff = K0 exactly with no heterogeneity)."""
    return PchipInterpolator(np.r_[0.0, rhos], np.r_[0.0, h], extrapolate=False)


def loo_errors(rhos, h, se, degrees, weighted=True):
    """Leave-one-out prediction error / se at the interior points (the last point is skipped: it would be
    extrapolation for PCHIP). Returns {name: rms of (prediction - h_i)/se_i}."""
    out = {f"poly{d}": [] for d in degrees}
    out["pchip"] = []
    for i in range(len(rhos) - 1):
        keep = np.arange(len(rhos)) != i
        for d in degrees:
            m = fit_model(rhos[keep], h[keep], se[keep], d, weighted)
            out[f"poly{d}"].append((evaluate(m, rhos[i])[0] - h[i]) / se[i])
        out["pchip"].append((float(pchip_model(rhos[keep], h[keep])(rhos[i])) - h[i]) / se[i])
    return {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in out.items()}


def formula(model, Kexp):
    names = ["c0", "a", "b", "c"]
    terms = [f"{names[i]}*rho^{p}" if p else "c0" for i, p in enumerate(model["powers"])]
    return "beta(rho) = Kexp*exp(-(" + " + ".join(terms) + "))  with Kexp=%.6e" % Kexp


# --------------------------------------------------------------------------- output
def make_figure(rhos, h, se, models, Kexp, path, pchip=None, loo=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(rhos.min(), rhos.max(), 400)
    styles = {2: ("tab:blue", "-"), 4: ("tab:orange", "--"), 6: ("tab:green", ":")}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    ax[0].errorbar(rhos, h, yerr=se, fmt="o", ms=3, color="k", label="MC mean of log(keff/K0)")
    for m in models:
        c, ls = styles[m["degree"]]
        ax[0].plot(grid, evaluate(m, grid), color=c, ls=ls, label=f"degree {m['degree']}")
    if pchip is not None:
        ax[0].plot(grid, pchip(grid), color="tab:purple", ls="-.", label="PCHIP")
    ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel(r"$h(\rho)$"); ax[0].legend(); ax[0].grid(alpha=.3)

    for m in models:
        c, ls = styles[m["degree"]]
        ax[1].plot(rhos, (h - evaluate(m, rhos)) / se, color=c, ls=ls, marker="o", ms=3,
                   label=f"deg {m['degree']}  chi2/dof={m['chi2_per_dof']:.2f}")
    ax[1].axhline(0, color="k", lw=.8)
    for y in (-2, 2):
        ax[1].axhline(y, color="gray", lw=.6, ls=":")
    ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel("residual / standard error")
    ax[1].legend(); ax[1].grid(alpha=.3)
    if loo:
        ax[1].set_title("leave-one-out rms/se: " + ", ".join(f"{k} {v:.1f}" for k, v in loo.items()), fontsize=8)

    ax[2].errorbar(rhos, Kexp * np.exp(-h) / Kexp, yerr=Kexp * np.exp(-h) * se / Kexp, fmt="o",
                   ms=3, color="k", label="MC")
    for m in models:
        c, ls = styles[m["degree"]]
        ax[2].plot(grid, np.exp(-evaluate(m, grid)), color=c, ls=ls, label=f"degree {m['degree']}")
    if pchip is not None:
        ax[2].plot(grid, np.exp(-pchip(grid)), color="tab:purple", ls="-.", label="PCHIP")
    ax[2].set_yscale("log"); ax[2].set_xlabel(r"$\rho$"); ax[2].set_ylabel(r"$\beta(\rho)/K_{exp}$")
    ax[2].legend(); ax[2].grid(alpha=.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)




def fit_all(rhos, keff, K0, Kexp, degrees=(2, 4, 6), weighted=True):
    """All fits from keff (n_rho, N).  Returns a dict (JSON-able) plus the objects needed to plot."""
    keff = np.asarray(keff)
    nbad = int(np.sum(~np.isfinite(keff)))
    if nbad:
        raise ValueError(f"{nbad} of {keff.size} keff values are NaN/inf (failed solves); "
                         "delete those realizations' parts and rerun first")
    h, se, sd, n = mc_statistics(rhos, keff, K0)
    models = [fit_model(rhos, h, se, d, weighted) for d in degrees]
    pchip = pchip_model(rhos, h)
    loo = loo_errors(rhos, h, se, degrees, weighted)
    out = dict(Kexp=Kexp, K0=K0, n_realizations=int(n), rho_range=[float(rhos.min()), float(rhos.max())],
               weighting="1/se^2" if weighted else "none",
               note="Polynomials are valid only inside rho_range; do not extrapolate.",
               models=[dict(m, formula=formula(m, Kexp)) for m in models],
               loo_rms_over_se=loo,
               pchip=dict(anchor="h(0)=0", rho=[0.0] + list(map(float, rhos)), h=[0.0] + list(map(float, h)),
                          note="PCHIP of h through the MC means; valid on [0, rho_max] only (NaN outside)"))
    return out, dict(h=h, se=se, sd=sd, n=n, models=models, pchip=pchip, loo=loo)


def write_fit(out, extra, rhos, tables_dir, figures_dir):
    """fits.json, fit_table.csv, beta_pchip_table.csv (tables) and fit_curves.png (figures)."""
    h, se, sd, models, pchip = extra["h"], extra["se"], extra["sd"], extra["models"], extra["pchip"]
    Kexp = out["Kexp"]
    with open(os.path.join(tables_dir, "fits.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(tables_dir, "fit_table.csv"), "w") as f:
        f.write("rho,h_mean,h_se,h_sd," + ",".join(f"h_fit_deg{m['degree']}" for m in models) + "\n")
        fits = [evaluate(m, rhos) for m in models]
        for i in range(len(rhos)):
            f.write(f"{rhos[i]:.6f},{h[i]:.8e},{se[i]:.8e},{sd[i]:.8e}," +
                    ",".join(f"{ft[i]:.8e}" for ft in fits) + "\n")
    fine = np.linspace(0.0, float(np.max(rhos)), 1001)
    hf = pchip(fine)
    with open(os.path.join(tables_dir, "beta_pchip_table.csv"), "w") as f:
        f.write("rho,h,beta_over_Kexp,beta\n")
        for r, v in zip(fine, hf):
            f.write(f"{r:.6f},{v:.8e},{np.exp(-v):.8e},{Kexp * np.exp(-v):.8e}\n")
    make_figure(rhos, h, se, models, Kexp, os.path.join(figures_dir, "fit_curves.png"),
                pchip=pchip, loo=extra["loo"])


def summary_lines(out, rhos):
    lines = [f"N = {out['n_realizations']} realizations, {len(rhos)} rho values in "
             f"[{rhos.min():.2f}, {rhos.max():.2f}]"]
    for m in out["models"]:
        lines.append(f"degree {m['degree']}: coef={np.round(m['coef'], 6).tolist()}  "
                     f"chi2/dof={m['chi2_per_dof']:.2f}  AIC={m['aic']:.1f}  BIC={m['bic']:.1f}  "
                     f"max|resid|/se={m['max_abs_resid_over_se']:.2f}")
    lines.append("leave-one-out rms error / se (lower = better prediction): " +
                 ", ".join(f"{k} {v:.2f}" for k, v in out["loo_rms_over_se"].items()))
    lines.append("c0 should be near 0 for a no-nozzle solver; a clearly nonzero c0 means check the solver or fields.")
    return lines
