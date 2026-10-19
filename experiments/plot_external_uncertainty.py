"""
Uncertainty figure for the external-solver beta(rho) curve: one panel per model (polynomials of
degree 2, 4, 6 and PCHIP), each with the scatter of the individual realizations, the MC mean, the
median fit and its 90% interval.  No solver is run; it reads curve_external.npz.

    python experiments/plot_external_uncertainty.py --Kexp 1.036016e-13          # study e3_external
    python experiments/plot_external_uncertainty.py --exp-tag pilot --n-boot 500

The interval is a bootstrap over the realizations (the same resample at every rho, because the fields
are shared across rho), refitting each model on every resample.  It is the sampling uncertainty of the
FITTED CURVE; the spread of single realizations is the grey scatter and the dashed 5-95% lines.
Bottom row: median fit minus MC mean at the rho nodes, with the same interval - the model error
(offset from zero) against the sampling error (bar length).

Output: figures/<study>/fit_uncertainty.png, tables/<study>/fit_uncertainty.csv
(median, lo, hi of h(rho) and of beta(rho) = Kexp exp(-h) on a 400-point grid, per model).
"""
import argparse
import os

import _common  # noqa: F401  (first: pins threads before NumPy loads)
import numpy as np

from brho import io
from brho.curvefit import bootstrap_fits, fit_all, make_uncertainty_figure


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Kexp", type=float, default=1.036016e-13, help="measured effective permeability [m^2]")
    ap.add_argument("--exp-tag", default="", help="study suffix: results/e3_external_<tag>")
    ap.add_argument("--degrees", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--level", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unweighted", action="store_true")
    ap.add_argument("--curve", help="curve_external.npz (default: the study's results folder)")
    ap.add_argument("--out", help="figure path (default: figures/<study>/fit_uncertainty.png)")
    ap.add_argument("--stamp", help="large watermark text (used for layout previews on synthetic data)")
    args = ap.parse_args()

    exp = io.EXP_CURVE_EXTERNAL + (f"_{args.exp_tag}" if args.exp_tag else "")
    path = args.curve or os.path.join(io.results_dir(exp), "curve_external.npz")
    z = np.load(path)
    rhos, keff, K0 = z["rhos"], z["keff"], float(z["K0"])
    bad = int(np.sum(~np.isfinite(keff)))
    if bad:
        raise SystemExit(f"{bad} failed solves (NaN) in {path}: rerun the missing realizations first")

    weighted = not args.unweighted
    out, _ = fit_all(rhos, keff, K0, args.Kexp, args.degrees, weighted)
    sub = {f"poly{m['degree']}": "chi2/dof %.1f   leave-one-out %.2f se" % (
        m["chi2_per_dof"], out["loo_rms_over_se"][f"poly{m['degree']}"]) for m in out["models"]}
    sub["pchip"] = "interpolates the means   leave-one-out %.2f se" % out["loo_rms_over_se"]["pchip"]

    print(f"bootstrap: {args.n_boot} resamples of {keff.shape[1]} realizations, {len(rhos)} rho values ...")
    boot = bootstrap_fits(rhos, keff, K0, args.degrees, args.n_boot, args.seed, weighted, args.level)

    fig = args.out or os.path.join(io.figures_dir(exp), "fit_uncertainty.png")
    os.makedirs(os.path.dirname(fig), exist_ok=True)
    make_uncertainty_figure(rhos, keff, K0, boot, fig, subtitles=sub, stamp=args.stamp, seed=args.seed)
    print("saved", fig)

    tab = os.path.join(io.tables_dir(exp), "fit_uncertainty.csv") if not args.out else \
        os.path.splitext(fig)[0] + ".csv"
    os.makedirs(os.path.dirname(tab), exist_ok=True)
    cols = ["rho"]
    data = [boot["grid"]]
    for nm in boot["names"]:
        b = boot[nm]
        for key in ("median", "lo", "hi"):
            cols.append(f"h_{nm}_{key}")
            data.append(b[key])
        # beta = Kexp exp(-h): the interval is reversed (high h = low beta)
        cols += [f"beta_{nm}_median", f"beta_{nm}_lo", f"beta_{nm}_hi"]
        data += [args.Kexp * np.exp(-b["median"]), args.Kexp * np.exp(-b["hi"]), args.Kexp * np.exp(-b["lo"])]
    with open(tab, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(boot["grid"])):
            f.write(",".join(f"{d[i]:.8e}" for d in data) + "\n")
    print("saved", tab)


if __name__ == "__main__":
    main()
