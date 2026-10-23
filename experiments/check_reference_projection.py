"""Model-space check with the TRUE field: can the sampler's parametrization represent the reference field?  Needs the
solver and tracer (GPU node) for the forward part; one process, no MPI.

  python experiments/check_reference_projection.py --exp-tag refcheck --ref-beta 9.869232667160132e-14 --ref-rho 1.0 \\
        --rhos 0.8 1.0 1.25 [--gauss-field external/data/<gaussian field file>] [the model flags of run_glam.py]

The sampler's field is  log K = log beta(rho) + rho * YP(theta),  YP = P A theta (assemble, whitened projection).  For each
rho and each LEVEL
    curve : log beta(rho) from the offline PCHIP curve (what the sampler uses)
    true  : log of the reference beta (--ref-beta), i.e. the level the reference field was generated with
the target is g = (log Kref - level) / rho, and theta* = argmin ||g - P A theta||^2 (brho/glam/projection.py, LSQR,
minimum-norm).  Printed per row:
    resid      ||g - YP|| / ||g||  over all cells    (how much of the field the model space cannot represent)
    dlogK      RMS of (log K_fit - log Kref) over the plug cells (and all cells)
    |theta|^2  mean theta*^2: about 1 for a typical prior draw; much larger means the true field is a far-out prior draw
    Phi, RE    the likelihood of the forward model run on the fitted field (the floor a sampler can reach with this rho and level)
plus the unprojected reference field itself (the forward-model sanity check).  If --gauss-field is given, it also checks
that Kref = beta * exp(rho * G) for the stated beta and rho.
Writes tables/e6_glam_<tag>/reference_projection.json and results/e6_glam_<tag>/reference_projection.npz (theta*, w, Phi
of every row: usable as starting states).
"""
import _common  # noqa: F401
import json
import os
import time

import numpy as np

import run_glam
from brho import config as C
from brho import io
from brho.glam import projection as PJ
from brho.glam.post import plug_mask
from brho.glam_likelihood import phi_like


def main():
    ap = run_glam.build_parser(__doc__)
    ap.add_argument("--ref-field", default=os.environ.get("EXT_REF_FIELD", "external/data/Kref.dat"),
                    help="reference permeability K [m^2], solver order")
    ap.add_argument("--ref-beta", type=float, default=9.869232667160132e-14, help="beta the reference field was generated with")
    ap.add_argument("--ref-rho", type=float, default=1.0, help="rho the reference field was generated with")
    ap.add_argument("--gauss-field", default=None, help="the Gaussian field G of the reference (text, solver order), optional")
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.8, 1.0, 1.25])
    ap.add_argument("--lsqr-iter", type=int, default=300)
    ap.add_argument("--damp", type=float, default=0.0, help="Tikhonov damping of the fit (0 = minimum-norm least squares)")
    ap.add_argument("--no-forward", action="store_true", help="projection only: no solver, no tracer")
    args = ap.parse_args()
    cfg = C.from_cli(args, C.GlamConfig, quick=False)
    from brho.darcy_external import physics_from_cfg
    ph = physics_from_cfg(args.cfg)
    grid = (ph["nx"], ph["ny"], ph["nz"])
    plug = plug_mask(*grid)
    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")

    t0 = time.time()
    model = run_glam.build_model(args, cfg, 0)
    print(f"[refproj] model built in {time.time() - t0:.0f} s; grid {grid}; sampler curve: Kexp {model.beta.Kexp:.4e}", flush=True)
    logk_ref = np.log(np.loadtxt(args.ref_field))
    sd_plug = float(np.std(logk_ref[plug]))
    lb_true = float(np.log(args.ref_beta))
    g_true = (logk_ref - lb_true) / args.ref_rho
    print(f"[refproj] reference field: mean log K {logk_ref[plug].mean():.4f} (plug), sd {sd_plug:.4f}; "
          f"with beta {args.ref_beta:.6e}, rho {args.ref_rho}: g = (log K - log beta)/rho has mean {g_true[plug].mean():.4f}, "
          f"sd {g_true[plug].std():.4f} over the plug cells (a unit Gaussian would give 0 and 1)", flush=True)
    if args.gauss_field:
        G = np.loadtxt(args.gauss_field)
        dev = np.abs(logk_ref - (lb_true + args.ref_rho * G))
        print(f"[refproj] Kref vs beta * exp(rho * G): max |log K - (log beta + rho G)| = {dev.max():.3e} "
              f"({'consistent' if dev.max() < 1e-4 else 'NOT consistent: check beta, rho and the file'})", flush=True)

    rows, store = [], {}

    def forward(K):
        t = time.time()
        sim = model.simulate(K)
        if sim.conc is None:
            return None
        phi, (pc, pd) = phi_like(model.data, sim.conc, sim.dp_pa, cfg.sigma_conc, cfg.sigma_dp)
        return dict(phi=float(phi), re_conc=float(np.sqrt(2 * cfg.sigma_conc ** 2 * pc)),
                    re_dp=float(np.sqrt(2 * cfg.sigma_dp ** 2 * pd)), dp_pa=float(sim.dp_pa), seconds=time.time() - t)

    if not args.no_forward:
        r = forward(np.exp(logk_ref))
        rows.append(dict(rho=None, level="unprojected reference", **(r or {})))
        if r:
            print(f"[refproj] unprojected reference: Phi {r['phi']:.4g}  RE_conc {r['re_conc']:.4g}  RE_dp {r['re_dp']:.4g}  "
                  f"dp {r['dp_pa']:.2f} Pa (data {model.data.dp_pa:.2f})", flush=True)

    for rho in args.rhos:
        for level in ("curve", "true"):
            lb = float(model.beta.log_beta(rho)) if level == "curve" else lb_true
            g = (logk_ref - lb) / rho
            t = time.time()
            fit = PJ.fit_theta(model.field, g, iter_lim=args.lsqr_iter, damp=args.damp)
            logk_fit = lb + rho * fit["yp"]
            d = logk_fit - logk_ref
            row = dict(rho=float(rho), level=level, log_level=lb, beta_level=float(np.exp(lb)), resid=fit["residual"],
                       dlogk_rms_plug=float(np.sqrt(np.mean(d[plug] ** 2))), dlogk_rms_all=float(np.sqrt(np.mean(d ** 2))),
                       theta_msq=fit["theta_msq"], lsqr_iterations=fit["iterations"], lsqr_istop=fit["istop"],
                       fit_seconds=time.time() - t)
            if not args.no_forward:
                row.update(forward(np.exp(logk_fit)) or {})
            rows.append(row)
            key = f"rho{rho:g}_{level}"
            store[f"theta_{key}"] = fit["theta"]
            store[f"w_{key}"] = (np.log(rho) - cfg.mu_logrho) / cfg.s_logrho
            print(f"[refproj] rho {rho:5.2f} level {level:5s}: resid {row['resid']:.4f}  dlogK(plug) {row['dlogk_rms_plug']:.4f} "
                  f"(all {row['dlogk_rms_all']:.4f}, sd logK {sd_plug:.3f})  |theta|^2 {row['theta_msq']:.3f}  LSQR {row['lsqr_iterations']} its"
                  + (f"  Phi {row['phi']:.4g}  RE_conc {row['re_conc']:.4g}  RE_dp {row['re_dp']:.4g}" if "phi" in row else ""), flush=True)

    os.makedirs(io.tables_dir(exp), exist_ok=True)
    with open(os.path.join(io.tables_dir(exp), "reference_projection.json"), "w") as f:
        json.dump(dict(ref_beta=args.ref_beta, ref_rho=args.ref_rho, sd_logk_plug=sd_plug, sigma_conc=cfg.sigma_conc,
                       sigma_dp=cfg.sigma_dp, damp=args.damp, rows=rows), f, indent=1)
    pdir = io.results_dir(exp)
    os.makedirs(pdir, exist_ok=True)
    np.savez_compressed(os.path.join(pdir, "reference_projection.npz"), **store)
    print(f"[refproj] wrote {io.tables_dir(exp)}/reference_projection.json and {pdir}/reference_projection.npz", flush=True)


if __name__ == "__main__":
    main()
