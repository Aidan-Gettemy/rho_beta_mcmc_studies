"""Effective rho (and beta) of the REFERENCE field, from the model's own definition.  No solver, no MPI.

    python experiments/reference_rho.py --ref-field external/data/Kref.dat
    python experiments/reference_rho.py --ref-field external/data/Kref.dat --ref-is-log --fits-json path/to/fits.json

The sampler's field is  log K = log beta(rho) + rho * YP  with YP of ~unit variance (whitened projection), so a field
that came from this model has  sd(log K) = rho  and  mean(log K) = log beta.  For the reference this gives
    rho_eff  = standard deviation of log Kref over the plug cells (the cylinder the tracer sees; all cells also printed)
    beta_eff = exp(mean log Kref) over the same cells,
compared with beta_curve(rho_eff) of the offline PCHIP curve (beta/Kexp).  CAVEATS: the reference may not have been
drawn from this model at all, in which case there is no 'true' rho; a single realization with a long correlation
length gives a noisy standard deviation (the number of independent cells is small); YP is unit-variance only
in expectation.  Use it as a rough reference for where the sampler's rho should land, not as a calibrated truth.
"""
import _common  # noqa: F401
import argparse
import json
import os

import numpy as np

from brho.glam.post import plug_mask


def reference_rho(logk, grid):
    """logk: solver-order vector (x fastest) of log K.  Returns dict with mean, sd and the effective rho/beta over the
    plug cells and over all cells, plus the per-z-layer sd of the plug cells."""
    logk = np.asarray(logk, float)
    nx, ny, nz = grid
    if logk.size != nx * ny * nz:
        raise ValueError(f"field has {logk.size} values, grid {grid} has {nx * ny * nz}")
    m = plug_mask(nx, ny, nz)
    out = {}
    for name, sel in (("plug", m), ("all", np.ones_like(m))):
        v = logk[sel]
        out[name] = dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std()), rho_eff=float(v.std()),
                         beta_eff=float(np.exp(v.mean())))
    layers = logk.reshape(nx, ny, nz, order="F")[m.reshape(nx, ny, nz, order="F").any(axis=2)]  # (cells in plug, nz)
    out["plug"]["layer_sd"] = [float(x) for x in layers.std(axis=0)]
    out["plug"]["layer_mean"] = [float(x) for x in layers.mean(axis=0)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default=os.environ.get("EXT_CFG", "external/DATA.cfg"))
    ap.add_argument("--ref-field", default=os.environ.get("EXT_REF_FIELD", "external/data/Kref.dat"))
    ap.add_argument("--ref-is-log", action="store_true", help="the file already holds log K")
    ap.add_argument("--fits-json", default=os.environ.get("GLAM_FITS", "runs/juno_external_2026-10/tables/e3_external/fits.json"))
    args = ap.parse_args()
    from brho.darcy_external import physics_from_cfg
    from brho.glam.model import BetaPchip
    ph = physics_from_cfg(args.cfg)
    grid = (ph["nx"], ph["ny"], ph["nz"])
    k = np.loadtxt(args.ref_field)
    logk = k if args.ref_is_log else np.log(k)
    r = reference_rho(logk, grid)
    curve = BetaPchip.from_fits_json(args.fits_json) if os.path.exists(args.fits_json) else None
    print(f"reference field {args.ref_field}: grid {grid}, {logk.size} cells")
    for name in ("plug", "all"):
        s = r[name]
        line = f"  {name:5s} ({s['n']:6d} cells): mean log K {s['mean']:.4f}, sd {s['sd']:.4f}  ->  rho_eff {s['rho_eff']:.3f}, beta_eff {s['beta_eff']:.4e} m^2"
        if curve is not None and 0.0 <= s["rho_eff"] <= curve.rho_max:
            b = float(curve.beta(s["rho_eff"]))
            line += f"   | curve beta({s['rho_eff']:.3f}) = {b:.4e}  (beta_eff / curve = {s['beta_eff'] / b:.3f})"
        print(line)
    if curve is not None:
        print(f"  Kexp (offline curve) = {curve.Kexp:.4e} m^2")
    ls = np.array(r["plug"]["layer_sd"])
    print(f"  per-z-layer sd over the plug cells: min {ls.min():.3f}, median {np.median(ls):.3f}, max {ls.max():.3f}")
    print("  (a rough reference only: see the caveats in the script's docstring)")


if __name__ == "__main__":
    main()
