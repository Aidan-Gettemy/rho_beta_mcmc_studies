"""Export the best sample (and the reference and posterior statistics) for ParaView, from the files written by
experiments/glam_postprocess.py --fields (no MPI, no solver):

    python experiments/export_glam_fields.py --exp-tag r12

Writes <results>/post/export/ :
  <name>.npy      3-D array of shape (nx, ny, nz), indexed [i, j, k] (x = i, y = j, z = k, cell (0,0,0) first);
                  np.load(...).ravel(order="F") is the solver's cell vector (x fastest)
  glam_fields.vtk legacy VTK (STRUCTURED_POINTS, ASCII), CELL_DATA: every array below; open in ParaView directly.
                  Spacing = cell size in mm (--units m for metres).  plug_mask = 1 inside the cylindrical core plug.
Arrays:  ref_lnK  best_lnK  best_K  mean_lnK  std_lnK  best_minus_ref  mean_minus_ref  zscore  plug_mask
         ref_conc_t<n>  best_conc_t<n>  mean_conc_t<n>  (each output time)  best_pressure
Also best_state.json: Phi, rho, beta, chain and iteration of the best sample.  Missing inputs are skipped.
"""
import _common  # noqa: F401
import argparse
import json
import os

import numpy as np

from brho import io
from brho.darcy_external import parse_cfg, physics_from_cfg
from brho.glam import post


def write_vtk(path, grid, spacing, arrays, title="GLAM fields"):
    nx, ny, nz = grid
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n%s\nASCII\nDATASET STRUCTURED_POINTS\n" % title)
        f.write("DIMENSIONS %d %d %d\nORIGIN 0 0 0\nSPACING %.9g %.9g %.9g\n" % (nx + 1, ny + 1, nz + 1, *spacing))
        f.write("CELL_DATA %d\n" % (nx * ny * nz))
        for name, v in arrays.items():
            f.write("SCALARS %s double 1\nLOOKUP_TABLE default\n" % name)
            np.savetxt(f, np.asarray(v, float).reshape(-1, order="F" if np.ndim(v) == 3 else "C"), fmt="%.9g")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--cfg", default=os.environ.get("EXT_CFG", "external/DATA.cfg"))
    ap.add_argument("--ref-field", default=os.environ.get("EXT_REF_FIELD", "external/data/Kref.dat"))
    ap.add_argument("--ref-is-log", action="store_true")
    ap.add_argument("--ref-conc", default=os.environ.get("EXT_DATA_CONC", "external/data/Syn_2mm_TC.dat"))
    ap.add_argument("--units", choices=["mm", "m"], default="mm", help="length unit of the VTK spacing")
    ap.add_argument("--no-vtk", action="store_true")
    args = ap.parse_args()

    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")
    pdir = os.path.join(io.results_dir(exp), "post")
    odir = os.path.join(pdir, "export")
    os.makedirs(odir, exist_ok=True)
    ph = physics_from_cfg(args.cfg)
    grid = (ph["nx"], ph["ny"], ph["nz"])
    sc = 1e3 if args.units == "mm" else 1.0
    spacing = (ph["Lx"] * sc / grid[0], ph["Ly"] * sc / grid[1], ph["Lz"] * sc / grid[2])
    g3 = lambda v: np.asarray(v, float).reshape(grid, order="F")      # noqa: E731
    out = {}

    def put(name, v):
        if v is not None:
            out[name] = g3(v)

    if os.path.exists(args.ref_field):
        k = np.loadtxt(args.ref_field)
        put("ref_lnK", k if args.ref_is_log else np.log(k))
    ref_tc = np.loadtxt(args.ref_conc) if os.path.exists(args.ref_conc) else None
    bf = os.path.join(pdir, "best_fields.npz")
    fs = os.path.join(pdir, "field_stats.npz")
    best = np.load(bf) if os.path.exists(bf) else None
    stats = np.load(fs) if os.path.exists(fs) else None
    if best is None and stats is None:
        raise SystemExit(f"neither best_fields.npz nor field_stats.npz in {pdir}: run glam_postprocess.py --fields first")
    if best is not None:
        put("best_lnK", best["logperm"])
        put("best_K", np.exp(np.asarray(best["logperm"], float)))
        if "pressure" in best.files and np.size(best["pressure"]) == np.prod(grid):
            put("best_pressure", best["pressure"])
        json.dump(dict(phi=float(best["phi"]), rho=float(best["rho"]), beta=float(best["beta"]), dp_pa=float(best["dp_pa"]),
                       rank=int(best["rank"]), iteration=int(best["iteration"])),
                  open(os.path.join(odir, "best_state.json"), "w"), indent=1)
    if stats is not None:
        put("mean_lnK", stats["logperm_mean"])
        put("std_lnK", stats["logperm_std"])
    if "ref_lnK" in out:
        for n in ("best", "mean"):
            if f"{n}_lnK" in out:
                out[f"{n}_minus_ref"] = out[f"{n}_lnK"] - out["ref_lnK"]
        if "std_lnK" in out and "mean_lnK" in out:
            sd = out["std_lnK"]
            out["zscore"] = np.where(sd > 0, (out["ref_lnK"] - out["mean_lnK"]) / np.where(sd > 0, sd, 1.0), 0.0)
    out["plug_mask"] = g3(post.plug_mask(*grid).astype(float))
    for tag, S in (("ref", ref_tc), ("best", best["conc"] if best is not None else None),
                   ("mean", stats["conc_mean"] if stats is not None else None)):
        if S is not None:
            S = np.asarray(S, float)
            for t in range(S.shape[1]):
                put(f"{tag}_conc_t{t + 1}", S[:, t])
    for name, v in out.items():
        np.save(os.path.join(odir, name + ".npy"), v)
    if not args.no_vtk:
        write_vtk(os.path.join(odir, "glam_fields.vtk"), grid, spacing, out)
    print(f"[export] {len(out)} arrays of shape {grid} in {odir}" + ("" if args.no_vtk else " (+ glam_fields.vtk)"))


if __name__ == "__main__":
    main()
