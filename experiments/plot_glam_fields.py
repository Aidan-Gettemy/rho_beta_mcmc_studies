"""Field figures of a GLAM run, from the files written by experiments/glam_postprocess.py (no MPI, no solver):

    python experiments/plot_glam_fields.py --exp-tag run1
    python experiments/plot_glam_fields.py --exp-tag run1 --tracer-layer 23 --conc-layers 1 12 23 34 45

Needs <results>/post/field_stats.npz and best_fields.npz (glam_postprocess.py --fields); each figure whose input is
missing is skipped.  Reference: log of external/data/Kref.dat (EXT_REF_FIELD) for the permeability, Syn_2mm_TC.dat
(EXT_DATA_CONC) for the concentration; grid, size and output times come from the solver cfg (EXT_CFG).
All log-permeability figures show ln K [m^2] (= log beta(rho) + rho Y_P of the posterior samples).

Figures (figures/<study>/fields/):
  logperm_slice_z<layer>.png      reference | best | posterior mean | posterior std, one z-layer (5 depths)
  logperm_layers_{ref,best,mean,std}.png   one field over many z-layers
  tracer_{ref,best,mean,std}.png  concentration at the --tracer-layer, one panel per output time
  conc_vs_time.png                plug-averaged concentration: reference, best, posterior mean +- 2 sd
  conc_z_vs_time.png              disc-averaged concentration of selected z-layers: posterior mean +- 2 sd (band over the
                                  samples of the per-sample disc average), reference (circles), best (dashed)
"""
import _common  # noqa: F401
import argparse
import os

import numpy as np

from brho import io
from brho.darcy_external import parse_cfg, physics_from_cfg
from brho.glam import fieldplots as fp
from brho.glam import post


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--cfg", default=os.environ.get("EXT_CFG", "external/DATA.cfg"))
    ap.add_argument("--ref-field", default=os.environ.get("EXT_REF_FIELD", "external/data/Kref.dat"),
                    help="reference permeability K [m^2] (log is taken); --ref-is-log if the file already holds log K")
    ap.add_argument("--ref-is-log", action="store_true")
    ap.add_argument("--ref-conc", default=os.environ.get("EXT_DATA_CONC", "external/data/Syn_2mm_TC.dat"))
    ap.add_argument("--slice-fractions", type=float, nargs="+", default=[0.025, 0.25, 0.5, 0.75, 0.975])
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="z-layers (1-based) of the multi-layer figures")
    ap.add_argument("--tracer-layer", type=int, default=None, help="z-layer of the tracer maps (default: middle)")
    ap.add_argument("--conc-layers", type=int, nargs="+", default=None, help="z-layers of conc_z_vs_time")
    ap.add_argument("--nsd", type=float, default=2.0)
    args = ap.parse_args()

    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")
    pdir = os.path.join(io.results_dir(exp), "post")
    odir = os.path.join(io.figures_dir(exp), "fields")
    os.makedirs(odir, exist_ok=True)

    ph = physics_from_cfg(args.cfg)
    nx, ny, nz = ph["nx"], ph["ny"], ph["nz"]
    grid = (nx, ny, nz)
    sc = 1e3 if ph["Lx"] < 1 else 1.0                  # cfg lengths in m -> mm
    dom = (ph["Lx"] * sc, ph["Ly"] * sc)
    z_mm = (np.arange(nz) + 0.5) * ph["Lz"] * sc / nz
    times = np.array([float(t) for t in parse_cfg(args.cfg)["*OUTPUT_TIMES_FOR_SATURATION"][0].split()])
    mask = post.plug_mask(*grid)
    mid = nz // 2 + 1
    tl = args.tracer_layer or mid
    layers = args.layers or sorted({min(nz, max(1, k)) for k in (1, 5, 10, 15, 20, 25, 30, 35, 40, 45)})

    ref_logk = ref_tc = None
    if os.path.exists(args.ref_field):
        k = np.loadtxt(args.ref_field)
        ref_logk = k if args.ref_is_log else np.log(k)
    else:
        print(f"[fields] reference field {args.ref_field} not found: reference panels skipped")
    if os.path.exists(args.ref_conc):
        ref_tc = np.loadtxt(args.ref_conc)
    else:
        print(f"[fields] reference concentration {args.ref_conc} not found: reference curves skipped")

    best = np.load(os.path.join(pdir, "best_fields.npz")) if os.path.exists(os.path.join(pdir, "best_fields.npz")) else None
    stats = np.load(os.path.join(pdir, "field_stats.npz")) if os.path.exists(os.path.join(pdir, "field_stats.npz")) else None
    if best is None and stats is None:
        raise SystemExit(f"neither best_fields.npz nor field_stats.npz in {pdir}: run glam_postprocess.py --fields first")
    b_logk = best["logperm"] if best is not None else None
    b_conc = best["conc"] if best is not None else None
    m_logk = stats["logperm_mean"] if stats is not None else None
    s_logk = stats["logperm_std"] if stats is not None else None
    m_conc = stats["conc_mean"] if stats is not None else None
    s_conc = stats["conc_std"] if stats is not None else None
    made = []

    for frac in args.slice_fractions:
        k = min(nz, max(1, int(round(frac * nz))))
        made.append(fp.plot_logperm_slice_stat(ref_logk, b_logk, m_logk, s_logk, grid, k - 1,
                                               os.path.join(odir, f"logperm_slice_z{k:02d}.png"), dom))
    for name, v in (("ref", ref_logk), ("best", b_logk), ("mean", m_logk), ("std", s_logk)):
        if v is not None:
            made.append(fp.plot_slices(v, grid, layers, os.path.join(odir, f"logperm_layers_{name}.png"), dom,
                                       cmap="plasma" if name == "std" else "jet",
                                       label="std of ln K" if name == "std" else "ln K [m$^2$]",
                                       title={"ref": "Reference", "best": "Best state", "mean": "Posterior mean",
                                              "std": "Posterior standard deviation"}[name] + " log permeability"))
    shared = [x for x in (ref_tc, b_conc, m_conc) if x is not None]
    clim = (min(float(x.min()) for x in shared), max(float(x.max()) for x in shared)) if shared else None
    for name, S in (("ref", ref_tc), ("best", b_conc), ("mean", m_conc), ("std", s_conc)):
        if S is not None:
            made.append(fp.plot_tracer_maps(S, grid, times, tl, os.path.join(odir, f"tracer_{name}.png"), dom,
                                            cmap="plasma" if name == "std" else "redblue",
                                            label="std of concentration" if name == "std" else "Tracer concentration",
                                            title={"ref": "Reference", "best": "Best state", "mean": "Posterior mean",
                                                   "std": "Posterior standard deviation"}[name] + f" concentration, z-layer {tl}",
                                            clim=None if name == "std" else clim))
    ref_plug = fp.plug_average(ref_tc, mask, grid).mean(axis=0) if ref_tc is not None else None
    best_z = fp.plug_average(b_conc, mask, grid) if b_conc is not None else None
    ref_z = fp.plug_average(ref_tc, mask, grid) if ref_tc is not None else None
    if stats is not None and "conc_plug_mean" in stats.files:
        made.append(fp.plot_conc_vs_time(times, ref_plug, best_z.mean(axis=0) if best_z is not None else None,
                                         stats["conc_plug_mean"], stats["conc_plug_std"],
                                         os.path.join(odir, "conc_vs_time.png"), args.nsd))
        cl = args.conc_layers or sorted({1, max(1, round(0.25 * nz)), mid, max(1, round(0.75 * nz)), nz})
        made.append(fp.plot_conc_z_vs_time(times, z_mm, stats["conc_z_mean"], stats["conc_z_std"], cl,
                                           os.path.join(odir, "conc_z_vs_time.png"), ref_z, best_z, args.nsd))
    else:
        print("[fields] conc_z / conc_plug statistics missing in field_stats.npz: re-run glam_postprocess.py --fields")
    for p in made:
        print("saved", p)


if __name__ == "__main__":
    main()
