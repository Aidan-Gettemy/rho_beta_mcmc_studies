"""
Visual and statistical check of the multiscale whitened fields used for the external-solver curve.

    python experiments/check_external_fields.py --ref-field external/data/Kref.dat \
        [--matrix-dir external/matrices --tag <tag> --whitening <npz> --phi-npy <npy>]

1. Plots slices of the saved fields (results/e3_external/fields/YP_*.npy) next to the log of the
   reference permeability field: figures/e3_external/field_check.png.
2. Prints the std of each field and the correlation between neighbouring cells along x, y, z (cells
   ordered x fastest).  For these fields the three should be high and similar (about 0.88 for
   ell = 2 cells); one much lower means the cell ordering of the matrices does not match the solver's.
3. With --whitening, draws --n-draws new fields and prints the mean per-draw std (expect near 1; about
   0.97 for the 4699-mode basis) and the spread of the per-draw means.
"""
import argparse
import glob
import os

import _common  # noqa: F401
import numpy as np

from brho import io

NX, NY, NZ = 38, 38, 45


def neighbour_corr(F):
    out = []
    for ax in (2, 1, 0):  # x, y, z
        a = np.take(F, range(F.shape[ax] - 1), axis=ax).ravel()
        b = np.take(F, range(1, F.shape[ax]), axis=ax).ravel()
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=io.EXP_CURVE_EXTERNAL)
    ap.add_argument("--ref-field", help="text file, one permeability per cell (x fastest)")
    ap.add_argument("--max", type=int, default=4, help="number of saved fields to plot")
    ap.add_argument("--matrix-dir")
    ap.add_argument("--tag")
    ap.add_argument("--n-sub", type=int, default=12)
    ap.add_argument("--whitening")
    ap.add_argument("--phi-npy")
    ap.add_argument("--n-draws", type=int, default=20)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(io.results_dir(args.exp), "fields", "YP_*.npy")))[: args.max]
    items = [(os.path.basename(f), np.load(f).astype(float)) for f in files]
    if args.ref_field:
        r = np.log(np.loadtxt(args.ref_field))
        items.append(("log Kref (centered, scaled)", (r - r.mean()) / r.std()))
    if items:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, len(items), figsize=(3.6 * len(items), 7), squeeze=False)
        for j, (name, y) in enumerate(items):
            F = y.reshape(NZ, NY, NX)
            c = neighbour_corr(F)
            print(f"{name:32s} mean {y.mean():+.3f}  std {y.std():.3f}  "
                  f"neighbour corr x/y/z = {c[0]:.2f} / {c[1]:.2f} / {c[2]:.2f}")
            im = ax[0, j].imshow(F[NZ // 2], origin="lower", cmap="viridis", vmin=-3, vmax=3)
            ax[0, j].set_title(f"{name}\nz-slice (k={NZ // 2})", fontsize=8)
            ax[1, j].imshow(F[:, NY // 2, :], origin="lower", cmap="viridis", vmin=-3, vmax=3, aspect="auto")
            ax[1, j].set_title("y-slice (z vertical)", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.6, label="unit-variance log perm")
        out = os.path.join(io.figures_dir(args.exp), "field_check.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print("saved", out)
    else:
        print("no saved fields found (run the study first, or pass --exp)")

    if args.whitening:
        from brho.multiscale_mat import load_multiscale
        F = load_multiscale(args.matrix_dir, args.tag, args.whitening, n_sub=args.n_sub, phi_npy=args.phi_npy)
        ys = [F.sample(np.random.default_rng(s)) for s in range(args.n_draws)]
        print(f"{args.n_draws} fresh draws, shape {ys[0].shape}: mean per-draw std {np.mean([y.std() for y in ys]):.3f} "
              f"(expect about 1), spread of per-draw means {np.std([y.mean() for y in ys]):.3f}")
        c = neighbour_corr(ys[0].reshape(NZ, NY, NX))
        print("neighbour corr x/y/z = %.3f / %.3f / %.3f" % tuple(c))


if __name__ == "__main__":
    main()
