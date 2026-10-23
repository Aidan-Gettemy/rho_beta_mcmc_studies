"""Replot MPSRF / PSRF curves from the CSV tables written by glam_mpsrf.py.  Needs only numpy and matplotlib, no chains
and no brho, so it runs anywhere the two CSV files have been copied to.

    python replot_glam_mpsrf.py --curves mpsrf_curves.csv --psrf psrf_max.csv --out-dir figs
    python replot_glam_mpsrf.py --curves a/mpsrf_curves.csv b/mpsrf_curves.csv --linear --level 1.1

Several --curves files are overlaid (columns are labelled "<file stem>:<spec>").  The CSV format: '# key: value'
comment lines (settings), one header row ('iteration' then one column per curve), one row per curve point.
"""
import argparse
import os

import numpy as np


def read_table(path):
    """-> (meta dict, column names, dict name -> float array)."""
    meta, rows, names = {}, [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition(": ")      # labels such as "first:100" hold colons
                meta[k.strip()] = v.strip()
            elif names is None:
                names = [c.strip() for c in line.split(",")]
            else:
                rows.append([float(c) for c in line.split(",")])
    a = np.array(rows, float).reshape(-1, len(names))
    return meta, names, {n: a[:, i] for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curves", nargs="*", default=[], help="mpsrf_curves.csv file(s)")
    ap.add_argument("--psrf", nargs="*", default=[], help="psrf_max.csv file(s)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--level", type=float, default=None, help="convergence level line (default: the file's, else 1.2)")
    ap.add_argument("--linear", action="store_true", help="linear y axis (default log)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out_dir, exist_ok=True)
    wrote = []
    if args.curves:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        level = args.level
        for path in args.curves:
            meta, names, d = read_table(path)
            level = level or float(meta.get("level", 1.2))
            stem = os.path.splitext(os.path.basename(path))[0] if len(args.curves) > 1 else ""
            for n in names[1:]:
                ok = np.isfinite(d[n])
                ax.plot(d["iteration"][ok], np.abs(d[n][ok]), lw=2, label=(stem + ":" if stem else "") + n
                        + (f" (dim {meta['dim[' + n + ']']})" if f"dim[{n}]" in meta else ""))
        ax.axhline(1.0, color="r", lw=2); ax.axhline(level or 1.2, color="r", ls=":", lw=2)
        ax.set_xlabel("iteration"); ax.set_ylabel("MPSRF"); ax.grid(alpha=0.3, which="both")
        if not args.linear:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8.5); ax.set_title("MPSRF versus iteration")
        fig.tight_layout(); fp = os.path.join(args.out_dir, "mpsrf_curves.png"); fig.savefig(fp, dpi=150); plt.close(fig)
        wrote.append(fp)
    if args.psrf:
        fig, ax = plt.subplots(figsize=(9, 4))
        level = args.level
        for path in args.psrf:
            meta, names, d = read_table(path)
            level = level or float(meta.get("level", 1.2))
            stem = os.path.splitext(os.path.basename(path))[0] if len(args.psrf) > 1 else ""
            for n, ls in (("max", "-"), ("p99", "--"), ("median", ":")):
                ok = np.isfinite(d[n])
                ax.plot(d["iteration"][ok], d[n][ok], ls, lw=1.8, label=(stem + ":" if stem else "") + n)
        ax.axhline(1.1, color="r", ls=":"); ax.axhline(level or 1.2, color="r", ls="--")
        ax.set_xlabel("iteration"); ax.set_ylabel("PSRF over all coordinates"); ax.grid(alpha=0.3, which="both")
        if not args.linear:
            ax.set_yscale("log")
        ax.legend(frameon=False)
        fig.tight_layout(); fp = os.path.join(args.out_dir, "psrf_max.png"); fig.savefig(fp, dpi=150); plt.close(fig)
        wrote.append(fp)
    print("wrote", ", ".join(wrote) if wrote else "nothing (give --curves and/or --psrf)")


if __name__ == "__main__":
    main()
