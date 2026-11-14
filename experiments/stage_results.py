"""Stage the SMALL results of one or more GLAM runs for git, and nothing else.

    python experiments/stage_results.py s80k_glam1 s80k_glamN --dry-run     # what would be staged, with sizes
    python experiments/stage_results.py s80k_glam1 s80k_glamN               # compress the figures, then `git add --sparse -f` the files
    git commit -m "..." && git push -u origin <branch>

What is staged, per run tag (folders tables/e6_glam_<tag> and figures/e6_glam_<tag> under the results root):
  tables    every file below --max-mb (the csv / json tables the comparison needs: ~50 KB in all)
  figures   --figures core (default): traces, MPSRF/PSRF, RE and acceptance traces, and the mean / std / z-score / mean-minus-reference
                                      log K layer figures, concentration versus time;   all: every figure;   none: tables only
            Each figure is staged as a 128-colour PNG in figures/e6_glam_<tag>/small/ (about 2-4x smaller; the original is left alone);
            --no-compress stages the originals.
NEVER staged: the results/ folder (chains, anchor_state, post/ samples and exports: hundreds of MB), logs, anything above --max-mb.
git add --force on a whole results/ folder would pull in the ignored raw chains, which is why this script adds files one by one.
"""
import _common  # noqa: F401
import argparse
import fnmatch
import os
import subprocess

from brho import io

CORE = ["traces_overview.png", "traces_phi.png", "mpsrf/*.png", "diagnostics/re_traces.png", "diagnostics/acceptance_traces.png",
        "fields/logperm_layers_mean.png", "fields/logperm_layers_std.png", "fields/logperm_layers_zscore.png",
        "fields/logperm_layers_mean_minus_ref.png", "fields/conc_vs_time.png"]


def compress(src, dst, colors=128):
    from PIL import Image
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(dst, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+", help="run tags, e.g. s80k_glam1 (results/e6_glam_<tag>)")
    ap.add_argument("--figures", choices=["core", "all", "none"], default="core")
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--max-mb", type=float, default=3.0, help="never stage a file above this size")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=None, help="BRHO_ROOT (default: $BRHO_ROOT, else runs/cluster if it exists)")
    args = ap.parse_args()
    if args.root:
        os.environ["BRHO_ROOT"] = args.root
    elif "BRHO_ROOT" not in os.environ and os.path.isdir(os.path.join(io.REPO, "runs", "cluster")):
        os.environ["BRHO_ROOT"] = os.path.join(io.REPO, "runs", "cluster")
    limit = args.max_mb * 1e6
    stage, skipped, total = [], [], 0
    for tag in args.tags:
        exp = io.EXP_GLAM + f"_{tag}"
        td, fd = io.tables_dir(exp), io.figures_dir(exp)
        for root, _, files in os.walk(td):
            for f in sorted(files):
                p = os.path.join(root, f)
                (stage if os.path.getsize(p) <= limit else skipped).append(p)
        if args.figures == "none" or not os.path.isdir(fd):
            continue
        for root, _, files in os.walk(fd):
            if os.path.basename(root) == "small":
                continue
            for f in sorted(files):
                if not f.lower().endswith(".png"):
                    continue
                p = os.path.join(root, f)
                rel = os.path.relpath(p, fd).replace(os.sep, "/")
                if args.figures == "core" and not any(fnmatch.fnmatch(rel, pat) for pat in CORE):
                    continue
                if args.no_compress:
                    q = p
                else:
                    q = os.path.join(os.path.dirname(p), "small", f)
                    if not args.dry_run:
                        compress(p, q, args.colors)
                    elif not os.path.exists(q):
                        q = p                                            # size shown is the original's
                if os.path.exists(q) and os.path.getsize(q) > limit:
                    skipped.append(q)
                else:
                    stage.append(q)
    for p in stage:
        total += os.path.getsize(p)
        print(f"{os.path.getsize(p) / 1e3:9.0f} KB  {p}")
    for p in skipped:
        print(f"  SKIPPED (above {args.max_mb} MB)  {p}")
    print(f"{len(stage)} files, {total / 1e6:.1f} MB in all" + ("   (dry run: nothing staged)" if args.dry_run else ""))
    if args.dry_run or not stage:
        return
    for i in range(0, len(stage), 100):
        subprocess.run(["git", "add", "--sparse", "-f", "--"] + stage[i:i + 100], check=True, cwd=io.REPO)
    print("staged.  Check with:  git status --short | grep -v '^??'   and   du -sh .git   then commit and push.")


if __name__ == "__main__":
    main()
