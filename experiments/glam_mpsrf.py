"""MPSRF / PSRF of a GLAM run from thinned chain samples (no MPI, no solver).  Kept separate because the multivariate
statistic is expensive: one Cholesky of the dim x dim within-chain covariance per curve point.

    python experiments/glam_mpsrf.py --exp-tag run1 --estimate                        # cost estimate only
    python experiments/glam_mpsrf.py --exp-tag run1 --dims rho first:100 block:0      # MPSRF curves of three subsets
    python experiments/glam_mpsrf.py --exp-tag run1 --dims blocks:20 --stride 10 --step 20 --burn-samples 30
    python experiments/glam_mpsrf.py --exp-tag run1 --psrf-only                       # cheap: per-dimension PSRF, all dims

Samples: every --stride-th iteration of every chain from --start-iteration (default 0; use window1 to skip the
adaptation window), the full state = the subdomain thetas (blocks of block_dims.json) followed by the whitened rho
coordinate w.  Curve (the group's MATLAB convention, brho/mpsrf.py): at thinned-sample counts k = step + burn_samples,
..., the window is samples burn_samples, burn_samples + skip, ..., k.  x axis = k * stride iterations.

--dims specs (several allowed; one curve each):
  all           every coordinate (needs chains * (window - 1) > dim for W to be invertible: curve points before that are NaN)
  first:K       the first K coordinates (the group's "first 120 dims")
  blocks:K      the first K entries of every subdomain block
  block:B[:K]   subdomain block B (first K entries; default all)
  rho           the rho coordinate w alone
  random:K[:S]  K random coordinates (seed S)
--estimate prints, per spec, the number of curve points, the dimension, the singular-W cutoff and a time estimate, then
stops.  A spec whose estimate exceeds --max-minutes is skipped unless --force.

Cheap diagnostics, always computed from the diagonal only (no matrix):
  psrf_dims.png   PSRF of every coordinate at the end (colour = block), with the 1.1 / 1.2 lines
  psrf_max.png    max / 99% / median PSRF over all coordinates versus iteration
Output: figures/<study>/mpsrf/, results/<study>/post/mpsrf.npz, tables/<study>/mpsrf.json.

The chains are not independent (exchanges and restarts couple them through the anchors), so the statistic is a
mixing diagnostic, not a formal convergence proof.
"""
import _common  # noqa: F401
import argparse
import json
import os
import time

import numpy as np

from brho import io
from brho.glam import post
from brho.mpsrf import convergence_iteration, mpsrf_curve

FLOPS_PER_SECOND = 3e10        # sustained Cholesky / BLAS-3 rate assumed for the estimate (one core, conservative)


def thinned_states(res, chains, stride, start, n_iter, dtype=np.float32):
    """(m, n, dim+1) states at iterations start, start + stride, ... < n_iter; last column = w (rho coordinate)."""
    out = []
    for r in chains:
        ch = post.load_chain(res, r)
        n = min(post.n_iterations(ch), n_iter)
        t = np.arange(int(start), n, int(stride))
        j = np.searchsorted(ch["start"], t, side="right") - 1
        out.append(np.concatenate([ch["theta"][j], ch["w"][j][:, None]], axis=1).astype(dtype))
    n = min(o.shape[0] for o in out)
    return np.stack([o[:n] for o in out])


def select_dims(spec, dims, rng_seed=0):
    """-> (label, column index array) for a --dims spec; `dims` = block sizes (the rho column follows them)."""
    d_tot = int(sum(dims))
    offs = np.concatenate([[0], np.cumsum(dims)])
    head, *rest = spec.split(":")
    if head == "all":
        return "all", np.arange(d_tot + 1)
    if head == "first":
        k = int(rest[0])
        return f"first {k}", np.arange(min(k, d_tot))
    if head == "blocks":
        k = int(rest[0])
        return f"first {k} of every block", np.concatenate([offs[b] + np.arange(min(k, dims[b])) for b in range(len(dims))])
    if head == "block":
        b = int(rest[0]); k = int(rest[1]) if len(rest) > 1 else dims[b]
        return f"block {b}" + (f" (first {k})" if len(rest) > 1 else ""), offs[b] + np.arange(min(k, dims[b]))
    if head == "rho":
        return "rho (w)", np.array([d_tot])
    if head == "random":
        k = int(rest[0]); seed = int(rest[1]) if len(rest) > 1 else rng_seed
        return f"{k} random", np.sort(np.random.default_rng(seed).choice(d_tot + 1, size=min(k, d_tot + 1), replace=False))
    raise SystemExit(f"unknown --dims spec '{spec}'")


def converged_at(its, curve, level, ignore_first=10):
    """First iteration where the curve is below `level`, ignoring the first points.  As in the MATLAB read-out the
    crossing is interpolated between the previous point and that one, but only when the previous point lies above the
    level (a curve that starts below it, e.g. a single well-mixed coordinate, is reported at its first point)."""
    c = np.abs(np.asarray(curve, float)).copy()
    c[:ignore_first] = np.inf
    c[np.isnan(c)] = np.inf
    idx = np.flatnonzero(c < level)
    if idx.size == 0:
        return None
    j = int(idx[0])
    if j == 0 or not np.isfinite(curve[j - 1]) or abs(curve[j - 1]) < level:
        return float(its[j])
    return convergence_iteration(its, curve, level, ignore_first)


def curve_points(n, step, burn):
    return len(range(step + burn, n + 1, step))


def estimate(n, m, dim, step, burn, skip):
    pts = curve_points(n, step, burn)
    first_ok = None
    for k in range(step + burn, n + 1, step):
        nw = len(range(burn - 1, k, skip))
        if nw >= 2 and m * (nw - 1) > dim:
            first_ok = k
            break
    sec = pts * (dim ** 3 / 3.0 + 2.0 * dim * dim * m) / FLOPS_PER_SECOND + n * m * dim * dim * 2.0 / FLOPS_PER_SECOND
    return pts, first_ok, sec


def psrf_diag_curve(s, skip, step, burn):
    """Per-coordinate PSRF at every curve point from running sums (diagonal only): returns k (n_pts,), psrf (n_pts, dim)."""
    m, n, dim = s.shape
    ks = np.arange(step + burn, n + 1, step)
    shift = s[:, burn - 1, :].astype(float)
    S = np.zeros((m, dim)); Q = np.zeros((m, dim)); nw, nxt = 0, burn - 1
    out = np.full((len(ks), dim), np.nan)
    for u, k in enumerate(ks):
        idx = np.arange(nxt, k, skip)
        if idx.size:
            X = s[:, idx, :].astype(float) - shift[:, None, :]
            S += X.sum(1); Q += (X ** 2).sum(1)
            nw += idx.size; nxt = int(idx[-1]) + skip
        if nw >= 2:
            mu_s = S / nw
            W = ((Q - nw * mu_s ** 2) / (nw - 1)).mean(0)             # mean over chains of the within variance
            mu = mu_s + shift
            Bn = mu.var(0, ddof=1)
            V = (nw - 1) / nw * W + (m + 1) / m * Bn
            out[u] = np.sqrt(V / np.maximum(W, 1e-300))
    return ks, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--chains", type=int, nargs="+", default=None)
    ap.add_argument("--dims", nargs="+", default=["rho", "first:100"])
    ap.add_argument("--stride", type=int, default=10, help="iterations between thinned samples")
    ap.add_argument("--start-iteration", type=int, default=0)
    ap.add_argument("--skip", type=int, default=1, help="use every skip-th thinned sample inside the window")
    ap.add_argument("--step", type=int, default=10, help="thinned samples between curve points")
    ap.add_argument("--burn-samples", type=int, default=10, help="first thinned sample of every window")
    ap.add_argument("--level", type=float, default=1.2, help="convergence level for the read-out")
    ap.add_argument("--ignore-first", type=int, default=10,
                    help="curve points that cannot count as convergence (the group's MATLAB read-out ignores the first 10; "
                         "lower it when the curve has few points, e.g. a large --step)")
    ap.add_argument("--estimate", action="store_true", help="print the cost of every spec and stop")
    ap.add_argument("--psrf-only", action="store_true", help="only the per-dimension (diagonal) PSRF diagnostics")
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")
    res = io.results_dir(exp)
    man = io.load_json(os.path.join(res, "manifest.json"))
    cfg = man["configs"]["GlamConfig"]
    chains = list(range(man["ranks"])) if args.chains is None else args.chains
    if len(chains) < 2:
        raise SystemExit("MPSRF needs at least 2 chains")
    bdp = os.path.join(res, "block_dims.json")
    s = thinned_states(res, chains, args.stride, args.start_iteration, cfg["n_iter"])
    m, n, dtot = s.shape
    dims = io.load_json(bdp)["block_dims"] if os.path.exists(bdp) else [(dtot - 1) // man.get("args", {}).get("toy_n_sub", 12)] * man.get("args", {}).get("toy_n_sub", 12)
    if sum(dims) != dtot - 1:
        raise SystemExit(f"block sizes {dims} do not add up to the stored theta length {dtot - 1}")
    print(f"[mpsrf] {m} chains, {n} thinned samples each (stride {args.stride}, from iteration {args.start_iteration}), "
          f"state dimension {dtot} (= {sum(dims)} theta + rho)", flush=True)
    if n < args.step + args.burn_samples:
        raise SystemExit(f"only {n} thinned samples; need more than step + burn = {args.step + args.burn_samples}")

    specs = [] if args.psrf_only else [select_dims(sp, dims) for sp in args.dims]
    plan = []
    for label, idx in specs:
        pts, first_ok, sec = estimate(n, m, len(idx), args.step, args.burn_samples, args.skip)
        ok = "never" if first_ok is None else f"from k={first_ok} (iteration {first_ok * args.stride + args.start_iteration})"
        print(f"[mpsrf] {label:28s} dim {len(idx):5d}  {pts:4d} curve points  W invertible {ok}  est. {sec / 60:.1f} min", flush=True)
        plan.append((label, idx, sec))
    if args.estimate:
        return
    odir = os.path.join(io.figures_dir(exp), "mpsrf")
    os.makedirs(odir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = dict(stride=args.stride, start_iteration=args.start_iteration, skip=args.skip, step=args.step,
               burn_samples=args.burn_samples, chains=np.asarray(chains))
    summary = dict(chains=chains, stride=args.stride, start_iteration=args.start_iteration, n_samples=n, curves={})

    # ---- cheap: per-coordinate PSRF
    ks, P = psrf_diag_curve(s, args.skip, args.step, args.burn_samples)
    its = ks * args.stride + args.start_iteration
    out.update(psrf_k=ks, psrf_iterations=its, psrf_final=P[-1])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    offs = np.concatenate([[0], np.cumsum(dims)])
    cm = plt.get_cmap("tab20")
    for b in range(len(dims)):
        ax.plot(np.arange(offs[b], offs[b + 1]), P[-1, offs[b]:offs[b + 1]], ".", ms=2.5, color=cm(b % 20),
                label=f"block {b}" if len(dims) <= 12 else None)
    ax.plot([sum(dims)], [P[-1, -1]], "k*", ms=11, label="rho (w)")
    ax.axhline(1.1, color="r", ls=":", lw=1.2); ax.axhline(args.level, color="r", ls="--", lw=1.2)
    ax.set_xlabel("coordinate index"); ax.set_ylabel("PSRF at the end")
    ax.set_title(f"PSRF of every coordinate ({m} chains, iterations {args.start_iteration}-{int(its[-1])})", fontsize=10.5)
    ax.grid(alpha=0.3); ax.legend(ncol=7, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(odir, "psrf_dims.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ok = np.isfinite(P).all(axis=1)
    ax.plot(its[ok], np.max(P[ok], 1), lw=2, label="max over coordinates")
    ax.plot(its[ok], np.percentile(P[ok], 99, axis=1), lw=1.5, label="99th percentile")
    ax.plot(its[ok], np.median(P[ok], axis=1), lw=1.5, label="median")
    ax.axhline(1.1, color="r", ls=":"); ax.axhline(args.level, color="r", ls="--")
    ax.set_xlabel("iteration"); ax.set_ylabel("PSRF"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False); ax.set_title("PSRF over all coordinates versus iteration", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(odir, "psrf_max.png"), dpi=150); plt.close(fig)
    summary["psrf_final"] = dict(max=float(np.nanmax(P[-1])), p99=float(np.nanpercentile(P[-1], 99)),
                                 median=float(np.nanmedian(P[-1])), frac_below_1p1=float(np.mean(P[-1] < 1.1)),
                                 frac_below_level=float(np.mean(P[-1] < args.level)))
    print(f"[mpsrf] final PSRF over all coordinates: max {summary['psrf_final']['max']:.3f}, 99% "
          f"{summary['psrf_final']['p99']:.3f}, median {summary['psrf_final']['median']:.3f}", flush=True)

    # ---- expensive: the multivariate curves
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, (label, idx, sec) in enumerate(plan):
        if sec / 60 > args.max_minutes and not args.force:
            print(f"[mpsrf] skip '{label}': estimated {sec / 60:.0f} min > --max-minutes {args.max_minutes:g} (use --force)")
            continue
        t0 = time.time()
        r = mpsrf_curve(s[:, :, idx], skip=args.skip, step=args.step, iter_burn=args.burn_samples,
                        dispiter=max(1, curve_points(n, args.step, args.burn_samples) // 10),
                        log=lambda msg: print("  " + msg, flush=True))
        it_k = r["k"] * args.stride + args.start_iteration
        conv = converged_at(it_k, r["mpsrf"], args.level, ignore_first=min(args.ignore_first, max(len(it_k) - 1, 0)))
        key = f"mpsrf_{j}"
        out.update({key: r["mpsrf"], f"{key}_iterations": it_k, f"{key}_psrf": r["psrf"], f"{key}_dims": idx})
        summary["curves"][label] = dict(dim=int(len(idx)), final_mpsrf=float(r["mpsrf"][-1]),
                                        convergence_iteration=conv, level=args.level, seconds=time.time() - t0)
        good = np.isfinite(r["mpsrf"])
        line, = ax.plot(it_k[good], np.abs(r["mpsrf"][good]), lw=2,
                        label=f"{label} (dim {len(idx)}): " + (f"< {args.level:g} at {int(conv)}" if conv else f"final {r['mpsrf'][-1]:.2f}"))
        if conv:
            ax.axvline(conv, color=line.get_color(), ls="--", lw=1)
        print(f"[mpsrf] {label}: final MPSRF {r['mpsrf'][-1]:.3f}, convergence iteration {conv}  ({time.time() - t0:.0f} s)", flush=True)
    ax.axhline(1.0, color="r", lw=2); ax.axhline(args.level, color="r", ls=":", lw=2)
    ax.set_xlabel("iteration"); ax.set_ylabel("MPSRF"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"MPSRF versus iteration ({m} chains, thinning {args.stride})", fontsize=11)
    if plan:
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(odir, "mpsrf_curves.png"), dpi=150); plt.close(fig)

    pdir = os.path.join(res, "post")
    os.makedirs(pdir, exist_ok=True)
    np.savez_compressed(os.path.join(pdir, "mpsrf.npz"), **out)
    tdir = io.tables_dir(exp)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "mpsrf.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[mpsrf] wrote {odir}, {pdir}/mpsrf.npz, {tdir}/mpsrf.json")


if __name__ == "__main__":
    main()
