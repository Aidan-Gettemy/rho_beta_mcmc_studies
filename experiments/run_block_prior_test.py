"""
Block (subdomain) pCN prior test, E8 step 1 (brho/blockpcn.py): the
geometry of the local subspaces and a likelihood-OFF check that the block
sweeps keep theta ~ N(0, I_r), on the KLE of a frozen run (default e5_soft).
No PDE solves.

Geometry (exact, no sampling):
  * dimensions of the local subspaces (primal blocks^dim, dual
    (blocks+1)^dim shifted by half a block) and of the coarse complement;
  * "reach": per cell, the prior variance that the local moves can change,
    over the cell's prior variance (summed over moves: overlap counts twice);
  * one example draw of the field change made by a primal and by a dual
    block move (smooth, mostly inside the block).
Prior test (likelihood off, every proposal accepted), chains started at
direct prior draws, fixed step s for every move:
  coarse+primal         coarse, primal 1..P              (P + 1 moves / sweep)
  coarse+primal+dual    coarse, primal 1..P, dual 1..D   (E8 "block")
  random                coarse + one local block         (E8 "block_random";
                        run for as many moves as the full sweep)
Checks: cov(theta) = I_r, pointwise mean 0 and variance diag(T T^T) (exact
from the theta moments), and the lag-1 autocorrelation of G at every cell,
per sweep and per move (= per PDE solve once the likelihood is on).  With the
likelihood off a pCN move with step s shrinks what it touches by
sqrt(1 - s^2), so the lag-1 map mostly counts how often a cell is refreshed;
it is a check of the bookkeeping, not a speed measurement (E8 with the
likelihood on is).

    python experiments/run_block_prior_test.py [--from-exp e5_soft] [--blocks 2]
           [--energy 0.9] [--halo 0] [--chains 32] [--sweeps 4000] [--s 0.3]

Writes figures/block_prior_test/fig_block_geometry.png,
fig_block_prior_fields.png, fig_block_lag1.png and
tables/block_prior_test/block_prior_test.{json,md}.
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import C  # noqa: E402
from brho import io  # noqa: E402
from brho.problem import Problem  # noqa: E402
from brho.blockpcn import from_problem  # noqa: E402
from brho import plotting as P  # noqa: E402
from brho import diagnostics as D  # noqa: E402

EXP = "block_prior_test"
SCHEDULES = ("coarse+primal", "coarse+primal+dual", "random")


def moves_of(bp, schedule, rng):
    if schedule == "coarse+primal":
        return [j for j, mv in enumerate(bp.moves) if mv["kind"] in ("coarse", "primal")]
    if schedule == "coarse+primal+dual":
        return list(range(len(bp.moves)))
    if schedule == "random":
        return [0, 1 + int(rng.integers(bp.n_local))]
    raise ValueError(schedule)


def run(bp, schedule, m, sweeps, thin, s, seed):
    """m chains at once (vectorized over chains); returns thinned theta
    (m, n, r), the lag-1 moment E[theta_t theta_{t+1}^T] and E[theta theta^T]
    over all sweeps, and the number of moves per sweep."""
    rng = np.random.default_rng(seed)
    r = bp.r
    th = rng.standard_normal((m, r))                    # exact prior start
    c = np.sqrt(1.0 - s * s)
    kept, C0, C1 = [], np.zeros((r, r)), np.zeros((r, r))
    n_moves = 0
    for it in range(sweeps):
        prev = th.copy()
        for j in moves_of(bp, schedule, rng):
            V = bp.moves[j]["V"]
            if V.shape[1] == 0:
                continue
            a = th @ V
            th = th + ((c - 1.0) * a + s * rng.standard_normal(a.shape)) @ V.T
            n_moves += 1
        C0 += prev.T @ prev
        C1 += prev.T @ th
        if it % thin == 0:
            kept.append(th.copy())
    N = m * sweeps
    return np.stack(kept, 1), C0 / N, C1 / N, n_moves / sweeps


def lag1_map(T, C0, C1):
    """lag-1 autocorrelation of G_i = t_i . theta per cell (mean 0 known)."""
    C1s = 0.5 * (C1 + C1.T)
    return np.einsum("ij,jk,ik->i", T, C1s, T) / np.einsum("ij,jk,ik->i", T, C0, T)


def draw_edges(ax, blocks, dual, n, color, ls):
    """Interfaces on a field panel (P.field draws on the unit square)."""
    e = (np.arange(blocks) + 0.5) / blocks if dual else np.arange(1, blocks) / blocks
    for x in e:
        ax.axvline(x, color=color, lw=0.8, ls=ls)
        ax.axhline(x, color=color, lw=0.8, ls=ls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-exp", default=io.EXP_SOFT)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--energy", type=float, default=0.9)
    ap.add_argument("--halo", type=float, default=0.0)
    ap.add_argument("--chains", type=int, default=32)
    ap.add_argument("--sweeps", type=int, default=4000)
    ap.add_argument("--thin", type=int, default=4)
    ap.add_argument("--s", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default=None,
                    help="output folder (default block_prior_test, + _NxN for --blocks != 2)")
    a = ap.parse_args(argv)

    man = io.read_manifest(a.from_exp)
    pc = C.from_dict(C.ProblemConfig, man["configs"]["ProblemConfig"])
    prob = Problem(pc, frozen=io.results_dir(a.from_exp))
    kle, n = prob.kle, pc.n
    bp = from_problem(prob, a.blocks, a.energy, True, a.halo)
    bs = bp.summary()
    grid = "x".join([str(n)] * pc.dim)
    print(f"KLE from {a.from_exp}: {grid}, ell={pc.ell}, r={kle.r}; block pCN, energy >= "
          f"{a.energy}, halo {a.halo}: primal {bs['primal_dims']} (sum {bs['primal_total']}), "
          f"dual {bs['dual_dims']} (sum {bs['dual_total']}), coarse {bs['coarse_dim']}; "
          f"coarse + primal span R^{kle.r}: {bs['spans']}")
    out = a.out or (EXP if a.blocks == 2 else f"{EXP}_{a.blocks}x{a.blocks}")
    fdir, tdir = io.figures_dir(out), io.tables_dir(out)

    # ------------------------------------------------------------ geometry
    reach = {"primal": bp.reach(("primal",)), "primal+dual": bp.reach(("primal", "dual")),
             "coarse": bp.reach(("coarse",))}
    rng = np.random.default_rng(a.seed)
    kinds = [mv["kind"] for mv in bp.moves]
    jp = kinds.index("primal")                          # first primal block
    duals = [j for j, k in enumerate(kinds) if k == "dual"]
    jd = max(duals, key=lambda j: bp.moves[j]["dim"]) if duals else jp   # largest dual
    ex = {}
    for j in (jp, jd):
        V = bp.moves[j]["V"]
        ex[bp.moves[j]["name"]] = kle.T @ (V @ rng.standard_normal(V.shape[1]))
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.6))
    for k, (lab, R) in enumerate(reach.items()):
        P.field(ax[k], R, n, f"reach: {lab} moves\n{R.min():.2f}-{R.max():.2f} "
                f"(median {np.median(R):.2f})", cmap="viridis", vmin=0, vmax=1.2)
    vx = max(np.abs(v).max() for v in ex.values())
    for k, (name, f) in enumerate(ex.items()):
        mv = bp.moves[jp if k == 0 else jd]
        out = float((f[~mv["mask"]] ** 2).sum() / (f ** 2).sum())
        P.field(ax[3 + k], f, n, f"one {mv['kind']} move ({name}, dim {mv['dim']}):\n"
                f"field change; {100 * out:.0f}% of its energy outside the block",
                cmap="RdBu_r", vmin=-vx, vmax=vx)
    for x in ax:
        draw_edges(x, a.blocks, False, n, "k", "--")
        draw_edges(x, a.blocks, True, n, "g", ":")
    fig.suptitle(f"Block pCN geometry, {pc.dim}D ({grid}, r={kle.r}): local subspaces = "
                 f"theta-directions with >= {a.energy:.0%} of their field energy in the block; "
                 f"primal (dashed) dims {bs['primal_dims']}, dual (dotted) dims "
                 f"{bs['dual_dims']}, coarse {bs['coarse_dim']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "fig_block_geometry.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------ prior test
    res = {}
    for i, sch in enumerate(SCHEDULES):
        # the random schedule gets as many MOVES as the full sweep (more sweeps)
        k = len(bp.moves) // 2 if sch == "random" else 1
        th, C0, C1, mps = run(bp, sch, a.chains, k * a.sweeps, k * a.thin, a.s,
                              a.seed + 1 + i)
        flat = th.reshape(-1, kle.r)
        pm, pv = D.pointwise_moments(flat, kle.T)
        lag = lag1_map(kle.T, C0, C1)
        res[sch] = dict(frob=float(np.linalg.norm(np.cov(flat.T) - np.eye(kle.r))
                                   / np.sqrt(kle.r)),
                        pm=pm, pv=pv, samples=kle.field(th[:3, -1]), lag=lag,
                        lag_move=lag ** (1.0 / mps), moves_per_sweep=mps)
        print(f"  {sch:20s} cov error {res[sch]['frob']:.3f}; lag-1 per sweep "
              f"{lag.min():.3f}-{lag.max():.3f}; per move {res[sch]['lag_move'].min():.4f}-"
              f"{res[sch]['lag_move'].max():.4f} ({mps:.1f} moves / sweep)")
    ref = np.random.default_rng(a.seed + 99).standard_normal(flat.shape)
    pm_r, pv_r = D.pointwise_moments(ref, kle.T)
    frob_ref = float(np.linalg.norm(np.cov(ref.T) - np.eye(kle.r)) / np.sqrt(kle.r))
    rows = [("direct draws", kle.field(ref[:3]), pm_r, pv_r)] + [
        (k, res[k]["samples"], res[k]["pm"], res[k]["pv"]) for k in SCHEDULES]
    fig = P.prior_fields(rows, bp.target, n, a.blocks,
                         f"Block pCN prior test (likelihood off), {a.chains} chains x "
                         f"{a.sweeps} sweeps, s={a.s}; primal interfaces dashed.")
    fig.savefig(os.path.join(fdir, "fig_block_prior_fields.png"), dpi=120)
    plt.close(fig)
    ratios = fig.ratios

    fig, ax = plt.subplots(2, len(SCHEDULES), figsize=(6 * len(SCHEDULES), 9.2))
    for k, sch in enumerate(SCHEDULES):
        for row, key, what in ((0, "lag", "per sweep"), (1, "lag_move", "per move (= per solve)")):
            L = res[sch][key]
            P.field(ax[row, k], L, n, f"{sch}: lag-1 autocorr {what}\n{L.min():.3f}-"
                    f"{L.max():.3f} ({res[sch]['moves_per_sweep']:.1f} moves / sweep)",
                    cmap="magma_r")
            draw_edges(ax[row, k], a.blocks, False, n, "k", "--")
            draw_edges(ax[row, k], a.blocks, True, n, "g", ":")
    fig.suptitle(f"Lag-1 autocorrelation of G per cell, likelihood off, s={a.s} for every "
                 f"move (sqrt(1-s^2) = {np.sqrt(1 - a.s ** 2):.3f}): lower = refreshed more. "
                 "Bookkeeping check, not a speed measure.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "fig_block_lag1.png"), dpi=120)
    plt.close(fig)

    def q(v):
        return dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()))
    table = {k: dict(cov_error=res[k]["frob"], moves_per_sweep=res[k]["moves_per_sweep"],
                     pointwise_var_ratio=q(ratios[k]),
                     pointwise_mean_max_abs=float(np.abs(res[k]["pm"]).max()),
                     lag1_per_sweep=q(res[k]["lag"]), lag1_per_move=q(res[k]["lag_move"]))
             for k in SCHEDULES}
    out = dict(settings=vars(a), block=bs, reach={k: q(v) for k, v in reach.items()},
               leak=bp.leak(), cov_error_reference=frob_ref,
               pointwise_var_ratio_reference=q(ratios["direct draws"]), results=table)
    io.save_json(os.path.join(tdir, "block_prior_test.json"), out)
    with open(os.path.join(tdir, "block_prior_test.md"), "w") as f:
        f.write(f"Block pCN, {grid}, r={kle.r}, energy >= {a.energy}, halo {a.halo}: "
                f"primal dims {bs['primal_dims']}, dual dims {bs['dual_dims']}, coarse "
                f"{bs['coarse_dim']}; spans R^r: {bs['spans']}\n\n")
        f.write("| reach (prior variance the moves can change / prior variance) | min | "
                "median | max |\n|---|---|---|---|\n")
        for k, v in reach.items():
            f.write(f"| {k} | {v.min():.2f} | {np.median(v):.2f} | {v.max():.2f} |\n")
        f.write(f"\nPrior test: {a.chains} chains x {a.sweeps} sweeps, s = {a.s}, likelihood "
                f"off; direct draws: cov error {frob_ref:.3f}\n\n"
                "| schedule | moves / sweep | cov(θ) error | pointwise var / target | "
                "lag-1 per sweep | lag-1 per move |\n|---|---|---|---|---|---|\n")
        for k in SCHEDULES:
            t = table[k]
            f.write(f"| {k} | {t['moves_per_sweep']:.1f} | {t['cov_error']:.3f} | "
                    f"{t['pointwise_var_ratio']['min']:.2f}–{t['pointwise_var_ratio']['max']:.2f}"
                    f" | {t['lag1_per_sweep']['min']:.3f}–{t['lag1_per_sweep']['max']:.3f} | "
                    f"{t['lag1_per_move']['min']:.4f}–{t['lag1_per_move']['max']:.4f} |\n")
    print(open(os.path.join(tdir, "block_prior_test.md")).read())


if __name__ == "__main__":
    main()
