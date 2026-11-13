"""
Run the chains of one decomposition of the domain-decomposed (DD-MCMC, spline interface) study
(brho/msm_dd.py, docs/dd_mcmc_experiment.md).  Every (block, chain) pair is an independent pCN chain
(multiprocessing, one task each); the blocks of one grid are independent because the interface pressure
(a thin-plate spline through all sensor data) is frozen.  Same data and truth as the MSM study (MS.Data).
Writes to <out>/ :
    chains.npz    c_<kk> (chains, n_save, r_k)  local states every --save-every iterations, block kk
                  acc_<kk> (chains, n_iter) uint8, relerr_<kk> (chains, n_iter) float32 (local relative error)
                  s_hist_<kk> (chains, n_save)
                  theta (chains, n_save, r): the GLOBAL state, theta = B c_all assembled from chain j of every
                  block (blocks are independent, so pairing by chain index is arbitrary; derived, not sampled jointly)
    summary.json  settings, data summary, spline settings, interface errors against the truth (eps_int per segment),
                  per-block acceptance / step size / relative error

    python experiments/run_dd_chains.py --grid primal --chains 64 --n-iter 50000 --adapt 5000 \
        --out runs/msm_local_kle_2026-11/exp7_dd_primal
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from brho import msm_local as ML, msm_sampler as MS, msm_dd as DD, msm_flux as MF, msm_dd_overlap as OV, msm_dd_gibbs as GB

GRID_TAG = {"primal": 0, "dual": 1}
_G = {}


def build(args):
    """Setup, data, spline and the DDGrid of the chosen decomposition (guards as in the MSM study)."""
    pdim, den = getattr(args, "primal_local_dim", 6), getattr(args, "dual_energy", 0.95)
    S = ML.MSMSetup(primal_local_dim=pdim, dual_energy=den)
    if (pdim, den) == (6, 0.95) and (S.primal.r_k != [6] * 16 or S.primal.r_plus != 64 or S.dual.r_plus != 64 or S.dual.p != 114):
        raise SystemExit(f"GUARD: expected 16 primal blocks of 6 modes (95% rule), r_+ 64 / 64 and dual p 114; got "
                         f"{S.primal.r_k}, {S.primal.r_plus}, {S.dual.r_plus}, {S.dual.p}")
    data = MS.Data(S, args.sigma_rel, args.data_seed, data_relerr=args.data_relerr)
    spline = DD.InterfaceSpline(S, data, args.smoothing, args.neighbors)
    ms = S.primal if args.grid == "primal" else S.dual
    flux = None
    if getattr(args, "flux_noise_rel", None) is not None:           # flux studies (docs/dd_flux_experiment.md)
        flux = MF.FluxData(S, data, args.flux_noise_rel, args.flux_sigma_rel)
    halo = getattr(args, "overlap_halo", 0) or 0
    if halo:                                                          # overlapping subdomains (docs/dd_overlap_robin_experiment.md)
        ms = OV.OverlapDecomp(S, ms, halo)
    alpha = getattr(args, "robin_alpha", None)
    if alpha is not None and flux is None:
        raise SystemExit("--robin-alpha needs flux data (--flux-noise-rel)")
    if getattr(args, "h_prior_rel", None) is not None and alpha is None:
        raise SystemExit("--h-prior-rel needs --robin-alpha")
    grid = DD.DDGrid(S, ms, data, spline=spline, flux=flux, robin_alpha=alpha,
                     robin_interp=getattr(args, "robin_interp", "cubic"))
    grid.flux_data = flux
    return S, data, spline, grid


def pilot_betas(args, nb):
    """Per-block frozen step sizes from the pilot run folder args.step_from (same grid), or None."""
    if not args.step_from:
        return None
    Z = np.load(os.path.join(args.step_from, "chains.npz"))
    return [DD.pilot_step(Z[f"s_hist_{k:02d}"]) for k in range(nb)]


def _init(args):
    S, data, spline, grid = build(args)
    gbs = ([GB.GibbsBlock(b, args.h_prior_rel, args.h_corr_len, fd=grid.flux_data,
                          interior=getattr(args, "interior_flux", False), eps_meas=getattr(args, "h_prior_meas_rel", None))
            for b in grid.blocks]
           if getattr(args, 'h_prior_rel', None) is not None else None)
    _G.update(a=args, S=S, data=data, grid=grid, gbs=gbs, beta=pilot_betas(args, len(grid.blocks)))


def _one(task):
    k, chain = task
    a = _G["a"]
    t = time.time()
    beta = _G.get("beta")                                          # frozen step sizes from a pilot run (or None)
    key = [a.seed0, GRID_TAG[a.grid], k, chain] + ([1] if beta is not None else [])
    if _G.get("gbs") is not None:                                  # plain Gibbs for the Robin flux h (docs/dd_gibbs_experiment.md)
        o = GB.run_gibbs_block_chain(_G["gbs"][k], key, a.n_iter, a.adapt, a.s0 if beta is None else beta[k],
                                     a.save_every, init_scale=a.init_scale)
    else:
        o = DD.run_block_chain(_G["grid"].blocks[k], key, a.n_iter, a.adapt, a.s0 if beta is None else beta[k],
                               a.save_every, init_scale=a.init_scale)
    return k, chain, o, time.time() - t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, choices=list(GRID_TAG))
    ap.add_argument("--chains", type=int, default=64)
    ap.add_argument("--n-iter", type=int, default=50000)
    ap.add_argument("--adapt", type=int, default=5000)
    ap.add_argument("--s0", type=float, default=0.2)
    ap.add_argument("--step-from", default=None,
                    help="pilot run folder (same grid): freeze every block's step size at its pilot value (use --adapt 0)")
    ap.add_argument("--init-scale", type=float, default=1.0,
                    help="initial local coefficients = init_scale x N(0, I) (default: a prior draw)")
    ap.add_argument("--burn", type=int, default=None,
                    help="iterations discarded in the summary and in the post-processing fields (default: --adapt)")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--smoothing", type=float, default=DD.SPLINE_SMOOTHING, help="thin-plate spline smoothing")
    ap.add_argument("--neighbors", type=int, default=DD.SPLINE_NEIGHBORS)
    ap.add_argument("--sigma-rel", type=float, default=0.003)
    ap.add_argument("--data-relerr", type=float, default=None)
    ap.add_argument("--data-seed", type=int, default=2026)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--flux-noise-rel", type=float, default=None,
                    help="flux studies: add interface flux data with this relative noise (default: no flux data)")
    ap.add_argument("--flux-sigma-rel", type=float, default=0.08,
                    help="flux studies: likelihood level sigma_rel of the flux term (accommodates the ~7%% spline flux error)")
    ap.add_argument("--primal-local-dim", type=int, default=6,
                    help="local KLE modes per primal block (default 6 = the 95%% rule; the truncation floor of the DD mean error)")
    ap.add_argument("--dual-energy", type=float, default=0.95, help="local KLE energy fraction of the dual blocks (default 0.95)")
    ap.add_argument("--overlap-halo", type=int, default=0,
                    help="extend every subdomain by this many cells towards each interior interface (likelihood on the core only)")
    ap.add_argument("--robin-alpha", type=float, default=None,
                    help="Robin interface condition alpha p + K dp/dn = alpha g + h with g the spline pressure and h the flux data "
                         "(needs --flux-noise-rel); replaces the flux penalty; sides off the flux lines stay Dirichlet")
    ap.add_argument("--h-prior-rel", type=float, default=None,
                    help="plain Gibbs for the Robin flux datum h (needs --robin-alpha): prior sd of h = this x the rms face flux of the block")
    ap.add_argument("--interior-flux", action="store_true",
                    help="with --h-prior-rel: measured flux faces strictly inside a block enter as likelihood terms (weight from --flux-sigma-rel)")
    ap.add_argument("--h-prior-meas-rel", type=float, default=None,
                    help="with --h-prior-rel: prior sd factor of h on faces that carry a flux measurement (default: same as --h-prior-rel)")
    ap.add_argument("--h-corr-len", type=float, default=4.0, help="correlation length (cells) of the h prior along a side")
    ap.add_argument("--robin-interp", default="cubic", choices=["linear", "cubic"], help="interpolation of the flux along a line")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S, data, spline, grid = build(a)
    nb = len(grid.blocks)
    tasks = [(k, j) for k in range(nb) for j in range(a.chains)]
    with Pool(a.workers or min(len(tasks), os.cpu_count()), _init, (a,)) as pool:
        res = pool.map(_one, tasks, chunksize=1)
    arr, secs = {}, []
    for k in range(nb):
        rows = sorted([r for r in res if r[0] == k], key=lambda r: r[1])
        for key, name in (("c", "c"), ("acc", "acc"), ("relerr", "relerr"), ("s_hist", "s_hist")):
            arr[f"{name}_{k:02d}"] = np.stack([r[2][key] for r in rows])
        secs += [r[3] for r in rows]
    arr["theta"] = grid.theta([arr[f"c_{k:02d}"].astype(float) for k in range(nb)]).astype(np.float32)
    np.savez_compressed(os.path.join(a.out, "chains.npz"), **arr)
    # ---- summary
    n_ = S.n
    K2 = np.exp(S.kle.field(data.theta_true)).reshape(n_, n_)
    p2 = S.phys.solve(K2.ravel()).reshape(n_, n_)
    iface = grid.interface_report(K2, p2, spline)
    M = a.burn if a.burn is not None else a.adapt
    betas = pilot_betas(a, nb)
    blocks = []
    for k, b in enumerate(grid.blocks):
        acc, rel = arr[f"acc_{k:02d}"][:, M:], arr[f"relerr_{k:02d}"][:, M:]
        sh = arr[f"s_hist_{k:02d}"][:, -1]
        blocks.append(dict(block=k, cells=f"{b.nxl}x{b.nyl}", r_k=b.r_k, n_sensors=int(len(b.d_loc)),
                           acc_post_M_mean=float(acc.mean()), acc_post_M_chain_min=float(acc.mean(1).min()),
                           acc_post_M_chain_max=float(acc.mean(1).max()),
                           s_used=(float(betas[k]) if betas else None),
                           s_pilot_or_final_geomean=float(DD.pilot_step(arr[f"s_hist_{k:02d}"])),
                           s_final_median=float(np.median(sh)), s_final_min=float(sh.min()), s_final_max=float(sh.max()),
                           relerr_post_M_median=float(np.median(rel)),
                           n_flux_faces=(b.flux.n_faces if b.flux is not None else 0),
                           kinds={s: b.kinds[s] for s in DD.SIDES}))
    summ = dict(settings=vars(a), data=data.summary(),
                flux=(grid.flux_data.summary() if grid.flux_data is not None else None),
                spline=dict(kernel=DD.SPLINE_KERNEL, smoothing=a.smoothing, neighbors=a.neighbors, n_sensors=int(len(spline.xy))),
                grid=dict(name=a.grid, n_blocks=nb, r_k=grid.r_k, p=grid.p, halo=a.overlap_halo,
                          robin_alpha=a.robin_alpha, robin_sides=sum(1 for b in grid.blocks if b.robin)),
                interface=dict(segments=iface, eps_int_mean=float(np.mean([r["eps_int"] for r in iface])),
                               eps_int_max=float(np.max([r["eps_int"] for r in iface]))),
                blocks=blocks, wall_sec=time.time() - t0, sec_per_chain=float(np.mean(secs)))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("settings", "blocks")}, indent=1))
    print("acceptance post-M (block mean): min %.3f  max %.3f" % (min(b["acc_post_M_mean"] for b in blocks),
                                                                 max(b["acc_post_M_mean"] for b in blocks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
