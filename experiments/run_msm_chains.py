"""
Run the chains of one experiment of the local-KLE (MSM) study (brho/msm_sampler.py,
docs/msm_local_kle_experiment.md).  One process per chain (multiprocessing), same seeds
across experiments.  Writes to <out>/ :
    chains.npz     per-chain arrays (relerr, acc, move, phi, theta, local states, s_hist)
    summary.json   settings, data summary, final step sizes, acceptance, relerr statistics

    python experiments/run_msm_chains.py --kind primal_raw --chains 64 --n-iter 100000 \
        --adapt 10000 --out runs/msm_local_kle_2026-11/exp2_primal_raw
(sequence: --n-iter 100000 --phase-k 50000.)
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from brho import msm_local as ML, msm_sampler as MS, msm_flux as MF

_G = {}


def _init(args):
    S = ML.MSMSetup(primal_local_dim=args.primal_dim)
    want_p = 58 if args.primal_dim == 4 else 64
    if S.primal.r_plus != want_p or S.dual.r_plus != 64:
        raise SystemExit(f"GUARD: primal r_+ must be {want_p} (primal_dim {args.primal_dim}) and dual r_+ 64 "
                         f"(decision of 2026-11-02); got {S.primal.r_plus} / {S.dual.r_plus}")
    _G["S"] = S
    _G["data"] = MS.Data(S, args.sigma_rel, args.data_seed, data_relerr=args.data_relerr)
    if args.flux_noise_rel is not None:                       # flux studies: add the interface-flux likelihood term
        _G["data"] = MF.FluxGlobalData(_G["data"], MF.FluxData(S, _G["data"], args.flux_noise_rel, args.flux_sigma_rel))
    _G["a"] = args


def _one(chain):
    a = _G["a"]
    t = time.time()
    o = MS.run_chain(a.kind, _G["S"], _G["data"], a.seed0, chain, a.n_iter, a.adapt,
                     a.phase_k, a.s0, a.save_every, 0.25, a.alt_L, not a.alt_no_offset)
    o["_sec"] = time.time() - t
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=MS.CHAIN_KINDS)
    ap.add_argument("--chains", type=int, default=64)
    ap.add_argument("--n-iter", type=int, default=100000)
    ap.add_argument("--adapt", type=int, default=10000)
    ap.add_argument("--phase-k", type=int, default=None)
    ap.add_argument("--s0", type=float, default=0.2)
    ap.add_argument("--primal-dim", type=int, default=4, help="local modes per primal block (study: 4)")
    ap.add_argument("--alt-L", type=int, default=1, help="kind alt: iterations per grid before switching")
    ap.add_argument("--alt-no-offset", action="store_true", help="kind alt: primal move is lift only (no frozen offset)")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--sigma-rel", type=float, default=0.003)
    ap.add_argument("--data-relerr", type=float, default=None)
    ap.add_argument("--data-seed", type=int, default=2026)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--flux-noise-rel", type=float, default=None,
                    help="flux studies (kind global only): add interface flux data with this relative noise")
    ap.add_argument("--flux-sigma-rel", type=float, default=0.08, help="likelihood level of the flux term")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers or min(a.chains, os.cpu_count()), _init, (a,)) as pool:
        res = pool.map(_one, range(a.chains))
    keys = [k for k in res[0] if not k.startswith("_") and k not in ("phi0", "phi1", "rel0", "rel1")]
    arr = {k: np.stack([o[k] for o in res]) for k in keys}
    for k in ("phi0", "phi1", "rel0", "rel1"):
        if k in res[0]:
            arr[k] = np.array([o[k] for o in res])
    np.savez_compressed(os.path.join(a.out, "chains.npz"), **arr)
    _init(a)
    d = _G["data"]
    n, M = a.n_iter, a.adapt
    summ = dict(settings=vars(a), data=d.summary(), wall_sec=time.time() - t0,
                sec_per_chain=float(np.mean([o["_sec"] for o in res])))
    sh = arr["s_hist"][:, :, :]
    def phase_stats(lo, hi, tag):
        acc, rel = arr["acc"][:, lo:hi], arr["relerr"][:, lo:hi]
        mv = arr["move"][:, lo:hi]
        st = {f"{tag}_acc_mean": float(acc.mean()), f"{tag}_acc_chain_min": float(acc.mean(1).min()),
              f"{tag}_acc_chain_max": float(acc.mean(1).max()),
              f"{tag}_relerr_median_of_chain_medians": float(np.median(np.median(rel, 1))),
              f"{tag}_relerr_chain_median_max": float(np.median(rel, 1).max()),
              f"{tag}_relerr_last": float(np.median(rel[:, -1]))}
        if a.kind in MS.TWO_GRID:
            for g in (0, 1):
                m = mv == g
                st[f"{tag}_acc_grid{g}"] = float((acc * m).sum() / m.sum())
        return st
    if a.kind in MS.SEQ:
        K = a.phase_k
        summ.update(phase_stats(M, K, "p1"), **phase_stats(K + M, n, "p2"))
        summ["s_final_phase1"] = sh[:, K // a.save_every - 1].ravel().tolist()
        summ["s_final_phase2"] = sh[:, -1].ravel().tolist()
        summ["phi_before_switch_median"] = float(np.median(arr["phi0"]))
        summ["phi_after_switch_median"] = float(np.median(arr["phi1"]))
        summ["relerr_before_switch_median"] = float(np.median(arr["rel0"]))
        summ["relerr_after_switch_median"] = float(np.median(arr["rel1"]))
    else:
        summ.update(phase_stats(M, n, "post"))
        summ["s_final_median"] = np.median(sh[:, -1], 0).tolist()
        summ["s_final_min"] = sh[:, -1].min(0).tolist()
        summ["s_final_max"] = sh[:, -1].max(0).tolist()
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("settings",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
