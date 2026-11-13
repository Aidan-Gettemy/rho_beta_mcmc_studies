"""Post-processing of a GLAM run (experiments/run_glam.py).

  python experiments/glam_postprocess.py --exp-tag run1 --stride 50                       # samples only, no MPI/solver
  mpirun -np 64 python experiments/glam_postprocess.py --exp-tag run1 --stride 50 --fields --forward external

Samples: for every chain, the state at iterations window1, window1 + stride, ... (the tail starts where the step-size
adaptation ends; rejected moves count with their repetitions).  Written to results/e6_glam_<tag>/post/:
  samples.npz   theta (n, 6000, float32 unless --float64), w_rho, rho, beta = PCHIP(rho) [m^2], Phi, rank, iteration
  samples_summary.csv   quantiles of rho, beta, Phi
  best_state.npz        the lowest-Phi state of all chains (theta, w_rho, rho, beta, Phi, rank, iteration); --best-scope
                        picks the whole run (default) or the tail only
  best_fields.npz       (--fields) forward model on the best state: logperm (= log K), conc (cells x times), pressure,
                        dp_pa, recomputed Phi
  field_stats.npz       (--fields) pointwise mean and std of pressure, concentration (cells x times) and log K
                        (= log beta(rho) + rho Y_P), plus the pressure drop; for the external forward model also
                        conc_z (per-sample concentration averaged over the plug disc of every z-layer, layers x times)
                        and conc_plug (whole plug, times), each with mean and std over the samples; obtained by re-running the forward model on
                        every strided sample, the samples being shared among the MPI ranks.  Also per-sample arrays
                        sample_index, phi, re_conc, re_dp, dp_pa (used by plot_glam_diagnostics.py when a chain has no
                        per-iteration RE trace, i.e. runs made before the trace existed).

Companions: plot_glam_diagnostics.py (RE traces, acceptance-rate histograms), glam_mpsrf.py (MPSRF from the saved
samples; expensive, run --estimate first), plot_glam_fields.py (slices, tracer maps, concentration curves).
"""
import _common  # noqa: F401
import json
import os

import numpy as np

from brho import io
from brho.glam import post
from brho.glam.comm import world
from brho.glam.model import BetaPchip


def main():
    import argparse
    import run_glam  # same folder: model construction and flags
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--tail-start", type=int, default=None, help="default: window1 from the manifest")
    ap.add_argument("--chains", type=int, nargs="+", default=None, help="chain (rank) numbers to use; default all (e.g. only the chains that finished)")
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--best-scope", choices=["all", "tail"], default="all", help="best state over the whole run or the tail")
    ap.add_argument("--fields", action="store_true", help="re-run the forward model on the samples (MPI)")
    ap.add_argument("--pressure", action="store_true",
                    help="also request the pressure field (PETSc binary file) from the solver; off by default: the figures do not "
                         "use it and that output path has not been exercised in the runs")
    ap.add_argument("--forward", choices=["external", "toy"], default="external")
    args, rest = ap.parse_known_args()
    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")
    out = io.results_dir(exp)
    man = io.load_json(os.path.join(out, "manifest.json"))
    cfgd = man["configs"]["GlamConfig"]
    margs = man["args"]
    if os.environ.get("GLAM_WORK"):                     # the manifest holds the run job's node-local scratch; use this job's
        margs["work"] = os.environ["GLAM_WORK"]
    import faulthandler, time
    faulthandler.dump_traceback_later(int(os.environ.get("POST_STALL_SECS", "900")), repeat=True)   # where is it stuck?
    t0 = time.time()
    tail = cfgd["window1"] if args.tail_start is None else args.tail_start
    comm = world()
    rank, P = comm.Get_rank(), comm.Get_size()
    pdir = os.path.join(out, "post")
    os.makedirs(pdir, exist_ok=True)
    nchains = man["ranks"] if args.chains is None else args.chains
    if rank == 0:
        s = post.strided_samples(out, nchains, tail, args.stride, cfgd["n_iter"], np.float64 if args.float64 else np.float32)
        beta = BetaPchip.from_fits_json(margs["fits_json"])
        rho = np.exp(cfgd["mu_logrho"] + cfgd["s_logrho"] * s["w"])
        s["rho"], s["beta"] = rho, beta.beta(rho)
        np.save(os.path.join(pdir, "samples_theta.npy"), s["theta"])
        np.savez(os.path.join(pdir, "samples.npz"), **{k: v for k, v in s.items() if k != "theta"}, tail_start=tail,
                 stride=args.stride)
        q = [5, 25, 50, 75, 95]
        with open(os.path.join(pdir, "samples_summary.csv"), "w") as f:
            f.write("quantity," + ",".join(f"q{x}" for x in q) + "\n")
            for name, v in (("rho", rho), ("beta_m2", s["beta"]), ("Phi", s["phi"])):
                f.write(name + "," + ",".join(f"{np.percentile(v, x):.6e}" for x in q) + "\n")
        b = post.best_record(out, nchains, 0 if args.best_scope == "all" else tail)
        b_rho = float(np.exp(cfgd["mu_logrho"] + cfgd["s_logrho"] * b["w"]))
        np.savez(os.path.join(pdir, "best_state.npz"), theta=b["theta"], w=b["w"], phi=b["phi"], rank=b["rank"],
                 iteration=b["iteration"], rho=b_rho, beta=float(beta.beta(b_rho)), scope=args.best_scope)
        print(f"[glam-post] best state: Phi {b['phi']:.4g} (chain {b['rank']}, iteration {b['iteration']}, rho {b_rho:.4f})",
              flush=True)
        print(f"[glam-post] {len(s['w'])} strided samples from {len(post.rank_list(nchains))} chains (tail {tail}, stride {args.stride})", flush=True)
    comm.Barrier()
    if args.fields:
        z = np.load(os.path.join(pdir, "samples.npz"))
        s = dict(theta=np.load(os.path.join(pdir, "samples_theta.npy"), mmap_mode="r"), w=z["w"])
        import types
        cfg_ns = types.SimpleNamespace(**margs)
        from brho import config as C
        cfg = C.GlamConfig(**cfgd)
        cfg_ns.forward = args.forward
        if rank == 0:
            print(f"[glam-post] building the forward model (work dir {margs['work']}) ...", flush=True)
        model = run_glam.build_model(cfg_ns, cfg, rank)
        comm.Barrier()
        if rank == 0:
            print(f"[glam-post] forward model built on {P} ranks in {time.time() - t0:.0f} s", flush=True)
        plug = grid = None
        if args.forward == "external":
            from brho.darcy_external import physics_from_cfg
            ph = physics_from_cfg(margs["cfg"])
            grid = (ph["nx"], ph["ny"], ph["nz"])
            plug = post.plug_mask(*grid)
        # the best state first.  Every rank solves it (the same state, in lockstep, exactly as the sampler runs its chains)
        # and only rank 0 keeps the result.  Reason: the tracer imports petsc4py lazily at its first call, and PETSc's
        # initialisation is collective over MPI_COMM_WORLD, so a first call on rank 0 alone deadlocks (seen on Juno:
        # rank 0 stuck in rwPETSc.py's import while ranks 1-11 waited in the barrier).
        bz = np.load(os.path.join(pdir, "best_state.npz"))
        if rank == 0:
            print(f"[glam-post] best state: forward solve on all {P} ranks (pressure output {'on' if args.pressure else 'off'}) ...", flush=True)
        t1 = time.time()
        bf = post.best_fields(model, dict(theta=bz["theta"], w=float(bz["w"]), phi=float(bz["phi"]),
                                          rank=int(bz["rank"]), iteration=int(bz["iteration"])), rank,
                              want_pressure=args.pressure)
        if rank == 0:
            np.savez_compressed(os.path.join(pdir, "best_fields.npz"), **bf)
            print(f"[glam-post] best fields: Phi recomputed {bf['phi']:.4g} (chain value {bf['phi_chain']:.4g}); one forward solve took {time.time() - t1:.1f} s", flush=True)
        comm.Barrier()
        res = post.field_statistics(model, s, comm, log=print, plug=plug, grid=grid, want_pressure=args.pressure)
        if rank == 0:
            np.savez_compressed(os.path.join(pdir, "field_stats.npz"), **res)
            print(f"[glam-post] field statistics from {res['n_used']} samples ({res['n_failed']} failed)", flush=True)


if __name__ == "__main__":
    main()
