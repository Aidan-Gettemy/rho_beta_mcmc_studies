"""Post-processing of a GLAM run (experiments/run_glam.py).

  python experiments/glam_postprocess.py --exp-tag run1 --stride 50                       # samples only, no MPI/solver
  mpirun -np 64 python experiments/glam_postprocess.py --exp-tag run1 --stride 50 --fields --forward external

Samples: for every chain, the state at iterations window1, window1 + stride, ... (the tail starts where the step-size
adaptation ends; rejected moves count with their repetitions).  Written to results/e6_glam_<tag>/post/:
  samples.npz   theta (n, 6000, float32 unless --float64), w_rho, rho, beta = PCHIP(rho) [m^2], Phi, rank, iteration
  samples_summary.csv   quantiles of rho, beta, Phi
  field_stats.npz       (--fields) pointwise mean and std of pressure, concentration (cells x times) and log K
                        (= log beta(rho) + rho Y_P), plus the pressure drop; obtained by re-running the forward model on
                        every strided sample, the samples being shared among the MPI ranks.
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
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--fields", action="store_true", help="re-run the forward model on the samples (MPI)")
    ap.add_argument("--forward", choices=["external", "toy"], default="external")
    args, rest = ap.parse_known_args()
    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")
    out = io.results_dir(exp)
    man = io.load_json(os.path.join(out, "manifest.json"))
    cfgd = man["configs"]["GlamConfig"]
    margs = man["args"]
    tail = cfgd["window1"] if args.tail_start is None else args.tail_start
    comm = world()
    rank, P = comm.Get_rank(), comm.Get_size()
    pdir = os.path.join(out, "post")
    os.makedirs(pdir, exist_ok=True)
    nchains = man["ranks"]
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
        print(f"[glam-post] {len(s['w'])} strided samples from {nchains} chains (tail {tail}, stride {args.stride})", flush=True)
    comm.Barrier()
    if args.fields:
        z = np.load(os.path.join(pdir, "samples.npz"))
        s = dict(theta=np.load(os.path.join(pdir, "samples_theta.npy"), mmap_mode="r"), w=z["w"])
        import types
        cfg_ns = types.SimpleNamespace(**margs)
        from brho import config as C
        cfg = C.GlamConfig(**cfgd)
        cfg_ns.forward = args.forward
        model = run_glam.build_model(cfg_ns, cfg, rank)
        res = post.field_statistics(model, s, comm, log=print)
        if rank == 0:
            np.savez_compressed(os.path.join(pdir, "field_stats.npz"), **res)
            print(f"[glam-post] field statistics from {res['n_used']} samples ({res['n_failed']} failed)", flush=True)


if __name__ == "__main__":
    main()
