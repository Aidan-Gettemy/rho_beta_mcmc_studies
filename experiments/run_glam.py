"""GLAM (global-local anchored multistage sampling) on top of the PCHIP beta(rho) curve.  One chain per MPI rank.

  mpirun -np 64 python experiments/run_glam.py --forward external --exp-tag run1 [GlamConfig flags]
  mpirun -np 8  python experiments/run_glam.py --forward toy --quick --exp-tag rehearsal     # no solver / tracer
  python experiments/run_glam.py --forward toy --quick                                       # one rank, no MPI needed

forward models (--forward)
  external  the external Darcy solver (pressure drop) + the GPU tracer (concentration), the group's multiscale
            fields (whitened projection) and the PCHIP table of the offline curve; data = Syn_2mm_TC.dat and
            Syn_2mm_pDrop.dat (psi).  Needs: external/solver.local (solver), $TRACER_DIR (the solver repository's
            transport/ folder, CuPy + petsc4py), external/matrices, the data files.
  toy       cheap surrogate with the same state/field structure (tests, dress rehearsal of the MPI job).

State, moves, schedule, exchanges: see brho/glam/sampler.py and brho.config.GlamConfig (every field is a --flag).
Output: results/e6_glam[_<tag>]/{manifest.json, chains/rankRRR_partPPPP.npz, anchor_state/, exchanges.pkl,
adaptation.json}.  --resume continues from the last exchange snapshot of an interrupted run.

The exchange/restart steps do not preserve detailed balance (they depend on the whole population); adaptation is
confined to window 1.  The post-processing tail therefore starts at window1 (experiments/glam_postprocess.py).
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import dataclasses
import os
import sys

import numpy as np

from brho import config as C
from brho import io
from brho.glam.comm import world
from brho.glam.model import BetaPchip
from brho.glam.sampler import load_keep_mask, run_glam


def build_model(args, cfg, rank):
    fits = args.fits_json
    beta = BetaPchip.from_fits_json(fits)
    if args.forward == "toy":
        from brho.glam.toy import ToyModel
        return ToyModel(beta, n_sub=args.toy_n_sub, m=args.toy_m, n_cells=args.toy_cells, seed=args.toy_seed,
                        sigma_conc=args.toy_sigma, sigma_dp=args.toy_sigma, mu_rho=cfg.mu_logrho, s_rho=cfg.s_logrho,
                        rho_range=(cfg.rho_min, cfg.rho_max))
    from brho.darcy_external import DarcyRunner, solver_exe
    from brho.glam.model import ExternalModel, TracerRunner
    from brho.glam_likelihood import GlamData
    from brho.multiscale_mat import load_multiscale
    field = load_multiscale(args.matrix_dir, args.tag, args.whitening, n_sub=args.n_sub,
                            phi_npy=args.phi_npy if args.phi_mode in ("mmap", "gpu") else None)
    if args.phi_mode == "gpu":                             # projection on the GPU: PHI (1.2 GB) per rank on the device
        import cupy
        from brho.glam.model import assign_gpu
        from brho.multiscale_mat import attach_gpu_projector
        assign_gpu(rank)                                   # before CuPy touches the device
        field = attach_gpu_projector(field, cupy)
    if args.phi_mode == "ram" and args.phi_npy:
        field.PHI = np.load(args.phi_npy)                  # private float32 copy per rank (1.2 GB; 64 ranks ~ 77 GB)
    runner = DarcyRunner(args.exe or solver_exe(), args.cfg, os.path.join(args.work, f"rank{rank:03d}"),
                         launcher=args.launcher, rtol=args.rtol, extra=args.solver_extra, keep_velocity=True,
                         clean_mpi_env=True)
    tracer = TracerRunner(args.cfg, args.tracer_dir, rank=rank)
    data = GlamData.load(args.conc, args.pdrop, dp_unit="psi")
    return ExternalModel(field, beta, runner, tracer, data, cfg.sigma_conc, cfg.sigma_dp, cfg.mu_logrho,
                         cfg.s_logrho, (cfg.rho_min, cfg.rho_max))


def build_parser(doc=None):
    ap = _common.parser(doc or __doc__, C.GlamConfig)
    ap.add_argument("--forward", choices=["external", "toy"], default="external")
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--restart-keep-mask", default="", help="npz (m0..m<n_sub>, True = keep the chain's own coordinate): a restart installs the "
                    "anchor only in the other coordinates (subspace transplant); empty = ordinary restarts")
    ap.add_argument("--selection-constrained-only", action="store_true", help="archive selection (distance, log-det) on the transplanted coordinates only")
    ap.add_argument("--fits-json", default=os.environ.get("GLAM_FITS", os.path.join(
        io.REPO, "runs", "juno_external_2026-10", "tables", "e3_external", "fits.json")),
        help="fits.json of the offline curve (holds the PCHIP nodes and Kexp)")
    ap.add_argument("--exe"); ap.add_argument("--cfg", default=os.environ.get("EXT_CFG", "external/DATA.cfg"))
    ap.add_argument("--launcher", default=""); ap.add_argument("--solver-extra", default="")
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--work", default=os.environ.get("GLAM_WORK", "/tmp/glam_work"), help="per-rank scratch for K.bin and solver output (local disk)")
    ap.add_argument("--matrix-dir", default=os.environ.get("EXT_MATRICES", "external/matrices"))
    ap.add_argument("--tag", default=os.environ.get("EXT_TAG", "38x38x45_38x38x45_l2.0x2.0x2.0_m500_IND"))
    ap.add_argument("--n-sub", type=int, default=12)
    ap.add_argument("--whitening", default=os.environ.get("EXT_WHITENING", "external/matrices/whitening_operator_SP1.npz"))
    ap.add_argument("--phi-npy", default=os.environ.get("EXT_PHI_NPY", "external/matrices/phi_full_f32.npy"))
    ap.add_argument("--phi-mode", choices=["ram", "mmap", "gpu"], default="ram")
    ap.add_argument("--tracer-dir", default=os.environ.get("TRACER_DIR"))
    ap.add_argument("--conc", default=os.environ.get("EXT_DATA_CONC", "external/data/Syn_2mm_TC.dat"))
    ap.add_argument("--pdrop", default=os.environ.get("EXT_REF_PDROP", "external/data/Syn_2mm_pDrop.dat"))
    ap.add_argument("--toy-n-sub", type=int, default=12); ap.add_argument("--toy-m", type=int, default=10)
    ap.add_argument("--toy-cells", type=int, default=400); ap.add_argument("--toy-seed", type=int, default=0)
    ap.add_argument("--toy-sigma", type=float, default=0.05)
    return ap


def main():
    args = build_parser().parse_args()
    cfg = C.from_cli(args, C.GlamConfig, quick=args.quick)

    comm = world()
    rank = comm.Get_rank()
    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")
    if cfg.window1 >= cfg.n_iter:
        raise SystemExit(f"window1 ({cfg.window1}) must be smaller than n_iter ({cfg.n_iter}): step-size adaptation would never "
                         "freeze and the post-processing tail would be empty. For short runs set --window1 and --window2 too.")
    out = io.results_dir(exp)
    if rank == 0:
        io.write_manifest(exp, C.to_dict(cfg), extra=dict(forward=args.forward, ranks=comm.Get_size(),
                                                          resume=args.resume, window1=cfg.window1,
                                                          tail_start=cfg.window1, args={k: v for k, v in vars(args).items()}))
        print(f"[glam] {exp}: {comm.Get_size()} ranks, forward {args.forward}, {cfg.n_iter} iterations "
              f"(window1 {cfg.window1}, window2 {cfg.window2})", flush=True)
    comm.Barrier()
    model = build_model(args, cfg, rank)
    if rank == 0:
        io.save_json(os.path.join(out, "block_dims.json"), dict(block_dims=[int(d) for d in model.block_dims]))
    keep = load_keep_mask(args.restart_keep_mask, args.n_sub) if args.restart_keep_mask else None
    run_glam(model, cfg, comm, out, resume=args.resume, keep_mask=keep, selection_constrained_only=args.selection_constrained_only)
    if rank == 0:
        print("[glam] done:", out, flush=True)


if __name__ == "__main__":
    main()
