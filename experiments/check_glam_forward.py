"""Real-world check of the GLAM forward model (solver + tracer + likelihood), before any long run.

  mpiexec -n 2 python experiments/check_glam_forward.py --ref-field external/data/Kref.dat --phi-mode gpu ...

Per rank it (1) evaluates the REFERENCE field (the permeability that generated the data): the pressure drop and the
concentrations must reproduce the data, so Phi should be tiny (the data are noise-free or nearly); (2) evaluates a few
prior draws and prints the cost of each stage (assemble, projection, solver, tracer, likelihood).  Run it with 2+
ranks under the MPI launcher to also test the nested launch of the solver from MPI ranks, and on a GPU node.
Accepts the same flags as run_glam.py.
"""
import _common  # noqa: F401
import os
import time

import numpy as np

import run_glam
from brho.glam.comm import world
from brho.glam_likelihood import phi_like


def main():
    ap = run_glam.build_parser(__doc__)
    ap.add_argument("--ref-field", default=os.environ.get("EXT_REF_FIELD", "external/data/Kref.dat"))
    ap.add_argument("--n-draws", type=int, default=3)
    args = ap.parse_args()
    from brho import config as C
    cfg = C.from_cli(args, C.GlamConfig, quick=False)
    comm = world()
    rank = comm.Get_rank()
    t0 = time.time()
    model = run_glam.build_model(args, cfg, rank)
    t_build = time.time() - t0
    lines = [f"rank {rank}: model built in {t_build:.1f} s"]

    def evaluate(K, label, solver_only=False):
        t = time.time()
        res = model.runner.solve(K, tag=f"check_r{rank}", pressure=False, velocity=True)
        t_solve = time.time() - t
        if res is None:
            lines.append(f"rank {rank} {label}: SOLVER FAILED")
            return
        t = time.time()
        conc = model.tracer(res["velocity_file"])
        t_trace = time.time() - t
        phi, (pc, pd) = phi_like(model.data, conc, res["dp"], cfg.sigma_conc, cfg.sigma_dp)
        lines.append(f"rank {rank} {label}: dp {res['dp']:.2f} Pa (data {model.data.dp_pa:.2f}), "
                     f"Phi {phi:.4g} (conc {pc:.4g}, dp {pd:.4g}), conc in [{conc.min():.3f}, {conc.max():.3f}], "
                     f"solver {t_solve:.2f} s, tracer {t_trace:.2f} s")

    Kref = np.loadtxt(args.ref_field)
    evaluate(Kref, "reference field")
    rng = np.random.default_rng(1000 + rank)
    for i in range(args.n_draws):
        st = model.sample_state(rng, model.block_dims)
        t = time.time(); field = model.build(st); t_build_field = time.time() - t
        K = field.simulation_field
        lines.append(f"rank {rank} draw {i}: rho {float(field.prior_field['rho']):.3f}, assemble+project {t_build_field:.2f} s, "
                     f"log K in [{np.log(K).min():.2f}, {np.log(K).max():.2f}]")
        evaluate(K, f"draw {i}")
    for ln in comm.gather("\n".join(lines), root=0) or []:
        print(ln, flush=True)
    comm.Barrier()


if __name__ == "__main__":
    main()
