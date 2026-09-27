"""Bootstrap shared by the experiment runners.  Import it FIRST in every
runner: it pins BLAS/OpenMP/numba threads to 1 (unless already set) before
NumPy loads.  Parallelism comes from processes and array jobs; each forward
solve is small, and threaded BLAS inside many processes oversubscribes the
node."""
import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from brho import config as C  # noqa: E402
from brho import io  # noqa: E402


def parser(desc, *cfg_classes):
    ap = argparse.ArgumentParser(description=desc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="tiny budgets for a smoke test")
    C.add_cli(ap, C.ProblemConfig)
    for cls in cfg_classes:
        C.add_cli(ap, cls)
    return ap


def configs(args, *cfg_classes):
    pc = C.from_cli(args, C.ProblemConfig)
    return (pc,) + tuple(C.from_cli(args, cls, quick=args.quick)
                         for cls in cfg_classes)


def configs_from_manifest(exp, *cfg_classes):
    """Array-job mode: the run's settings come from the manifest written by
    --setup-only, never from this job's command line."""
    man = io.read_manifest(exp)["configs"]
    return (C.from_dict(C.ProblemConfig, man["ProblemConfig"]),) + tuple(
        C.from_dict(cls, man[cls.__name__]) for cls in cfg_classes)


def kle_summary(prob):
    k = prob.kle
    return dict(method=k.method, r=k.r, energy=k.energy, total=k.total,
                **{key: k.meta[key] for key in ("nq", "quadrature", "nq_check")
                   if key in k.meta})


_RANK_VARS = (("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),   # OpenMPI mpirun
              ("PMI_RANK", "PMI_SIZE"),                           # MPICH / Intel MPI
              ("PMIX_RANK", "PMIX_SIZE"),
              ("SLURM_PROCID", "SLURM_NTASKS"))                   # srun


def task_indices(value, n_total):
    """Which chains / grid pairs this process runs.

    --chain-id 7          -> [7]
    --chain-id env        -> array job: [$SLURM_ARRAY_TASK_ID] (or PBS array index)
                             srun / mpirun: rank r of size s runs r, r+s, r+2s, ...
                             (< n_total), so 64 ranks cover 25 pairs (ranks >= 25
                             idle) or 256 chains (4 each, one after the other).
    No MPI library is needed: each rank only reads its rank from the
    environment the launcher sets.  Always start ranks with srun or mpirun;
    run directly in a batch script, a process would see rank 0 only."""
    if value is None:
        return None
    if value != "env":
        return [int(value)]
    for var in ("SLURM_ARRAY_TASK_ID", "PBS_ARRAY_INDEX", "PBS_ARRAYID"):
        if var in os.environ:
            return [int(os.environ[var])]
    for rv, sv in _RANK_VARS:
        if rv in os.environ:
            rank, size = int(os.environ[rv]), int(os.environ.get(sv, "1"))
            idx = list(range(rank, n_total, size))
            print(f"rank {rank}/{size} ({rv}): tasks {idx}", flush=True)
            return idx
    raise SystemExit("--chain-id/--pair-index env: no array index or launcher "
                     "rank in the environment (use sbatch --array, srun or mpirun)")
