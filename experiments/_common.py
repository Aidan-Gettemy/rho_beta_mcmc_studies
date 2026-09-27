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


def task_index(value):
    """--chain-id / --pair-index: an int, or 'env' for $SLURM_ARRAY_TASK_ID
    (also accepts $PBS_ARRAYID / $PBS_ARRAY_INDEX)."""
    if value is None:
        return None
    if value != "env":
        return int(value)
    for var in ("SLURM_ARRAY_TASK_ID", "PBS_ARRAY_INDEX", "PBS_ARRAYID"):
        if var in os.environ:
            return int(os.environ[var])
    raise SystemExit("--chain-id env: no array-task variable in the environment")
