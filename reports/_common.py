"""Bootstrap shared by the report scripts."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from brho import config as C, io  # noqa: E402
from brho.problem import Problem  # noqa: E402
from brho.truth import Synthetic  # noqa: E402


def have(exp, fname="manifest.json"):
    return os.path.exists(os.path.join(io.results_dir(exp), fname))


def load_problem(exp):
    """Rebuild the Problem exactly as the run saw it (from its manifest)."""
    man = io.read_manifest(exp)
    pc = C.from_dict(C.ProblemConfig, man["configs"]["ProblemConfig"])
    prob = Problem(pc)
    saved = Synthetic.load(os.path.join(io.results_dir(exp), "truth.npz"))
    assert abs(saved.keff_obs - prob.syn.keff_obs) < 1e-12, "truth mismatch"
    return prob, man
