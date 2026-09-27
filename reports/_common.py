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
    """The Problem exactly as the run saw it.  Runs set up by the current
    code carry their KLE basis and data (kle.npz, truth.npz) and are loaded
    from them.  Older runs are rebuilt from the manifest; their saved data
    are then used as-is after a consistency check (a config mismatch shows
    up as an O(1) difference, rounding as ~1e-12)."""
    man = io.read_manifest(exp)
    pc = C.from_dict(C.ProblemConfig, man["configs"]["ProblemConfig"])
    d = io.results_dir(exp)
    if os.path.exists(os.path.join(d, "kle.npz")):
        return Problem(pc, frozen=d), man
    prob = Problem(pc)
    saved = Synthetic.load(os.path.join(d, "truth.npz"))
    rel = abs(saved.keff_obs / prob.syn.keff_obs - 1)
    assert rel < 1e-9, f"truth mismatch ({rel:.1e}): manifest and data disagree"
    prob.syn = saved
    return prob, man
