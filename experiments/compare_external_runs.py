"""
Compare two external-solver curve runs made with the SAME seed and realizations but different solver
settings (for example --rtol 1e-8 against --rtol 1e-11).  Realization i has the same field in both
runs, so the ratio of keff values isolates the solver-accuracy effect.

    python experiments/compare_external_runs.py e3_external e3_external_tight

(arguments are study names under results/; the runs are made with --exp-tag to keep them apart)
"""
import sys

import _common  # noqa: F401
import numpy as np

from brho import io


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a = np.load(f"{io.results_dir(sys.argv[1])}/curve_external.npz")
    b = np.load(f"{io.results_dir(sys.argv[2])}/curve_external.npz")
    assert np.allclose(a["rhos"], b["rhos"]) and a["keff"].shape == b["keff"].shape, "runs are not comparable"
    ratio = b["keff"] / a["keff"]
    K0 = float(a["K0"])
    print(f"{'rho':>6} {'median ratio':>14} {'max |ratio-1|':>15} {'mean log(keff_A/K0)':>21} {'mean log(keff_B/K0)':>21}")
    for i, r in enumerate(a["rhos"]):
        print(f"{r:6.2f} {np.median(ratio[i]):14.6f} {np.max(np.abs(ratio[i] - 1)):15.2e} "
              f"{np.mean(np.log(a['keff'][i] / K0)):21.5f} {np.mean(np.log(b['keff'][i] / K0)):21.5f}")
    print("max |ratio-1| should be far below the MC standard error of the mean (1e-2 or larger); "
          "if it is comparable, the solver tolerance is limiting the curve.")


if __name__ == "__main__":
    main()
