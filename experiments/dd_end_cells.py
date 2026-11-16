#!/usr/bin/env python3
"""The dd_end study: the sheet (dd_end.xlsx) as one definition.

Frozen problem settings per correlation length (noise calibrated on the global pCN so the pressure + flux posterior sd of log k
averages ~0.25; the pressure-only cells reuse the same eps), the cell list of the sheet, and the runner flags of every cell.

    python experiments/dd_end_cells.py --list            # counts per block and the cell labels
    python experiments/dd_end_cells.py --show j-16-DD    # runner flags of one cell

Sheet labels: a..f global pCN (a,b ell 0.2; c,d ell 0.1; e,f ell 0.05; odd = pressure, even = pressure + flux);
g,h / i,j / k,l = global pCN in the MSM basis (same ell / data order), suffix -4, -16, -64 = number of primal blocks
(2x2, 4x4, 8x8); +"-DD" = domain decomposition on the primal blocks, +"-DD+" = DD on the dual blocks plus the pooled estimate.
The 8x8 cells at ell 0.2 are dropped (a block would have 2 cells).  The single overlap test is j-16-DD-H (primal, ell 0.1, 4x4, flux)."""
import argparse
import math

# ---- frozen settings (calibration: docs / experiment summary; eps = per-datum noise relative to the rms pressure and flux).
# eps: pressure + flux cells (global posterior sd of log k ~ 0.25).  eps_p: pressure-only cells (sd ~ 0.5; interpolated from three 64-chain
# pCN runs per ell, then raised ~25-45% after the converged ell 0.2 run (eps 0.0102) gave sd 0.458, not 0.5: the short calibration
# chains had not converged and read high.  Only ell 0.2 has a converged check; the official runs show where the others land).
PROBLEMS = {
    0.2:  dict(n=16, kle_energy=0.966, eps=0.235, eps_p=0.0148, n_obs=128,  n_flux=112, blocks=(2, 4),    letters=("a", "b", "g", "h")),
    0.1:  dict(n=32, kle_energy=0.966, eps=0.2,   eps_p=0.0364, n_obs=512,  n_flux=224, blocks=(2, 4, 8), letters=("c", "d", "i", "j")),
    0.05: dict(n=64, kle_energy=0.95,  eps=0.09,  eps_p=0.034, n_obs=2048, n_flux=448, blocks=(2, 4, 8), letters=("e", "f", "k", "l")),
}
DATA_SEED = 2026
ROBIN_ALPHA = 0.1
H_INTERP_ERR = 0.03       # interpolation error of the flux data along a line (relative to the rms flux)
H_CORR_LEN = 4.0          # cells
METHODS = ("pcn", "msm", "dd", "ddplus", "ddH")


def problem(ell):
    p = dict(PROBLEMS[ell]); p["ell"] = ell
    p["sigma_rel"] = p["eps"] / math.sqrt(p["n_obs"])          # pressure likelihood level (per datum noise eps), pressure + flux cells
    p["flux_sigma_rel"] = p["eps"] / math.sqrt(p["n_flux"])    # flux weight per datum, as in the calibration runs
    p["h_prior_meas_rel"] = p["eps"]                           # Gibbs prior sd of h on measured faces
    p["h_prior_rel"] = math.sqrt(p["eps"] ** 2 + H_INTERP_ERR ** 2)   # ... and on unmeasured faces
    return p


def label(ell, flux, method, nb):
    a, b, g, h = PROBLEMS[ell]["letters"]
    if method == "pcn":
        return b if flux else a
    stem = (h if flux else g) + f"-{nb * nb}"
    return stem + {"msm": "", "dd": "-DD", "ddplus": "-DD+", "ddH": "-DD-H"}[method]


def cells():
    """All cells of the sheet at one truth: dicts with ell, flux (bool), method, nb (primal blocks per side or None), label."""
    out = []
    for ell, p in PROBLEMS.items():
        for flux in (False, True):
            out.append(dict(ell=ell, flux=flux, method="pcn", nb=None, label=label(ell, flux, "pcn", None)))
            for nb in p["blocks"]:
                for m in ("msm", "dd", "ddplus"):
                    out.append(dict(ell=ell, flux=flux, method=m, nb=nb, label=label(ell, flux, m, nb)))
    out.append(dict(ell=0.1, flux=True, method="ddH", nb=4, label=label(0.1, True, "ddH", 4)))
    return out


def flags(cell, truth_seed=DATA_SEED, eps_factor=1.0):
    """(script, flag string) of the sampler runs for one cell.  The DD scripts take --grid primal|dual; the pooled estimate of
    the dual cells is formed in post-processing from the primal-and-dual runs.  Iteration, chain and output flags are added by the
    job generator.  eps_factor scales eps (the noise sweep)."""
    p = problem(cell["ell"])
    eps = (p["eps"] if cell["flux"] else p["eps_p"]) * eps_factor
    sig = eps / math.sqrt(p["n_obs"])
    data = f"--sigma-rel {sig:.8g} --data-relerr {eps:.8g} --data-seed {truth_seed}"
    if cell["flux"]:
        fs = eps / math.sqrt(p["n_flux"])
        data += f" --flux-noise-rel {eps:.8g} --flux-sigma-rel {fs:.8g}"
    m, nb = cell["method"], cell["nb"]
    kle = f"--kle-energy {p['kle_energy']}"
    if m == "pcn":
        return "run_msm_chains.py", f"--kind global --ell {cell['ell']} --grid {p['n']} {kle} {data}"
    if m == "msm":
        return "run_msm_chains.py", f"--kind primal_raw --ell {cell['ell']} --grid {p['n']} {kle} --blocks {nb} --primal-dim 0 {data}"
    grid = "dual" if m == "ddplus" else "primal"
    f = f"--grid {grid} --n {p['n']} --ell {cell['ell']} {kle} --blocks {nb} --primal-local-dim 0 --dual-energy 0.95 {data}"
    if cell["flux"]:                                           # Robin + Gibbs for the interface flux
        hr = math.sqrt(eps ** 2 + H_INTERP_ERR ** 2)           # measured faces: sd eps; unmeasured: sqrt(eps^2 + 0.03^2)
        f += f" --robin-alpha {ROBIN_ALPHA} --h-prior-rel {hr:.8g} --h-prior-meas-rel {eps:.8g} --h-corr-len {H_CORR_LEN} --interior-flux"
    if m == "ddH":
        f += f" --overlap-halo {p['n'] // (2 * nb)}"            # half a block: the halo reaches the dual lines (4 cells at 32 x 32, 4 x 4)
    return "run_dd_chains.py", f                               # pressure only: spline interface (the runner default)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", default=None, help="label of one cell")
    a = ap.parse_args()
    cs = cells()
    if a.list or not a.show:
        from collections import Counter
        print("frozen eps (pressure + flux):", {ell: PROBLEMS[ell]["eps"] for ell in PROBLEMS}, " pressure only:", {ell: PROBLEMS[ell]["eps_p"] for ell in PROBLEMS})
        print("cells per method:", dict(Counter(c["method"] for c in cs)), "total", len(cs))
        print(" ".join(c["label"] for c in cs))
    if a.show:
        for c in cs:
            if c["label"] == a.show:
                s, f = flags(c); print(c); print(s, f)


if __name__ == "__main__":
    main()
