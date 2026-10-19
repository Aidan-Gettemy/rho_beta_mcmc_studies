#!/usr/bin/env python3
"""
STAND-IN for an external Darcy solver, used only to test the driver's plumbing (file I/O, flags,
parsing, parallel runner).  It honours the solver contract of brho/darcy_external.py: it reads the
binary permeability file and writes pressure_drop.txt from a layered-series model
Keff = Lz / sum_k(dz / mean_xy K_k)   (exact for uniform K, linear in K).
It is NOT a Darcy solver; real validation must use the real executable (--selfcheck).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brho.darcy_external import physics_from_cfg  # noqa: E402


def main():
    argv = sys.argv[1:]
    cfg = argv[0]
    opts = {}
    i = 1
    while i < len(argv):
        opts[argv[i].lstrip("-")] = argv[i + 1]
        i += 2
    out = opts.get("output_dir", "./output")
    os.makedirs(out, exist_ok=True)
    p = physics_from_cfg(cfg)
    K = np.fromfile(opts["permeability_file"], dtype="<f8")
    assert K.size == p["nx"] * p["ny"] * p["nz"], "wrong file size"
    K = K.reshape(p["nz"], p["ny"], p["nx"])
    dz = p["Lz"] / p["nz"]
    keff = p["Lz"] / np.sum(dz / K.mean(axis=(1, 2)))
    dp = p["v"] * p["mu"] * p["Lz"] / keff
    with open(os.path.join(out, "pressure_drop.txt"), "w") as f:
        f.write(f"{dp:.15e}\n")
    print("[Darcy] KSP converged reason: CONVERGED_RTOL, iterations: 1")


if __name__ == "__main__":
    main()
