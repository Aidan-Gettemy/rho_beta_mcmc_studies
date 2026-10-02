# Linearized (Gaussian) posterior: analysis scripts (2026-10-02)

Report: *The linearized (Gaussian) posterior: a cheap lens on the permeability
inversion* (Overleaf zip shared 2026-10-02).  Run from this folder; the
intermediate .npz / .npy files are written here and git-ignored.

| script | what | cost |
|---|---|---|
| `lin.py` | Jacobian at the truth (central differences, 2r+1 solves) for the main 2D study (r = 57) and the scaling study (r = 208); d_eff, predicted posterior Phi, truth's informed energy and predicted rho | ~1 min |
| `sim.py global|2x2|3x3` | global / block pCN on the Gaussian surrogate (r = 208, beta and rho fixed), IACT per theta-solve | ~1.5-3 min each (needs lin.py first) |
| `visit.py` | 3x3 with visits weighted ~1/s^2, and a second global seed | ~5 min (needs sim.py 3x3) |
| `screen.py` | informed energy of truth seeds 2000-2099 (same truth recipe, same frozen basis) | ~1 min |
