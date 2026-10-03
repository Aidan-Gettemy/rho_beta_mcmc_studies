# Linearized (Gaussian) posterior: analysis scripts (2026-10-02)

Report: *The linearized (Gaussian) posterior: a cheap lens on the permeability
inversion* (Overleaf zip shared 2026-10-02).  Run from this folder; the
intermediate .npz / .npy files are written here and git-ignored.

| script | what | cost |
|---|---|---|
| `lin.py` | Jacobian at the truth (central differences, 2r+1 solves) for the main 2D study (r = 57) and the scaling study (r = 208); d_eff, predicted posterior Phi, truth's informed energy and predicted rho | ~1 min |
| `sim.py global|2x2|3x3` | global / block pCN on the Gaussian surrogate (r = 208, beta and rho fixed), IACT per theta-solve | ~1.5-3 min each (needs lin.py first) |
| `visit.py` | 3x3 with visits weighted ~1/s^2, and a second global seed | ~5 min (needs sim.py 3x3) |
| `visit_adaptive.py` | importance-weighted random-scan visiting of the blocks (6 rules, 2x2 and 3x3) vs global pCN | ~25 min |
| `screen.py` | informed energy of truth seeds 2000-2099 (same truth recipe, same frozen basis) | ~1 min |
| `da_test.py` | delayed acceptance (Christen & Fox) with the surrogate linearized at the truth: global pCN, both studies | ~1 min |
| `da_eval.py` | in the typical set: true vs surrogate misfit changes for surrogates linearized at the truth, at the chain state, and at the chain state with the level treated exactly; accepted moves per full solve | ~3 min (r = 57) |

Delayed-acceptance finding (2026-10-02, r = 57): a surrogate linearized once at
the truth is counter-productive inside the posterior (correlation with the true
misfit change 0.14; 0.03-0.06 accepted moves per full solve vs 0.24 for plain
pCN).  Re-linearized at the chain state: 0.35 per solve right after the refresh
(about 1.4x), before paying 2r+1 solves per refresh; the Jacobian goes stale as
the chain drifts O(1) in the prior-dominated directions.  Making the level move
exact (p(e^a K) = e^-a p(K)) does not help.  A coarse-grid or multiscale screen
is the better delayed-acceptance surrogate here; the linearization is best used
for analysis.

## Importance-weighted visiting (2026-10-03, `visit_adaptive.py`)

IACT per theta-solve, r = 208 surrogate. Visit probabilities and steps are
learned in burn-in and then frozen (a fixed random scan, so the kernel is exact).

| rule | 2x2 median / max | 3x3 median / max |
|---|---|---|
| global pCN | 13.2k / 21.0k | (same) |
| uniform | 12.9k / 24.7k | 12.4k / 24.2k |
| info (data-informed dim of block) | 11.5k / 20.9k | 10.7k / 19.5k |
| dim (block dimension) | 11.3k / 20.4k | 10.4k / 19.0k |
| 1/s | 11.3k / 19.9k | 10.7k / 20.2k |
| 1/s^2 | 13.1k / 20.2k | 12.1k / 19.9k |
| dim/s^2 | 12.9k / 21.4k | 15.9k / 26.1k |

- Moderate weighting helps by 1.15-1.2x on the median over uniform visiting.
- The slowest direction, which sets the MPSRF, gains only 1.0-1.1x over global pCN.
- Weighting by block size alone ("dim") does as well as the data-informed rule. So the gain is mostly a fix of uniform visiting, which under-visits large blocks, not data importance.
- Aggressive weighting (1/s^2, dim/s^2) starves blocks and is worse.
- Conclusion: this does not close the 1.55x gap to E5 with theta-repeat 14.
