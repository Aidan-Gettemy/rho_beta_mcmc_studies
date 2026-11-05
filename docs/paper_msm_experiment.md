# Paper-MSM study: Gaussian MSM versus the smoothed MSM of Ali, Al-Mamun, Pereira, Rahunanthan (JCP 497, 2024)

Code: `brho/msmpaper.py`, `experiments/run_paper_{chains,post,prior}.py`, `cluster/gen_paper_sbatch.py`,
`cluster/submit_paper_msm.sh`, tests `tests/test_msmpaper.py`.

## Problem
Same as arm 1 of the grid study at 16 x 16 (sqexp kernel, correlation length 0.2, 20-mode canonical Nystrom KLE,
truth = first 20 draws of seed 2026, relative-error likelihood eps = 0.003, 128 observations). Partition 2 x 2 (p = 20),
local dimension 5 per subdomain. Local KLE of each 8 x 8 block is built in block coordinates with the canonical basis;
the 5-mode cut splits the tied pair (5,6) of the block spectrum (31.27, 11.01, 11.01, 3.88, 2.16, 2.16, ...), which the
canonical basis makes reproducible.

## Experiments
| id | field | update | iterations | adaptation M |
|---|---|---|---|---|
| e0 | global theta (control) | all 20 | 100k | 10k |
| e1 | global theta | one theta at a time | 150k | 30k |
| e2 | local modes, zero-extended, no projection (discontinuous) | all | 100k | 10k |
| e3 | projected, theta = B c | all | 100k | 10k |
| e4 | projected | one coefficient at a time | 150k | 30k |
| e5 | paper smoothing | all | 100k | 10k |
| e6 | paper smoothing | one coefficient at a time | 150k | 30k |
| e7 | paper smoothing | one subdomain at a time (its 5 coefficients), others stored | 150k | 30k |

Step size adapted to 25% acceptance (multiplicative, frozen after M); likelihood-only pCN acceptance; cyclic scan.

## Smoothing operator
Cells within H = 0.1 of an interface (112 of 256) are replaced by the plain average of the raw field over the cells
inside a disc of radius 0.1 (6-9 cells, mean 8.6). Plain average chosen: variance factor 0.89, versus 7.6 for the
literal sqrt(n)-rescaling "variance preserving if uncorrelated" (rejected; no variance inflation, as requested).
e5: all bands from the same raw field, linear in c (Gaussian with singular covariance). e6/e7: when one subdomain
moves, its band is averaged from the new raw values of that subdomain and the *stored* values of the others, so the
state is (c, field) and is history dependent; e7 isolates this lag. The paper's coarse-scale two-stage filter is not included.

## Diagnostics
MPSRF/PSRF in the group's convention (window from iteration 1000, adaptation included, growing to the end, every 2nd
sample, curve step 15 saved samples, W divisor n-1, B/n divisor m-1, max PSRF reported, convergence = first point
below 1.2, not split-chain), on the native state (20) and on common coordinates z (L2 projection of the field onto
the global KLE). Also AR histograms, relative-error traces, posterior fields, Gaussian KL of z and field comparison
against the control (`runs/grid_arms_2026-11/arm1_check_n16/post`).

## Prior test (likelihood off)
Short-pilot values (mean variance band/interior; correlation of cell pairs across interfaces, target 0.956):
global 1.0/0.98, 0.96; raw 0.92/0.96, 0.01; projected 0.62/1.02, 0.88; smooth-all 0.66/0.96, 0.79;
smooth-one 0.67/1.05, 0.62 (s=0.2) and 0.52 (s=0.7); smooth-subdomain 0.67/0.96, 0.59 and 0.51.
The smoothed prior has a band variance deficit and step-size-dependent cross-interface correlation in e6/e7.
Final numbers come from `paper_prior.sbatch`.

## Running
    git fetch origin paper-msm-study && git checkout origin/paper-msm-study -- brho cluster docs experiments tests
    python tests/run_tests.py test_msmpaper
    bash cluster/submit_paper_msm.sh
