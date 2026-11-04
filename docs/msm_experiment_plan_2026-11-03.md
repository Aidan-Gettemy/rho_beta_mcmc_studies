# MSM local-KLE study: experiment plan of 2026-11-03

Follows the first production runs (experiments 1, 2, 3a, 3b, 4, 5; `docs/msm_local_kle_experiment.md`,
report `runs/msm_local_kle_2026-11/report/`). Agreed with Aidan on 2026-11-03. All runs use the
common settings below; only the listed items differ.

## Common settings (all runs)
- Problem, data, likelihood, seeds: as in the spec (32x32, Dirichlet 1/0 left/right, sigma_rel = 0.003, checkerboard
  data, truth seed 2026, chain seeds [7, chain], 64 chains, N = 100,000, step size adapted to 25% acceptance during
  the first M = 10,000 iterations of a run (of each phase for sequences) and then frozen, one forward solve per
  iteration, likelihood-only acceptance).
- **Saved states:** the global theta and the local state(s) every **50** iterations from iteration 0 (the first runs
  saved every 100). Per iteration: relative error, Phi, accepted flag, grid used.
- **MPSRF/PSRF method** (group convention; also written into every `post_summary.json`): `brho/mpsrf.py`; window
  starts at **iteration 1,000** (of the phase for the second phase of a sequence), adaptation period included,
  grows to the end of the run; every 2nd saved sample (skip 2), a curve point every 15 saved samples;
  W = mean within-chain covariance (divisor n-1), B/n = covariance of the chain means (divisor m-1),
  MPSRF = sqrt((n-1)/n + (m+1)/m lambda_max(W^-1 B/n)); PSRF_i = sqrt(V_ii/W_ii), V = (n-1)/n W + (m+1)/m B/n, reported
  PSRF = max_i PSRF_i; convergence iteration = first point (after the first 10) below 1.2, interpolated; not
  split-chain. Computed on theta (64, or the 58 range coordinates where theta lives in the rank-58 range), on
  theta in the **common 58 range coordinates for every run** (like-for-like comparison), and on the local
  coefficients (c_local64 for the primal raw state, etc.).
- **Posterior fields:** 10,000 samples (5,000 per phase for sequences) drawn at random from the saved post-M states;
  pointwise mean and sd of log kappa and pressure; contourf/jet plots (`experiments/msm_fieldplots.py`).
- Each run is one self-contained script `cluster/slurm/msm_<folder>.sbatch` (flags written in the script, conda
  environment `gd24_numba` activated and checked), output `runs/msm_local_kle_2026-11/<folder>/`.

## Notation: lifts and moves
B_g (r x p_g): local coefficients c -> global theta = B_g c (g = primal p, dual d); r = 64.
Pi_g (p_g x r): the original lift (local projection of the global field onto the local modes): lossy.
**Field-norm lift** L_g = (B_g' Lam B_g)^+ B_g' Lam, Lam = diag(T'T): c = L_g theta minimises ||T(theta - B_g c)||
(`brho.msm_local.field_lift`; the group's G = B' Lam B). Dual (B_d surjective, rank 64): B_d L_d = I, **exact**.
Primal with 4 modes (rank 58): B_p L_p is the Lam-orthogonal projector onto the 58-dim range; the remaining part of
theta (6 directions, including part of the largest-variance modes) cannot be represented.
Measured on 300 post-adaptation pilot states (round trip theta -> c -> B c, then Phi): primal Pi: field change 0.21,
median dPhi 9.7; primal B+ (theta-norm LS): 0.29, 135.6; primal field-norm LS: 0.14, 2.2; dual Pi: 0.07, 0.68; dual any
least-squares lift: 0, 0.

**alt (6a-6d).** State theta. Grid g = (it // L) mod 2 (primal first). c0 = L_g theta, c' = sqrt(1-s_g^2) c0 + s_g xi,
`theta' = theta + B_g (c' - c0)`: the part of theta the grid cannot represent is an untouched **offset** (s = 0 gives
theta' = theta exactly). 6d: primal move without the offset, `theta' = B_p c'` (lift only). Two step sizes (one per
grid), adapted separately.
**mixture_inn (4b).** Random grid per iteration, `theta' = sqrt(1-s^2) theta + s B_g xi`: never projects the state.
**sequence_exact (5b).** As experiment 5, but at iteration K the dual coefficients are `c = L_d theta` (B_d c = theta
exactly: Phi unchanged), and the step-size adapter restarts.
**mixture with 8 modes (4-8).** Experiment 4 unchanged (Pi lift, random grid per iteration) with 8 local modes per primal
block (128 coefficients, full rank).

## The runs
| ID | folder / script (`msm_<folder>.sbatch`) | what it is | question |
|---|---|---|---|
| 1 | exp1_global_s50 | global pCN, rerun with 50-it saving | control; extra local states (none for global) |
| 2 | exp2_primal_raw_s50 | primal raw, 4 modes, rerun | local coefficients every 50 |
| 3a | exp3a_primal_pinv_s50 | pinv-whitened, rerun | same |
| 3b | exp3b_primal_aug_s50 | whitened + null-space term, rerun | same; (c, zeta) every 50 |
| 6a | exp6a_alt_L1 | alt, L = 1, offset | is the corrected experiment 4 workable? |
| 6b | exp6b_alt_L10 | alt, L = 10, offset | does staying on a partition a few iterations help? |
| 6c | exp6c_alt_L100 | alt, L = 100, offset | long-block limit |
| 6d | exp6d_alt_L1_nooffset | alt, L = 1, lift only | how much comes from the offset vs the lift? |
| 5b | exp5b_sequence_exact | sequence, exact switch | does the exact switch change phase-2 MPSRF? |
| 2-6 | exp2_6_primal_raw_d6 | primal raw, 6 modes/block | long-range-modes hypothesis (full-rank Sigma_ms) |
| 2-8 | exp2_8_primal_raw_d8 | primal raw, 8 modes/block | same |
| 4-8 | exp4_8_mixture_d8 | experiment 4 with 8 modes/block | does more local modes alone rescue experiment 4? |
| 4b | exp4b_mixture_innovation | never-project mixture | compare with 6a: effect of the offset |
| post | msm_post_rerun_exp4_exp5.sbatch | post-processing only of the first-pass 4 and 5 | new MPSRF window, 10,000-sample fields |

The first-pass scripts `msm_exp{1_global,2_primal_raw,3a_primal_pinv,3b_primal_aug,4_mixture,5_sequence}.sbatch`
stay in the repository as the record of the first results (marked FIRST PASS: do not resubmit, they would overwrite
them). Run all new jobs with
`for f in cluster/slurm/msm_exp*_s50.sbatch cluster/slurm/msm_exp6*.sbatch cluster/slurm/msm_exp5b*.sbatch cluster/slurm/msm_exp2_[68]*.sbatch cluster/slurm/msm_exp4_8*.sbatch cluster/slurm/msm_exp4b*.sbatch cluster/slurm/msm_post_rerun_exp4_exp5.sbatch; do sbatch $f; done`.
The reruns 1, 2, 3a, 3b use the same seeds as the first runs; saving does not touch the random stream, so they should
reproduce acceptance and relative-error traces exactly (a check).

## Prior tests of the new samplers (likelihood off, s = 0.5, 24 chains x 20,000 iterations;
`experiments/run_msm_prior_test2.py`, `tables/msm_prior_test2/`)
| sampler | cov err | pointwise var ratio mean (min-max) |
|---|---|---|
| alt L = 1 / 10 / 100 (offset) | 0.471 / 0.447 / 0.455 | 0.773 / 0.777 / 0.778 (0.64-1.25 / 0.64-1.18 / 0.64-1.18) |
| alt L = 1, no offset | 0.506 | 0.759 (0.61-1.23) |
| mixture_inn | 0.460 | 0.769 (0.62-1.17) |
| mixture, Pi, 8 modes | 0.318 | 0.711 (0.59-1.07) |
| mixture, Pi, 4 modes (experiment 4) | 0.424 | 0.620 (0.51-0.97) |
| sequence_exact, phase 1 / 2 | 0.644 / 0.753 | 0.783 / 0.749 |
The alternating samplers sit at the predicted stationary covariance (Sigma_p + Sigma_d)/2 (variance ratio 0.77): the
frozen offset and the exact lifts remove the *lossy projection*, not the missing between-block correlation (that is
a property of Sigma_ms, whatever the local dimension). These are not prior-preserving; correctness is set aside by
decision, as in experiments 2, 4 and 5.

## Expectations to compare with the results (not assumed)
6a-6c should recover about 25% acceptance on both grids (6d probably lower on the primal grid); 4-8 should recover
primal acceptance (pilot: 0.22); 5b should show no jump in Phi at the switch (tested: Phi before = after);
2-6/2-8 test whether the six unreachable directions (partly the largest modes) are what makes experiment 2
mix faster than the global sampler (compare on the common 58 coordinates).

## Status
2026-11-03: implemented and tested (`brho/msm_sampler.py`, `brho/msm_local.py: field_lift`,
`tests/test_msm_sampler.py`, tiny runs of every kind through the chain runner and the post-processing). Not yet run
at production size; numba path untested here.
