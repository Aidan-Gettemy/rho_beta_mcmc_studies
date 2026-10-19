# Run log

One entry per real run: date, command, settings, headline numbers, what changed.
Quick (`--quick`) smoke runs are not logged; they go to `runs/quick/`, which git ignores.

## 2026-09-26 - E2 grid study on the July data (`runs/july_raw/`)

- Command: `make e2 ROOT=runs/july_raw ARGS="--truth-mode raw --seed-scheme legacy --n-workers 2"` (legacy seeds were the only scheme at the time), plus the setup and combined reports
- Settings: 5×5 grid, β ∈ {0.5, 1, 2.3, 5, 10} × ρ ∈ {0.5, 1, 1.5, 2.25, 3}; 4000 iterations, 2000 burn-in per chain; r = 56
- Data check: σp = 3.61e-3, K*_obs = 3.19, Ḡ = 0.286, which matches the July reports exactly
- Result: every pair fits at RMS/σp 1.31–2.77 (July report: 1.4–2.5); best Φ 100–421 (July: 110–268) against the floor of 64; acceptance 0.08–0.21. The top row (ρ = 3) is warmest, as in the July report.
- Open: the upper ends of RMS/σp and best Φ are wider than in the July report (not yet investigated). Runtime 0.6 min on 2 cores.

## 2026-09-27 - Faster solver, cluster workflow, frozen KLE basis (PR #2)

- Solver: `brho/fv.py`, one 2D/3D permeameter. The numba CSR fill follows the slab solver (NumPy fallback, identical values). The solver is `splu` + MMD_AT_PLUS_A in 2D and Jacobi-scaled CG in 3D. It matches the July solver to 1e-14 (`tests/test_fv.py`).
- Timings here, without numba: 100×100 went from 32 ms to 22–27 ms per solve and 30³ from 5.5 s to 0.14 s. The numba speed-up still needs measuring on the cluster (`make bench`).
- KLE: separable (Kronecker) build above 4,096 cells. It is exact (same spectrum and truncated covariance to 1e-12) and builds 100×100 in 0.01 s.
- Finding: the covariance has repeated eigenvalues, so `eigh`'s basis rotates with the BLAS thread count. On this machine 20 of 56 modes rotated, and the same θ gave log K differing by up to 1.1. Fix: setup saves `kle.npz`, and every chain and report loads it.
- Check: rerunning July E2 with pinned threads and a recomputed basis gives different chains (per-pair RMS/σp moves by up to about 0.5; range 1.28–2.26 against the committed 1.31–2.77) but the same conclusion. With the July basis frozen (`runs/july_raw/results/e2_grid/kle.npz`, added), the rerun reproduces the committed chains to 1e-15.
- Implication: one 4,000-iteration chain per grid pair gives noisy per-pair numbers. Read the pattern, not individual disks, or run several chains per pair on the cluster.
- Seeds: `SeedSequence([seed0, k])` is now the default, and `--seed-scheme legacy` gives the July seeds.

## 2026-09-27 - Gauss-Legendre Nyström KLE (port of KLE_matlab) and build-once store

- `brho/kle.py` now has four builders: the discrete `dense` and `kron`, and the new
  Nyström `nystrom` (full, any kernel) and `nystrom_sep` (tensor of 1D weighted
  problems, for separable kernels). `auto` keeps the dense discrete KLE up to 4,096
  cells and uses `nystrom_sep` above.
- Checks: the midpoint-rule Nyström at the cell centers equals the discrete KLE
  (1e-12); the separable and full Nyström solvers agree (spectra 1e-13, covariance
  1e-11, 2D and 3D); the July path is bitwise unchanged; the legacy frozen basis
  still loads.
- Convergence (sq-exp, 16×16 output): at ℓ = 0.1, nq = 20 gives 5e-7 and nq = 30
  machine precision; your ℓ = 0.022 example needs nq = 80 (7e-5; nq = 60 is 6% off).
  Rule: nq ≈ 3/ℓ. Each build records `nq_check`.
- Your MATLAB example (ℓ = 0.022, nq = 80, 95%) gives r = 1,025 modes in 0.24 s on the
  separable path and 93 s on the full one. That means a sampler dimension of about
  1,000 at that ℓ.
- A Nyström study at 16×16 (ℓ = 0.1, nq = 30) keeps r = 58 modes against 56 for the
  discrete KLE (95% of the continuous trace, plus tied modes). The field variance is
  lower in the corners (about 0.89), the usual effect of truncating near a boundary.
- Workflow: setup steps load the basis and truth from `$BRHO_ROOT/store/`, or build
  them once under a lock. E4 checks that the curve's basis is the sampler's basis.
- Bug fixed: a refused E4 setup used to delete the existing chains before refusing.
  Validation now happens first.
- Not verified here: a direct comparison with the MATLAB output. Octave is not
  available in this workspace; `benchmarks/kle_compare_matlab.py` prints and saves
  the eigenvalues to compare.

## 2026-09-27 - E1 driver: naive global pCN on (θ, β, ρ), mpi4py

- New: `brho/naive.py` (kernel, resumable chain, field reconstruction),
  `experiments/run_e1_naive.py` (mpi4py driver, one chain per rank; process-pool
  fallback without MPI), `reports/report_e1.py`, `cluster/slurm/mpi_e1_naive.sbatch`.
- Saved per chain: per-iteration acceptance, Φ, relative error, log β, ρ, level
  and amplitude (float32); every `thin` iterations θ, β, ρ, Φ and p_obs (float64),
  from which the report rebuilds log K and full pressure fields exactly (tested).
  Resuming from a checkpoint is bit-identical to an uninterrupted chain (tested).
- Local demo (4 chains × 30k iterations, standardized truth, prior
  β ~ logN(0, 1), ρ ~ logN(0, 0.75²); not a production run): acceptance 26.7%
  after adaptation (step 0.0066); relative error 0.026–0.030 against noise 0.021,
  still decreasing. The chains end at different pairs, (β, ρ) = (2.57, 1.10),
  (1.29, 1.44), (2.15, 1.22) and (1.86, 0.98), against the truth (2.3, 1.5).
  Spread of chain means over the prior sd: log β 0.26 and ρ 0.14, against level 0.04
  and amplitude 0.08. PSRF is > 2 for all four: global pCN mixes slowly at this length.
- For the cluster: 64 chains × 200k iterations (the sbatch default). Expect the level
  to settle while β and ρ stay spread.
- Not verified here: a real mpi4py run (not installable in this workspace). The MPI
  code path ran against a one-rank stand-in for COMM_WORLD.

## 2026-09-27 - E1 blocked pCN (the standard) vs global pCN

- Why: the global move adapted to a step of about 0.005. The curvature of Φ in the
  whitened coordinates is dominated by ξ_β: β scales the whole field, and the inflow
  data measure the field level very precisely. The top eigenvalue is 3.1e5, against
  1.4e4 next; the typical θ direction has about 80. A single step must respect the
  stiffest direction (σp = 5.4e-3, 1% of the pressure range; precision 3.4e4).
- New default `--move blocked`: A) pCN on θ, B) pCN on (ξ_β, ξ_ρ), with separate
  adapted steps and 2 solves per iteration. `--move global` is kept. Each move writes
  its own folder (`e1_blocked`, `e1_global`), and the report compares them at equal
  PDE solves.
- Local comparison (4 chains, 60k solves each, not a production run):

  | | blocked | global |
  |---|---|---|
  | steps | θ 0.016, hyper 0.0066 | 0.0066 |
  | median relative error (noise 0.0214) | 0.0248 | 0.0270 |
  | log β: spread / prior sd (PSRF) | 0.26 (5.8) | 0.32 (5.3) |
  | ρ | 0.04 (1.18) | 0.16 (5.3) |
  | level | 0.03 (1.49) | 0.03 (1.90) |
  | amplitude ρ·std(G) | 0.00 (1.00) | 0.10 (2.2) |

- Finding: with the blocked sampler, β (the split between log β and the mean of the
  Gaussian field) is the non-identified direction: β ranges from 1.4 to 2.9 across
  chains. ρ and the amplitude are identified, weakly, through the prior
  normalization of θ (non-uniqueness report, Remark 2). ρ was still rising toward
  the truth of 1.5 at this length (about 1.3). The apparent ρ drift under the global
  move was mostly slow mixing.
- Cluster: `MOVE=both sbatch cluster/slurm/mpi_e1_naive.sbatch` runs both at 400k
  solves per chain, followed by the comparison.

## 2026-09-27 - 32×32 default grid, 512 chessboard observations

- Default `n = 32` (h = 1/32): ℓ = 0.1 is resolved by 3.2 cells, against 1.6 on
  16×16. Observations are the fine chessboard: 512 cells. `Problem` warns when
  n·ℓ < 3. The July runs and tests use `--n 16` explicitly and stay exact.
- Cost: 1.45 ms per solve (0.46 ms at 16×16). The KLE is still dense and r = 57
  at 95%; σp = 5.8e-3 (1% of the pressure range).
- Four times the data tightens the posterior. Local blocked E1 (4 chains × 20k
  iterations) adapted to steps θ 0.0059 and hyper 0.0033 (at 16×16: 0.016 and
  0.0066). It is not converged at this length (PSRF 3–17), with relative error
  0.025 against noise 0.020. The E2 grid at 4,000 iterations reaches RMS/σp
  1.4–3.2 and best Φ 477–2,115 against a floor of 256: longer chains are needed
  at 32×32.
- Finding, truncation floor: the best Φ reachable in the truncated KLE span (true
  β and ρ, log K projected onto the basis):

  | grid, obs | 95% (r) | 99% (r) | 99.9% (r) | floor |
  |---|---|---|---|---|
  | 16×16, 128 | 203 (56) | 78 (86) | 67 (129) | 64 |
  | 32×32, 512 | 682 (57) | 310 (89) | 240 (136) | 256 |

  At 95% energy the model cannot fit the data to the noise level (2.7–3.2× the
  floor). This is the offset the July report attributed to truncation, and the
  likely cause of its 65% band coverage. 99% brings it to 1.2×.

## 2026-09-27 - Relative-error likelihood as the default (advisor's convention)

- Φ = ‖d − G‖²/(2σ_rel²‖d‖²) = relerr²/(2σ_rel²), using the global normalized L2
  norm, as in `normaL2squared` in `MCMCpar.py`. σ_rel = ε/√n_eff; only the ratio
  matters: (0.02, 128) ≡ (0.04, 512). The default is σ_rel = 1.77e-3. Set σ_rel
  directly with `--sigma-rel`, or through the pair.
- It equals the absolute form with a per-sensor sd of σ_rel‖d‖, which grows like
  √n_obs, so the information does not rescale with grid or sensor count. That is
  appropriate here because the misfit is dominated by model error (the 95% KLE
  floor is 2.7–3.2× the noise), which is correlated between sensors.
- Floor at the truth: 16×16: 73 (absolute 64); 32×32: 61 (absolute 256). The
  32×32 likelihood carries 0.24 of the absolute weight.
- 32×32 blocked E1 (4 chains × 20k, local): steps θ 0.0117 and hyper 0.0066,
  against 0.0059 and 0.0033 with the absolute likelihood (16×16 absolute: 0.016
  and 0.0066).
- Old manifests without the key load as `absolute` (config.LEGACY_DEFAULTS), so
  the committed July run is unchanged.
- Confirmed: `MCMCpar.py` / `mcmctools` take a σ = s and use relerr²/(2s²), the
  same form. So `--sigma-rel s` reproduces their likelihood exactly, and our
  default equals s = 1.77e-3.

## Pending decisions before E3 and E4 at full budget

1. Truth: `standardized` (default) or `raw`
2. Curve statistic: `ratio_of_mean` (default) or `mean_of_ratio`
3. Accepted: the ridge move costs one solve with the curve (3 solves per iteration)
4. Accepted: the ρ prior is truncated at the curve range, ρ ≤ 5 (1.6% of prior mass)

## 2026-09-27: MPSRF / PSRF of all theta (MATLAB convention)
- `brho/mpsrf.py` ports getMSPRF (MPSRF; PSRF = worst component) and the
  crossing read-out (first point after the first 10 with MPSRF < 1.2,
  linearly interpolated).  `reports/report_mpsrf.py` draws the curves for
  theta (r dims) and (theta, log beta, log rho), and runs at the end of the
  E1 job and in report_all.
- Curve exactly as computeMPSRFcurve: points at thinned counts
  k = step + iter_burn : step : n, window iter_burn : skip : k (growing,
  starts at sample iter_burn, every skip-th sample); x = k * thinning.
  With thin = 200, the default iter_burn = 10 drops only the first 2000
  iterations; the E1 adaptation runs to 20000 (use --iter-burn 100 to drop it).
- E1 blocked run 2026-09 (64 chains, 200k it., r = 57): log beta PSRF 2.10,
  rho 1.21, level 1.15, amplitude 1.11; chains settle at relerr 0.0253 vs
  noise 0.0196 (KLE truncation: the full-rank truth has 5.5% of its energy
  outside the 57 modes).  Kept: full-rank truth, 95% energy.

## 2026-09-28: E3 beta_bar(rho), 40 rho x 1024 realizations (runs/e3_curve_2026-09)
- Decisions: rho_max stays 5; statistic ratio_of_mean.
- rho = 1.5: beta_bar 2.18 vs beta_true 2.3 (-5%); 5-95% of beta_i 1.27-4.14;
  KLE vs full-rank Cholesky: 1% at the truth.
- The std of beta_i explodes (0.95 at rho 1.5, 80 at 3, 2.7e5 at 5).  Cause:
  sd(log beta_i) = sd(log Keff_i) grows with rho -- the realization's own
  level (sd Gbar = 0.231 exactly for sq-exp, l = 0.1; R^2 0.96 at rho 0.56)
  plus flow structure (series low-K bands; skew of log Keff -0.8 at rho 5,
  level R^2 0.15).  Near-lognormal: std/median ~ exp(s^2).  ratio_of_mean
  averages Keff (s.e. 1-11% of mean) so the curve is stable; mean_of_ratio
  inherits the tail (9.5e3 at rho 5).  Full note for the paper:
  project doc claude/e3_beta_curve_spread_note.md.
## 2026-09-27: E2 grid, 32x32, relative likelihood (sigma_rel 2e-3)
- 25 fixed-(beta, rho) pCN chains, 50000 it. (10000 burn), same KLE basis and
  full-rank truth as E1 (runs/e2_grid_2026-09).  Acceptance 0.20-0.30.
- Posterior-mean RMS/sigma_p 1.14-1.24 for every pair with rho >= 1 (beta from
  0.5 to 10, a factor 20); 1.36-1.41 for rho = 0.5.  beta is not identified;
  rho is bounded below by the data only (too little variability to fit).
- ||theta||^2 / r (prior value 1) goes 4-7.5 at rho = 0.5, ~1.0 at rho = 1.5
  (the truth), 0.3-0.6 at rho = 3: the prior penalty on theta is what points
  at rho; posterior-mean log K maps agree across beta at fixed rho.
- Link to E1: MPSRF(theta) final 2.17 (never below 1.2); worst PSRFs are
  theta1 (1.87) and theta5 (1.63), the modes that carry the spatial mean,
  i.e. the ones that absorb a change of log beta.

## 2026-09-28: DECISIONS after E4, and E5 (beta free, soft curve prior)
Why: E4's hard constraint beta = beta_bar(rho) gives beta an interval of sd
0.05 around a curve value 5% below the truth.  The pressures fix the level
log beta + rho * Gbar, and only the theta prior separates beta from Gbar, so
the honest posterior sd of log beta is ~ rho / |c| = 0.32 at rho = 1.5
(|c| = 4.65, c = KLE coordinates of the constant field; also rho * sd(Gbar)
= 0.35).  E1's pooled chains already spread over 0.37.

Decided (Aidan, 2026-09-28):
1. Main demonstration uses a truth in the samplers' own 57-mode KLE:
   truth_mode "kle" = the E1-E4 (standardized, seed 2026) field projected onto
   the KLE span (theta_true = T^+ G; spatial mean 0.001, sd 0.97,
   |theta|^2/r = 1.27), i.e. the same field minus its small scales.
   Alternative "kle_raw" = the raw draw projected (exactly N(0, I) in theta,
   but its spatial mean is 0.35 = 1.5 prior sd, which offsets beta by design).
   Projection keeps the truth machine-independent (eigh may rotate tied modes); noise added to the pressures (1% of range) and to Keff
   (sigma_e = 0.05) exactly as before.  This removes the truncation floor;
   the inverse crime is deliberate and must be stated in the paper.  The
   full-rank "standardized" runs (E1-E4 above) stay as the robustness check.
   Run it in its own study folder (runs/cluster_kle), E3 first (K* changes).
2. beta is FREE with the soft curve prior of report.pdf, Remark 3:
   log beta | rho ~ N(log beta_bar(rho), s(rho)^2), s^2 = (E3 scatter of
   log Keff_i)^2 + sigma_e^2.  Likelihood = pressures only, so K* is used
   once (through the prior).  New move D (beta-theta swap): log beta += delta,
   theta -= (delta/rho) c; log K changes only by the 3.4% (rms) of the
   constant field outside the 57-mode span.  The ridge move C is now free
   (beta fixed, log K exactly unchanged), as in the hierarchical report.
3. sigma_rel stays 2e-3.
4. NO posterior-informed / preconditioned proposal: it would have to be given
   to the naive sampler (E1) too; we keep the group's pCN standard.
Code: brho/samplers.py SoftCurveSampler, config.SoftSamplerConfig,
experiments/run_soft_sampler.py, reports/report_soft*.py,
cluster/slurm/node_e5_soft.sbatch; truth_mode "kle" in brho/truth.py.
Tests: prior invariance of the 4 moves (mutation-checked: dropping the curve
prior ratio from C or D, or the Jacobian, fails the test), C/D preserve log K,
the kle truth lies in the basis.
Values for this truth are in the E5 setup log and manifest (Gbar, G_std,
theta_norm2_over_r, noise_rel, noise floor at sigma_rel = 2e-3).

Local check before the cluster (8 chains x 24k it., 6k burn, truth "kle",
sigma_rel 2e-3, 2 cores): beta 2.23 [1.52, 3.32] (truth 2.3), PSRF 1.015,
ESS 9900; rho 1.39 [1.17, 1.67] (truth 1.5), PSRF 1.04; acceptance A/B/C/D
0.22/0.26/0.46/0.47, steps 0.016/0.041/0.20/0.50; mean Phi 80 (floor 52 +
r/2).  Still slow: level (PSRF 1.36, ESS 50) and theta, which move only
through A's small pCN steps -- the same pace as E4.  A pure level move
(random walk on log beta alone) was tried and dropped: given theta the data
pin the level to ~0.006, so its step adapted to 0.0065 and it did not help.
Move B was changed from pCN on (w, xi_b) to pCN on w with beta fixed: the
joint version dragged the level along with beta_bar(rho) and adapted to a
step of 0.010 (now 0.041).
## 2026-09-28: E4 three-step sampler, 64 chains x 150k it. (runs/e4_sampler_2026-09)
- Prior test (16 chains, Phi = 0) passed: w mean/sd -0.042/0.953 vs truncated
  N(0,1) -0.041/0.955; sd(Gbar) 0.231 vs 0.227; ridge acceptance 45%.
- rho 1.509 [1.26, 1.83] (truth 1.5), PSRF 1.035, ESS ~2500.  E1: log beta
  PSRF 2.10.  theta_norm2/r = 1.00; mean Phi 80 (same fit as E1).
- beta = beta_bar(rho) 2.18 [2.12, 2.29] misses 2.3: the curve's own 5%
  offset at rho 1.5 (E3); the interval carries only rho's uncertainty, not
  the realization spread of E3 (1.27-4.14).  To discuss for the paper.
- level 0.867 [0.74, 0.95] covers 0.833 but PSRF 1.20: a few chains dip to
  0.5-0.6 for tens of thousands of iterations (mean-carrying theta modes).
- MPSRF of all theta at equal work: E4 1.66 vs E1 2.17 (448k vs 400k solves);
  neither crosses 1.2.  E4: worst single PSRF 1.33 (21/57 above 1.2, spread
  over modes); E1: worst 1.87 (theta1), 5/57 above 1.2.

## 2026-09-28: E6 planned (adaptive theta proposal for BOTH samplers); spectrum check first
Revises decision 4 above (no posterior-informed proposal).  Reason: E5 fixed
beta and rho (PSRF 1.007 / 1.024, honest beta width); what is left is theta
only (pCN step 0.012 in E1, E4 and E5 alike; 22/57 modes at PSRF 1.2-1.28;
level moves only through the theta move).  Plan (report "E5: the
hyperparameters converge, theta does not", 28 Sep): generalized pCN for the
theta move, reference N(m, Gamma) estimated from pooled burn-in and frozen,
given identically to E1 (E6a) and E5 (E6b), same truth, sigma_rel, budget.
Expected: E6b converges, E6a still fails on log beta (a theta-only proposal
cannot move along E1's theta-beta ridge); reported either way.
Step 0, before any sampler code: reports/report_spectrum.py on the existing
E1 (kle truth) and E5 chains.  The argument needs (1) posterior sds of theta
spread over orders of magnitude, (2) pCN step ~ smallest sd, (3) in E5, PSRF
per principal direction growing with its sd; in E1, the worst direction
aligned with c.  If the spectrum is flat, stop and rethink.
## 2026-09-28: E5 on the cluster (runs/e5_kle_2026-09), truth_mode kle
64 chains x 150k it. (30k burn), sigma_rel 2e-3, same budget as E4.
- Prior test (likelihood off): w -0.041/0.954 (target -0.041/0.955), xi_b
  -0.001/1.000, corr 0.002, |theta|^2/r 1.000, cov error 0.027 (direct 0.028).
- rho 1.472 [1.23, 1.78] (truth 1.5), PSRF 1.024, ESS 3840.
- beta 2.15 [1.43, 3.23] (truth 2.3), PSRF 1.007, ESS 38800; sd(log beta)
  0.247 = the prediction from theta prior (0.32) + soft prior (0.377)
  combined: 1/sqrt(1/0.32^2 + 1/0.377^2) = 0.244.
- level 0.864 [0.74, 0.94] (truth 0.835), PSRF 1.16, ESS 660: a few chains
  make slow excursions to 0.55-0.6 (the theta-limited direction).
- mean Phi 70 vs floor 52 (+ r/2 at most); log K pointwise 90% coverage 94.6%
  (82% with the full-rank truth): no truncation floor any more.
- Acceptance A/B/C/D 0.25/0.26/0.46/0.46; steps 0.012/0.041/0.20/0.50.
- MPSRF of all theta at equal work (MATLAB convention): E5 1.56 at 448k
  solves vs E1 (same truth) 2.05 at 400k; neither crosses 1.2.  E1 worst
  components log beta 1.96, theta1 1.85, theta5 1.56 (the beta ridge); E5
  worst 1.28, 22/57 above 1.2, spread over modes (theta-pCN pace).
- E1 on the same truth: log beta PSRF 2.03, chain-mean spread 0.29 prior sd;
  relerr 0.0236 vs noise 0.0203.

## 2026-09-28: spectrum check results (runs/e5_kle_2026-09, PR #30)
- theta posterior widths (sqrt eig of pooled cov): 0.05 to 0.97 (E5), 0.05 to
  1.08 (E1); smooth decay, no gap.  E1 and E5 spectra coincide (same theta
  posterior) except E1's broadest direction.
- Mixing: PSRF along each principal direction rises with its width (rank
  corr. +0.89 E5, +0.85 E1); all directions above 1.2 have width > 0.7;
  within-chain variance is ~0.65 of pooled in the 5 broadest, ~0.95 in the
  20 tightest.  E1's worst direction (PSRF 1.74) is aligned with c (cos
  0.88): the beta ridge.  In E5 the c-aligned direction has PSRF 1.13.
- NOT as predicted: pCN step 0.012 is 1/4 of the narrowest marginal width,
  and sum_k (s/sd_k)^2 = 0.44, far below ~5.7 for a Gaussian random walk at
  25%.  The marginal covariance does not explain the step size.  Likely
  reason: move A holds (beta, rho) fixed, and given them the data pin the
  level direction (Gbar); marginally that direction is wide (0.72 = sd(log
  beta) / (rho * 0.231)) because beta moves.  Consequence for E6: Gamma must
  be the covariance of theta GIVEN (beta, rho), not the pooled covariance.
  report_spectrum.py now estimates it (Schur complement); rerun to confirm.

## 2026-09-28: E6 detour closed (considered, not pursued); plan = extend E1 and E5
Conditional spectrum (PR #32, Cov(theta | log beta, log rho), Schur estimate):
width along the level direction 0.72 -> 0.19 given (beta, rho) (E5; E1 0.98
-> 0.21) = the level's own posterior sd (0.062 / (1.47 * 0.231)); narrowest
width 0.051 -> 0.037; sum (s/sd_k)^2 only 0.44 -> 0.54 (a Gaussian random
walk at 25% would need ~5.7).  So neither the marginal nor the conditional
covariance explains the pCN step 0.012: the theta posterior is locally much
stiffer than its global spread (non-Gaussian).  A covariance-preconditioned
theta proposal is therefore not supported by the data; a Hessian-informed
one might be, but was NOT pursued:
Decided (Aidan, 2026-09-28): the paper's story is (1) beta-rho-theta
non-identifiability (E1, E2) and (2) a corrected sampler that is not much
more expensive and needs one extra datum (K_eff -> soft curve prior, moves
C and D).  A new theta proposal (covariance or Hessian) would blur that
story and is not what the advisor's industrial codes use.  Decision 4 of
2026-09-28 (plain pCN for theta, same in E1 and E5) STANDS.  The E6 section
of the E5 report and its step-size argument (Sec. 5, point 3) are
withdrawn.  report_spectrum.py stays as a diagnostic.
Requirement (advisor): theta itself must reach MPSRF < 1.2 in one method
and not in the other.  Extrapolation of the existing curves (power law on
the last half): E5 crosses at ~290k it. (~870k solves), E1 at ~1.1M it.
(~2.2M solves).  Plan: extend both runs from their checkpoints to equal
work, 1.2M PDE solves per chain: E5 150k -> 400k it. (3 solves/it), E1
200k -> 600k it. (2 solves/it).  experiments/extend_run.py raises the
length in the manifest (recorded under "extensions"); RESUME=1 continues
each chain exactly (tested).  Nothing else changes (burn-in, thinning,
frozen steps, truth, sigma_rel).  Expected E1 theta MPSRF at 1.2M solves
~1.4 (extrapolated), i.e. no crossing.

## 2026-09-28: setup stays as laid out (sigma, prior, boundary conditions)
Considered and NOT adopted (Aidan): (a) loosening sigma_rel to 5e-3 or 1e-2
(faster theta mixing, but the beta ridge fattens and E1 may converge too;
possible later sweep: E1 log-beta PSRF vs sigma_rel, E5 alongside);
(b) a flat prior on log beta; (c) the Dirichlet slab.
Notes for the paper: with proper priors E1's posterior is proper and pCN is
ergodic, so E1 converges in principle; its cost grows like (ridge length /
width)^2, the width set by how tightly the pressures pin the level (~sigma).
Claim "impractically slow, and slower as the data sharpen", not "never".
In this flux-driven problem (q = 1 prescribed, pressures measured) the data
see the level, so a flat log-beta prior still gives a proper posterior
(log beta ~ level - rho Gbar, width ~0.32-0.35) and changes E1 little; the
pressures already carry K_eff, so the extra K_eff datum informs beta through
the E3 curve (log beta sd 0.32 -> 0.25) and moves C/D make E5 mix.  Only a
Dirichlet-Dirichlet slab with a flat prior would make E1's posterior
improper (never converges).
Next: extend E5 to 400k and E1 to 600k iterations (1.2M solves each) at
sigma_rel 2e-3; goal theta MPSRF < 1.2 in E5, not in E1.

## 2026-09-29: E5 results at 1.2M solves; plan to convergence; E7 (multiscale) code
Extension to 1.2M PDE solves per chain (runs/e5_kle_2026-09, PR #36):
- E5: theta MPSRF 1.207 (not yet < 1.2); worst single-theta PSRF 1.104, every
  theta below 1.2 from 603k solves; log beta 1.003, rho 1.014, level 1.088
  (ESS 1293); posterior unchanged (beta 2.15 [1.42, 3.21], rho 1.47 [1.23,
  1.79], sd log beta 0.248).
- E1: theta MPSRF 1.339, worst PSRF 1.289 (theta1), theta5 1.238, log beta
  1.318: still the beta ridge.  Trend fits: E5 MPSRF 1.2 at ~1.23M solves,
  E1 at ~1.8M.  Both converge in principle; report the crossing cost of
  each rather than one cut-off.
Decided (Aidan): run BOTH to convergence (E5 -> 667k it., E1 -> 1M it.,
2.0M solves each), then the report.
Next (Aidan): the multiscale (local-KLE) theta innovation of Pereira,
"Variance control in distribution" (July 2026), and Sec. 6 of "Two
parametrization degeneracies" (Sept. 2026), on the 2D problem:
  1. prior test, naive (raw) multiscale innovation -> must fail;
  2. prior test, whitened innovation -> must pass;
  3. E7 = E5 (soft curve prior, moves A-D, same truth and sigma_rel) with
     the whitened multiscale innovation in move A.
  Possibly 3D afterwards (advisor's call).
Code (brho/multiscale.py; SoftSamplerConfig.theta_innovation = global |
ms_raw | ms_white, ms_blocks = 2, ms_local_energy = 0.95; run folders
e5_prior_test_ms_raw / _ms_white, e5_soft_ms_white; INNOV=... in
node_e5_soft.sbatch; experiments/run_ms_prior_test.py = the 2D replica of
Pereira's pure-pCN prior test).  2D operator: 2x2 subdomains, 17 local
modes each (p = 68), Sigma_ms eigenvalues 0.31-1.99, full rank (r_plus =
57), whitening check 2.4e-15.  Local checks (not the study runs):
  pure pCN, 16 x 40k it.: cov error raw 0.338 / whitened 0.026 / direct
  0.019; spatial-mean sd 0.212 / 0.231 / 0.230; per-sample normalized
  collapses both field laws to points.
  full E5 sampler, 4 x 20k it., likelihood off: cov error ms_raw 0.347,
  ms_white 0.089 = direct draws 0.089.
Expectation for step 3 (Sec. 6 of the degeneracies notes): the whitened
innovation is exactly N(0, I_r), so move A has the SAME transition kernel
as global pCN.  E7 must reproduce E5 within Monte Carlo error (posterior,
step size, MPSRF); it demonstrates correctness of the multiscale
machinery, not a speed-up.  The global path is unchanged bit for bit
(tested), so the running E5 extension is unaffected.

## 2026-09-29: E5 and E1 at 2.0M PDE solves per chain (runs/e5_kle_2026-09, PR #38)
- theta MPSRF crosses 1.2: E5 at 1.26M solves (it. 418,847); E1 not by 2.0M
  (final 1.212; trend ~2.03M).  Worst single-theta PSRF crosses: E5 0.60M,
  E1 1.60M.  E5: all 59 components (theta, log beta, log rho) < 1.06 at 2.0M.
- log beta: E5 PSRF 1.002, ESS 36,000; E1 PSRF 1.207, ESS 467 (77x); E1
  chain-mean spread 0.19 vs within-chain sd 0.27; E1 worst components log
  beta, theta1, theta5 (level ridge).  rho ESS 7,095 vs 1,598.  Level ESS
  E5 1,807 vs E1 2,316: the level moves only through the theta move in both;
  per theta move comparable (2.8e-3 vs 2.4e-3), E1 spends 1/2 of its solves
  on theta moves, E5 1/3.
- E5 posterior (stable since 450k solves): rho 1.473 [1.23, 1.79], beta
  2.144 [1.43, 3.21], sd log beta 0.247 (predicted 0.244), level 0.865
  [0.74, 0.94]; log K 90% coverage 95.8%.  One chain's level dips to ~0.2
  near it. 190k and returns.
- Paper claim: E1 converges in principle (proper priors) but its cost is set
  by the level ridge; at this sigma_rel it is 1.6-2.7x costlier in theta and
  77x less efficient in beta than E5, which needs one extra datum (K_eff).
- Progress report (LaTeX, 29 Sept) written from these runs.  Next: E7
  multiscale prior tests (ms_raw, ms_white) and E7 = E5 + whitened
  multiscale theta innovation (PR #37).

## 2026-09-29: multiscale prior tests (study runs) -- raw fails, whitened passes
Standalone pure-pCN test (experiments/run_ms_prior_test.py, 16 x 40k it.,
runs/e5_kle_2026-09/tables/ms_prior_test): cov(theta) error / spatial-mean sd
/ pointwise variance-to-target: direct 0.019 / 0.230 / 0.99-1.01; raw 0.340 /
0.212 / 0.34-1.13 (collapse along the subdomain interfaces, lowest at the
centre cross); whitened 0.026 / 0.231 / 0.99-1.02; per-sample normalized 0.225
/ 0.000 / 1.01-1.18 (spatial mean and field variance forced to points).
Full E5 sampler, likelihood off, 64 x 150k it. (e5_prior_test_ms_raw /
_ms_white): cov error 0.333 / 0.028 (direct 0.028); spatial-mean sd 0.216 /
0.232 (0.231); field variance 0.836 / 0.899 (0.899); pointwise ratio 0.36-1.13
/ 0.99-1.01.  ms_raw also shifts the hyperparameters slightly (w mean -0.061
vs -0.041, xi_b sd 0.979 vs 1; ms_white -0.040 / 1.000): the wrong theta law
leaks into rho and beta, plausibly through the ridge move (theta norm <->
rho).  Next: E7 = E5 with ms_white, 667k it. (2.0M solves).
Housekeeping: these results were first exported to runs/results-ms-prior-full
(PR #46; a full copy of the study); the 16 new files were moved into
runs/e5_kle_2026-09 and the 74 duplicates removed.  export_results.sh now
warns when the name looks like a branch name (results-...).

## 2026-09-29: 3D study -- settings (confirmed by Aidan) and code
Mirror of the 2D main study, only the dimension changes: unit cube 16^3
(4,096 cells), ell = 0.2 (3.2 cells per ell as in 2D), sq-exp, separable
Nystrom KLE nq = 40, 95% energy -> r = 78 (2D: 57); unit influx at x = 0,
p = 0 at x = 1, no flow on the side faces (the data see the level, Lemma 2);
3D chessboard sensors, 2,048 (half the cells); truth kle, beta 2.3, rho 1.5,
noise 1% of the pressure range, sigma_e 0.05, relative likelihood
sigma_rel = 2e-3; priors as in 2D.  Level ridge: |c| = 4.44, honest sd of
log beta rho/|c| = 0.34 (2D 0.32); sd(Gbar) = 0.274 (2D 0.231).  Solve ~10
ms, so 2.0M solves per chain ~5.5 h.  Study folder runs/cluster_3d, export
runs/e5_3d_2026-09.  Expected 3D-specific result: E3's curve bends
differently (Matheron f(rho) = exp(rho^2/6) grows in 3D; damped on a finite
plug).  Local tiny-budget check: K* = 2.846 (2D 2.036); multiscale 2x2x2,
16 local modes each (p = 128), Sigma_ms eigenvalues 0.13-3.44, whitening
3.5e-15; pure-pCN prior test (16 x 4k it.): raw spatial-mean sd 0.202 vs
0.273 (Pereira's 3D plug: 0.204 vs 0.276), pointwise ratio 0.17-1.29;
whitened 0.274, 0.93-1.06; normalized collapses.
Code: ProblemConfig.dim (2 | 3; 2D store keys unchanged, tested);
Problem builds Permeameter((n,)*3) for dim 3; plotting.field / line show the
mid-plane z = 0.53 in 3D; multiscale is dimension-generic (2x2x2 blocks).
Tests: 3D problem (flux scaling p(cK) = p(K)/c, truth in the KLE), E5 steps
with the likelihood on and the ms_white prior test in 3D, 2x2x2 whitening
exact, slice plotting.
## 2026-09-29: E7 (E5 + whitened multiscale theta innovation), 2.0M solves (PR #49)
64 chains x 667k it., same study, seeds and settings as E5; only move A's
innovation differs (ms_white, 2x2 subdomains, p = 68, whitening check 2.6e-15).
                          E5 (global pCN)        E7 (multiscale, whitened)
theta MPSRF crosses 1.2   1.26M solves           1.20M solves
worst theta PSRF < 1.2    0.60M                  0.70M
final MPSRF / PSRF        1.122 / 1.059          1.107 / 1.062
theta step (move A)       0.0124, acc 0.253      0.0124, acc 0.248
rho                       1.473 [1.23, 1.79]     1.473 [1.23, 1.79], PSRF 1.013, ESS 7,452
beta                      2.144 [1.43, 3.21]     2.144 [1.43, 3.21], PSRF 1.002
sd log beta               0.247                  0.247 (ESS 42,070 vs 36,000)
level                     0.865, ESS 1,807       0.865, ESS 1,751
log K 90% coverage        95.8%                  95.7%;  mean Phi 69.57 / 69.59
Differences go both ways and are Monte Carlo size: as predicted (Sec. 6 of
the degeneracies notes), the whitened innovation is exactly N(0, I_r), so
move A has the same kernel as global pCN; E7 reproduces E5.  The adapted
steps agree to 4 digits because the adaptation moves on the grid
s_init * 1.12^k and both land on the same point.
Conclusion: the multiscale machinery is correct in the full sampler (prior
tests: raw fails, whitened passes; posterior: whitened = global).  It gives
no speed-up in its whitened single-innovation form; a speed-up would need
block (per-subdomain) updates, which is outside this study.

## 2026-09-30: 3D E3 curve (runs/e5_3d_2026-09, PR #52)
1024 realizations x 40 rho in [0.05, 5], r = 78, 4.9 min.  K* = 2.846 (2D
2.036).  f(rho) = ensemble Keff at beta = 1: rises to ~1.14 at rho ~1.6-2.1,
back to 1 at rho ~3, 0.70 at rho = 4.6; the infinite-medium 3D value
exp(rho^2/6) is followed only up to rho ~1 (1.18 at rho = 1; 1.45 at 1.5,
observed 1.14): the finite 16^3 plug (L/ell = 5) strongly damps the 3D
enhancement and then reverses it, as the degeneracies notes anticipated.
beta_bar(rho) therefore dips below K* (2.50 at rho = 1.5-2.1) before rising.
At rho_true: beta_bar = 2.507 vs beta_true 2.3 (+9%; 2D +5%), realization
band 5-95% [1.38, 5.57]; Cholesky vs KLE at the truth 3.6%.
Note for E5: this truth has |theta|^2 / r = 0.72 (2D: 1.27); through the
shell argument that pulls rho's posterior down somewhat -- watch rho's
coverage.
report_curve: the f(rho) reference is now dimension-aware (2D: 1; 3D:
exp(rho^2/6)); it previously read "Matheron 2D" for the 3D curve too.

## 2026-10-01: 3D study, E5 prior test, E1 and E5 at 2.0M solves (runs/e5_3d_2026-10, PRs #54, #55)
Settings as confirmed (16^3, ell 0.2, r = 78, 2,048 sensors, truth kle,
sigma_rel 2e-3); 64 chains each; E5 667k it. x 3 solves, E1 1M x 2.
- E5 prior test (likelihood off): every check equals direct draws (cov(theta)
  error 0.032 vs 0.032; spatial-mean sd 0.274 vs 0.275; pointwise variance
  ratio 0.99-1.02; w -0.042/0.955; xi_b 0.001/0.999).
- theta MPSRF crosses 1.2: E5 1.14M solves, E1 1.69M (1.5x).  Worst
  single-theta PSRF: E5 0.75M, E1 1.44M (1.9x).  Final MPSRF 1.104 / 1.159.
  In 3D both converge within the budget; E1's slowest components are again
  log beta (1.136), theta1 (1.127), theta10 (1.118).
- log beta: E5 PSRF 1.005, ESS 16,090; E1 PSRF 1.139, ESS 562 (29x).
  rho ESS 10,250 vs 1,971; level 1,938 vs 2,612 (theta-limited in both).
- E5 posterior: beta 2.50 [1.69, 3.66] covers 2.3; level 0.845 [0.74, 0.93]
  covers 0.833; log K pointwise 90% band covers 100% of cells.
  rho 1.26 [1.08, 1.48] does NOT cover 1.5 (z = -1.9).  E1 gives the same
  rho (mean of chain means 1.28), so this is the posterior, not the sampler.
  Cause: the amplitude is identified only by the prior's typicality of
  |theta|^2.  This truth has |theta|^2/r = 0.72 (about 1.75 sd below the
  chi^2_78 mean); the posterior keeps |theta|^2/r at 1.005, and since the
  data fix rho*theta, rho shrinks: 1.5 * sqrt(0.72) = 1.27.  By mode, the
  3D truth is rich in the leading (data-informed) modes (1.54 per mode in the
  first 10) and poor in the tail (0.54 beyond mode 40).  The 2D truth
  (|theta|^2/r = 1.28) carried its excess in the tail (2.32 per mode beyond
  40), which the posterior replaces with prior draws, so 2D rho was unbiased
  (1.47).  This is the "information floor" of the degeneracies notes: rho is
  only as good as the realization is prior-typical.
- Housekeeping: runs/e5_3d_2026-09 (E3 and the prior check, 21 files) was
  byte-identical to the copies in runs/e5_3d_2026-10 and is removed; the 3D
  study's export folder is e5_3d_2026-10.

## 2026-10-01: E8 block (subdomain) pCN for theta, implementation (PR, no cluster runs yet)
The advisor's subdomain idea, made exact.  brho/blockpcn.py; the
SoftCurveSampler's move A becomes a sweep (--theta-innovation block) or a
random choice (block_random) of block moves; every move has its own step.
- Local subspace of block k: generalized eigenproblem T_k^T T_k v =
  mu T^T T v (mu = fraction of the field energy of v inside the block), keep
  mu >= 0.9, QR.  Move: pCN on alpha = V_k^T theta, rest of theta fixed;
  preserves N(0, I) exactly, likelihood-only ratio, one solve.  Coarse move on
  the complement of the primal locals (needed to span R^r).  Cell-space
  conditioning is ruled out: with r = 57 the field inside a block given the
  outside has ~zero variance.
- 2D (r = 57): primal 9 dims per block (36), dual 3x3 [3,4,3,4,6,4,3,4,3]
  (34), coarse 21; ~1% of a move's field energy leaks outside its block.
  Reach of local moves (prior variance they can change / prior variance):
  primal 0.00-1.00 (zero on the x, y = 0.5 interfaces), primal + dual
  0.21-1.99.  At energy 0.8: primal 13, dual 40, coarse 5.
- 3D (r = 78): at 0.9 primal 4 per block, dual blocks 0-1 dims (empty blocks
  get no move); at 0.8 primal 7, dual 27 x 1, coarse 22.  3D needs 0.8.
- Likelihood-off test (run_block_prior_test.py, 32 chains x 4000 sweeps,
  s = 0.3): cov(theta) error 0.099 / 0.082 / 0.080 (coarse+primal / + dual /
  random) against 0.043 for as many direct draws; pointwise variance 0.95-1.05;
  no seams.  Started at exact prior draws, 15 sweeps of either schedule keep
  N(0, I) to the precision of direct draws (test).  With the likelihood off
  the lag-1 maps only count refreshes (sqrt(1 - s^2) per move).
- StepAdapter now adapts on acceptance per PROPOSAL (identical to the old
  rate when every move is proposed once per iteration, i.e. all earlier
  runs); checkpoints store acc_tot_prop.
- Smoke run (4 chains x 3000 it., 2D, likelihood on): the block steps adapt
  to 0.03-0.37 against E5's global theta step 0.0124; acceptance on target.
  Whether this beats E5 per PDE solve is the cluster run (125k it. x 16
  solves = 2.0M).

## 2026-10-01: 3D multiscale -- prior tests and E7 (runs/e5_3d_2026-10, PRs #57, #58); local-KLE tie fix
Pinned (Aidan): the multi-truth / simulation-based-calibration study of rho's
coverage (see the 3D entry above); revisit later.
Standalone pure-pCN test (16 x 40k it.), raw / whitened / normalized vs direct:
cov(theta) error 0.640 / 0.031 / 0.265 (0.022); theta variances 2.65-0.43 /
1.01-0.99 / 1.33-0.09; spatial-mean sd 0.201 / 0.275 / 0.000 (0.274;
Pereira's 3D plug 0.204 vs 0.276); field variance 0.737 / 0.879 / 1.000
(0.879); pointwise variance ratio 0.18-1.29 / 0.99-1.01 / 0.92-1.24.  Raw
fails about twice as badly as in 2D (8 subdomains, more interfaces).
Full E5 sampler, likelihood off, 64 x 150k: ms_raw cov error 0.616 (direct
0.032), spatial-mean sd 0.216 (0.275), pointwise 0.19-1.26, and it leaks into
the hyperparameters more than in 2D (xi_b sd 0.939, w mean -0.064);
ms_white equals direct draws on every check (0.032, 0.275, 0.98-1.01).
E7 (ms_white, 667k it.) vs E5: theta MPSRF < 1.2 at 1.08M vs 1.14M solves;
worst PSRF 0.65M vs 0.75M; rho 1.261 [1.08, 1.48] in both (the same
realization-driven miss); beta 2.49 [1.69, 3.64] vs 2.50 [1.69, 3.66]; level
0.846 vs 0.845; theta step 0.014 vs 0.0124 (one 1.12x adaptation step
apart).  E7 reproduces E5 in 3D, as in 2D.
Found: the local KLE truncation (95% energy) cut through a tied group of local
eigenvalues in 3D (each 8^3 subdomain is a cube; positions 12-16 tied, the
cut at 16), so which modes were kept depended on eigh's basis (machine,
threads), and Sigma_ms differed between runs that used the same global KLE:
0.130-3.44 (standalone test) vs 0.156-3.48 (sampler runs).  The whitened
innovation is exactly N(0, I) for whichever operator a run builds, so the
verdicts above (and E7) stand; only the raw operator's exact numbers were not
reproducible.  2D was not affected (its cut falls between distinct values).
Fix: multiscale.local_truncation keeps every mode tied with the last kept one
(the global KLE's rule), so the kept subspace and Sigma_ms are unique.  3D
now 17 modes per subdomain (p = 136, 96.5% local energy), Sigma_ms
0.157-3.48; 2D unchanged (17 each).  Test: rotating eigh's vectors inside
tied groups leaves Sigma_ms unchanged, and changes it under the old rule.
Local check of the corrected 3D standalone test: raw 0.646 / 0.201 /
0.18-1.31, whitened passes -- same conclusions.  Recommended: rerun the cheap
3D prior tests (standalone + ms_raw + ms_white) for reproducible numbers in
the paper; E7 need not be rerun (its kernel is global pCN in law).

## 2026-10-01: E8 block pCN at 2.0M solves (runs/e5_kle_2026-09/e5_soft_block, PRs #61, #62)
64 chains x 125k iterations x 16 solves (coarse + 4 primal + 9 dual block
moves, B, D); burn-in 10k, thin 25.  Prior tests passed first (full sampler:
cov(theta) error 0.027 vs 0.028 direct; pointwise variance 0.99-1.01).
- theta MPSRF crosses 1.2 at 0.94M solves (E5 1.26M, E7 1.20M): 1.33x.
  Not a lucky crossing: at equal solves E8's MPSRF is lower all along
  (0.75M 1.278 vs 1.345; 1.0M 1.185 vs 1.241; 1.5M 1.111 vs 1.167; 2.0M
  1.075 vs 1.123).  Worst single-theta PSRF crosses at the same cost
  (0.60M vs 0.60M): the slowest single mode does not gain.
- ESS at equal solves (E8 vs E5): log beta 102,500 vs 36,000 (2.8x); rho
  12,410 vs 7,095 (1.7x); level 2,903 vs 1,807 (1.6x; PSRF 1.010 vs 1.045);
  Phi 9,663 vs 4,573 (2.1x).
- Posterior identical to E5 to the reported digits: rho 1.472 [1.23, 1.79],
  beta 2.145 [1.43, 3.21], sd log beta 0.247; log K 90% coverage 96%.
- Adapted steps: coarse 0.025, primal 0.031-0.095, dual 0.048-0.30 (global
  pCN in E5: 0.0124); every move at its 25% target.  Per-block tuning is
  what buys the gain: information is uneven across the domain.
- Caveat: one run per sampler; E5 vs E7 (same kernel) differ by ~5% in the
  crossing, so the 1.33x is well outside run-to-run noise, but a replicate
  would firm it up.

## 2026-10-01: E8 without the coarse move, setup (no runs yet)
- Question (Aidan): does the coarse move cost more than it gives?  It takes
  1 of 16 solves per iteration, adapted to a step of 0.025, and supplies about
  2% of the per-iteration refresh of theta.
- Without it, primal + dual still span R^57 in 2D (irreducible), so the
  sampler stays exact.  But two theta-directions are barely reached: the
  refresh weight (eigenvalues of sum_j V_j V_j^T) drops to 0.002-0.008 against
  a median of 1.1.  A 2-cell halo on the dual blocks lifts the minimum to 0.17
  (dual dims 3, 6, 3, 6, 13, ...); also loosening the dual energy to 0.8 lifts
  it to 0.61.
- New options: --block-coarse 0 (refused if the locals do not span R^r),
  --block-dual-energy, --block-dual-halo, and --exp-tag (TAG in the sbatch)
  for a separate run folder.  Runs: E8a (no coarse) and E8b (no coarse + dual
  halo 0.0625), 133,400 iterations x 15 solves = 2.0M.


## 2026-10-02: scaling study, 64x64, ell = 0.05, r = 208 (runs/cluster_l005, PR #65)
E3, then E5, E8 2x2 (2x2 primal + 3x3 dual) and E8 3x3 (3x3 primal + 4x4
dual), 64 chains each at 2.0M solves (3 / 16 / 28 solves per iteration).
Block prior tests (likelihood off) pass for both layouts (pointwise variance
0.93-1.06 at 4000 sweeps; cov error 0.15-0.19 vs 0.08 direct, autocorrelation).
- theta MPSRF first below 1.2: E5 1.14M, E8 2x2 0.67M (1.69x), E8 3x3 1.03M
  (1.11x).  Below 1.15: 1.44M / 0.91M (1.58x) / 1.29M.  Below 1.10: E5 never
  (final 1.108), 2x2 1.26M, 3x3 1.78M.  At equal solves (E5 / 2x2 / 3x3):
  0.75M 1.315 / 1.177 / 1.291; 1.0M 1.231 / 1.129 / 1.208; 2.0M 1.108 /
  1.063 / 1.089.  Worst single-theta PSRF below 1.2: 0.39M / 0.36M / 0.49M.
- ESS at equal solves (E5 / 2x2 / 3x3): log beta 39,360 / 106,700 / 116,500;
  rho 9,715 / 14,120 / 11,650; level 3,760 / 8,205 / 9,079.
- Same posterior in all three: beta 2.25 [1.87, 2.70] (truth 2.3), level
  0.833 [0.76, 0.90] (truth 0.832), log K 90% coverage 90%.
- Gain of 2x2 over E5 grew from 1.33x (r = 57) to about 1.6-1.7x (r = 208).
  3x3 is better than E5 but worse than 2x2: more blocks did NOT help.  Its
  coarse move holds 63 of 208 directions (29% of the prior variance) at a
  small step (0.039), and several small dual blocks adapted to very large
  steps (up to the 0.95 cap: nearly prior-like, so each of those solves buys
  little).  28 solves per iteration spent evenly on blocks of very different
  value.
- rho = 1.204 [1.05, 1.38] misses 1.5 (z = -2.9) in all three samplers, so it
  is the posterior, not the sampling.  Not the 3D explanation: this truth has
  |theta|^2/r = 0.96 (predicts rho ~ 1.47).  The soft curve prior is not the
  cause either (truth at z = +0.41; the curve rises with rho, so it would push
  rho up).  The posterior mean Phi is 108 against a noise floor of 49 (2D
  main study: 69.5 vs 51.7), which points to tension between the data and the
  prior for this realization.  Open; needs a per-mode check of the posterior
  rho*theta against 1.5*theta_true (chains on the cluster).  Another case for
  the pinned multi-truth / SBC study.
- report_mpsrf: crossing marks now take the curve's colour, E8 labels name the
  block layout, distinct colours for E5 / E8 2x2 / E8 3x3 (rerun the report).

## 2026-10-02: housekeeping (scaling study)
- runs/cluster_1005 (a typo of cluster_l005, PR #67) was a byte-identical copy
  of runs/cluster_l005 apart from the redrawn comparison figure (PR #68);
  removed.  The scaling study lives in runs/cluster_l005 only.
- On the cluster the sparse clone had removed the study's committed files
  (manifests, frozen kle/truth/curve, figures) after the results commit, leaving
  only the git-ignored chains; `git sparse-checkout add runs/cluster_l005`
  restored them.  Warning added to cluster/README.md.
- report_mpsrf: smaller legend when the labels are long (the E8 labels now name
  the block layout and ran off the figure).

## 2026-10-02: questions on the r = 208 study, answered with the linearized posterior
Gaussian approximation at the truth (finite-difference Jacobian, 2r + 1 solves).
- Higher misfit (Phi 108 vs floor 49): each data-informed direction adds ~1/2
  to the posterior mean of Phi.  Effective number of informed directions 46
  (r = 57) vs 97 (r = 208).  Linear prediction of the posterior mean Phi: 70.1
  vs MCMC 69.5 (r = 57), 106.7 vs 108.3 (r = 208).  Not a worse fit.
- rho miss: rho is pinned by the prior's typicality in the DATA-INFORMED
  directions (eigenvectors of the Gauss-Newton Hessian, not the leading KLE
  modes).  The truth's energy there is 0.96 (r = 57) and 0.66 (r = 208), the
  0.3rd prior percentile.  Prediction rho = 1.5 sqrt(energy): 1.468 / 1.216
  against MCMC medians 1.473 / 1.204.  The posterior is right; the truth is a
  1-in-300 draw.
- Block layouts on the Gaussian surrogate (beta, rho fixed; 24 chains x 200k
  theta-solves; tuned steps match the cluster's): median IACT per theta-solve
  global 13.6k (13.6k with another seed), 2x2 13.0k, 3x3 12.4k, 3x3 with
  visits weighted ~1/s^2 12.3k.  Per theta-solve the layouts are equivalent and
  weighted visiting adds nothing.  Hypothesis: E8's gain is the budget split
  (theta share 33% in E5, 88% in E8 2x2, 93% in 3x3); 3x3 updates beta / rho /
  level too rarely.  The earlier explanation (per-block steps exploit uneven
  information) is NOT supported.  Control: E5 with --theta-repeat k (this PR).
- Truth screening (reports/linearized/screen.py, seeds 2000-2099): at r = 208
  the original seed 2026 is the most atypical of 101 (informed energy 0.66,
  0.3rd percentile).  Representative: 2074 (0.99, 49.5th; predicted rho 1.49).
  Stubs: 2063 (0.82, 9.8th), 2086 (1.22, 93.4th).  At r = 57 seed 2026 is
  already typical (45th percentile).  The analysis scripts are in
  reports/linearized/ (lin.py, sim.py, visit.py, screen.py).


## 2026-10-03: control for E8 (theta-repeat) and the scaling study on new truths
**Control** (main 2D study, r = 57, 64 chains, 2.0M solves; runs/e5_kle_2026-09,
e5_soft_a4 / a7 / a14 = E5 with move A repeated 4 / 7 / 14 times per iteration):

| solves to theta MPSRF < | E5 | E5 x4 | E5 x7 | E5 x14 | E8 2x2 |
|---|---|---|---|---|---|
| 1.2  | 1.26M | 0.71M | 0.63M | **0.61M** | 0.95M |
| 1.15 | 1.65M | 0.87M | 0.76M | **0.71M** | 1.12M |
| 1.1  | never | 1.17M | 1.04M | **0.96M** | 1.58M |
| final (2.0M) | 1.123 | 1.047 | 1.043 | **1.039** | 1.075 |

- E8's speed-up is entirely the budget split, and then some: E5 x14 spends
  the same theta share as E8 2x2 (14 of 16 solves) and converges 1.55x faster
  than E8, 2.1x faster than E5.  Diminishing returns past k = 7.
- Worst single-theta PSRF below 1.2: E5 0.60M, x4 0.48M, x7 0.38M, x14 0.43M,
  E8 0.60M.
- ESS (rho / log beta / level): E5 7.1k / 36k / 1.8k; x14 11.8k / 66k / 3.3k;
  E8 12.4k / 103k / 2.9k.  Block moves still mix log beta best.
- Posterior unchanged (rho 1.473-1.478 [1.23, 1.79] in all).
- Conclusion: block pCN is exact but not faster than global pCN; the lever
  is how many theta moves are made per round of B, C, D.  The Gaussian
  surrogate predicted this (equal efficiency per theta-solve).
- Fairness: E1 must get the same knob (theta moves repeated) before the final
  E1 vs E5 comparison.

**Scaling study on new truths** (64x64, ell 0.05, r = 208):
- Seed 2074, E8 2x2 / 3x3 at 2.0M; E5 stopped at 1.59M of 2.0M (time limit,
  no report yet; resume).  theta MPSRF < 1.2: E5 1.53M, E8 2x2 0.91M, 3x3 1.30M.
- rho, all low: seed 2074 1.34 [1.19, 1.53] (covers), stubs 2063 1.24
  [1.09, 1.40] and 2086 1.28 [1.13, 1.47] (E5 at 0.5M; miss), and seed 2026
  1.20.  beta and the level are covered in all.
- The truth screen of 2026-10-02 was wrong: it reused the Jacobian of truth
  2026.  With each truth's own Jacobian the linear prediction tracks the MCMC
  (rho 1.42 / 1.28 / 1.33 vs 1.34 / 1.24 / 1.28; Phi within 1%).
- Over 30 truths with their own Jacobians (reports/linearized/truth_energy.py)
  the informed energy averages 0.77 (sd 0.09) for our "kle" truths and 0.79
  (sd 0.15) for exact prior draws ("kle_raw"): the data are most informative
  in directions where the truth itself has little energy.  Predicted rho
  ~1.32 on average.  At r = 208 the posterior of rho is biased low for a fixed
  rho* = 1.5 whatever the truth; not bad luck.  An SBC study (rho drawn from
  its prior) is now needed to say whether the posterior is calibrated.
- **Why rho is low: the tempered likelihood, not prior shrinkage** (same day;
  reports/linearized/calib_linear.py, exact 1-D posterior of rho in the
  linearized hierarchical model, 400 truths per rho*, SBC with 1000 draws).
  With noise matched to the likelihood, rho is calibrated at every rho*
  (90% coverage 88-93%, bias within 2%, SBC uniform).  With the likelihood as
  run (relative, sigma_rel = 2e-3, which assumes a sensor noise 2.2x (r = 57)
  / 4.6x (r = 208) the actual one, kappa = w sigma_p^2 = 0.20 / 0.048), rho is
  biased low at EVERY rho*: r = 208 -12% to -20%, coverage 31-43%, SBC 37%;
  r = 57 -5% to -10%, coverage 83-86%.  Overstated noise makes the model
  assign too little of the data's variance to the signal (rho^2).  Predicts
  the MCMC: typical median 1.27 at rho* = 1.5, r = 208 (MCMC truths: 1.20,
  1.24, 1.28, 1.34); probably also most of the 3D miss (rho 1.26, 2048
  sensors).  beta and the level are pinned directly by the data: unaffected.
  Confirmation: --likelihood absolute on r = 208, seed 2026 (predicted rho
  ~1.5).

## 2026-10-03 - Learned noise level (`--learn-noise 1`, move E)

- Option 2 of the likelihood fix: keep the relative likelihood, treat
  s2 = sigma_rel^2 as unknown with an inverse-gamma prior IG(a0, b0)
  (default a0 = 2, b0 = a0 x configured sigma_rel^2) and draw it exactly from
  IG(a0 + n_obs/2, b0 + sigma_rel_cfg^2 Phi_cfg) once per iteration (move E,
  after D).  No PDE solve: Phi at the configured sigma_rel is already known,
  and every other move's misfit is rescaled by sigma_rel_cfg^2 / s2.
  K_eff term unchanged.  Prior test: s2 drawn from its prior.
- Check: tests/test_core.py::test_learned_noise_level (32x32, start at truth):
  sigma_rel settles within 0.7-1.5x the matched value sigma_p / |d|.  All 55
  tests pass.  report_soft adds a "sigma_rel (learned)" row (truth column =
  matched value).
- Expected cost: the posterior sharpens to the matched noise level.  On the
  r = 57 Gaussian surrogate (reports/linearized) the global pCN step drops
  0.0124 -> 0.0050 but the IACT per theta-solve only rises 5.7k -> 6.2k
  (median; max 16.4k -> 18.6k): about 10% slower mixing, no extra solves.
- Runs: cluster/README.md "Learned noise level" (r = 57 and r = 208, A x7).

## 2026-10-03 - E1 gets theta-repeat; seed 2074 E5 completed

- `--theta-repeat k` for E1 blocked (`brho/naive.py`): move A k times per
  (xi_b, xi_r) move, 1 + k solves per iteration, run folder e1_blocked_<TAG>
  (`--exp-tag`).  k = 1 is the original path.  The global move refuses k > 1.
  Reports (report_e1, report_mpsrf) count the solves from the chain files.
  Test: tests/test_naive.py::test_theta_repeat (resume exact, solves, refusal).
  Why: the paper's efficiency claim is E5 x k vs E5; the reviewer's question
  "did E1 get the same knob?" needs E1 x k.  Command: cluster/README.md.
- Seed 2074 E5 (runs/cluster_l005_s2074, PR #79) completed to 2.0M solves:
  theta MPSRF < 1.2 at 1.53M solves (final 1.15); E8 2x2 0.91M, 3x3 1.30M.
  Posterior identical in all three: rho 1.34 [1.19, 1.53] (truth 1.5,
  covered), beta 2.52 [2.06, 3.10] (2.3), level 0.90 [0.82, 0.96] (0.83).
  rho matches the tempered-likelihood prediction for this truth (1.33).

## 2026-10-04 - Learned noise level (r = 57, r = 208) and E1 x7 (PR #82)

- Learned sigma_rel (`--learn-noise 1`, A x7, 64 chains, 2.0M solves):
  - r = 57 (runs/e5_kle_2026-09, e5_soft_a7_noise): sigma_rel 8.72e-4
    [8.26e-4, 9.22e-4] vs matched 8.98e-4.  rho 1.64 [1.40, 1.96] (1.5),
    beta 1.95 [1.26, 2.99] (2.3), level 0.833 [0.72, 0.90] (0.835).  Phi at
    the configured sigma 46.9 vs floor 51.7 -> d_eff about 47 of 57 (as
    E[Phi] = floor - kappa d_eff / 2 predicts).  logK 90% coverage 87%.
  - r = 208, seed 2026 (runs/cluster_l005): sigma_rel 4.48e-4 [4.36e-4,
    4.60e-4] vs matched 4.38e-4.  rho 1.445 [1.31, 1.60] (1.5; fixed sigma
    gave 1.20 [1.05, 1.38], missed), beta 2.29 [1.84, 2.85] (2.3), level
    0.854 [0.79, 0.90] (0.832).  logK 90% coverage 88%.
  - Prediction confirmed: the data recover the sensor noise to 2-3% and the
    rho bias disappears at both resolutions.
- Cost: much larger than the Gaussian-surrogate estimate (about 10%).
  - theta MPSRF < 1.2: r = 57 at 1.78M solves (A x7 with fixed sigma: 0.63M,
    2.8x); r = 208 not reached at 2.0M (final 1.67; E5 fixed sigma 1.14M).
  - Move A step 0.0124 -> 0.0050 (r = 57; surrogate predicted exactly this)
    and 0.0245 -> 0.0050 (r = 208).  Lag is spread over theta, not along the
    constant-field direction.
  - Why the surrogate missed it: in the linearized model with beta, rho fixed,
    scaling the likelihood weight w cancels out of the IACT of informed
    directions.  In the full posterior the widths of rho and log beta are set
    by the prior (rho sd 0.17 in both runs at r = 57) while the theta step
    shrinks like 1/sqrt(w), so the theta directions coupled to them slow down
    by up to w-fold.  Hypothesis; testable with a surrogate that includes rho.
  - Next: extend the r = 208 run (experiments/extend_run.py) to 4-6M solves.
- E1 with theta-repeat 7 (runs/e5_kle_2026-09, e1_blocked_a7, 2.0M solves):
  WORSE than plain E1.  theta MPSRF final 1.42 (E1: 1.21); log beta PSRF
  1.41 (1.21), spread of chain means / prior sd 0.31 (0.19); rho and level
  unchanged.  E1's bottleneck is the beta ridge, moved only by block B, and
  theta-repeat cuts B's share of the solves from 1/2 to 1/8.  E5 removes that
  ridge (soft curve prior, moves C and D), which is why the same knob gives
  E5 a 2x gain.  For the paper: theta-repeat pays only once the
  non-identifiability is fixed.

## 2026-10-04 - E9: scale-block pCN for theta (`--theta-innovation scale`)

- Move A on blocks of KLE modes by index (brho/scaleblocks.py): default
  modes 1-20, 21-60, 61-r (`--scale-edges 20,60`; edges >= r dropped, so r = 57
  gets 1-20, 21-57); one block per theta move, drawn with fixed probabilities
  (`--scale-probs`, default equal), each block with its own adapted step;
  combines with `--theta-repeat` (2 + k solves).  Exactly prior-preserving
  (pCN on independent coordinates of N(0, I)); gradient-free; blocks fixed by
  the prior alone.
- Why: data informativeness falls steeply with KLE index (r = 208 surrogate:
  mean diag H 879 for modes 1-20, 2.3 for 151-208), so scale blocks separate
  stiff from soft directions; surrogate gain 1.7-1.8x on the slowest
  direction (reports/linearized/spectral_split*.py, PR #84).  Target: the slow
  learned-noise runs (rho build-up in the fine-scale modes under move A's
  0.005 step).
- Test: tests/test_core.py::test_scale_blocks (block parsing, proposal
  touches one block only, step slots, 2 + k solves, prior preserved).  All 57
  tests pass; smoke run through setup, chains, report_soft, report_mpsrf.
- Runs: cluster/README.md "Scale-block pCN with learned noise" (r = 57 and
  r = 208, theta-repeat 7, 2.0M solves, same budget as the a7_noise runs).

## 2026-10-05 - MPSRF curve: running sums instead of recomputing every window

- report_mpsrf stalled on the extended r = 208 a7_noise run (1325 curve
  points, about 20k kept samples per chain, dim 208): the curve rebuilt the
  full within-chain covariance at every point, cost ~ points x samples x dim^2.
- brho/mpsrf.mpsrf_curve now updates per-chain sums and one pooled sum of
  outer products with only the samples each point adds (the windows are
  nested), so the samples are read once; each point then costs one 208 x 208
  solve and eigen-decomposition.  Samples are shifted by each chain's first
  window sample for conditioning (the shift cancels in the within-chain
  covariances and is added back to the chain means).
- Same numbers as before: equal to the direct computation to 1e-15
  (mpsrf_curve_direct kept as the reference; tests/test_naive.py::
  test_mpsrf_curve_incremental_matches_direct).  7.5x faster at 266 points,
  and the gap grows linearly with the run length.

## 2026-10-05 - E10: log beta collapsed into moves A and B (`--collapse-beta 1`)

- Physics: with a prescribed inflow flux and zero outflow pressure,
  p(beta K) = p(K) / beta exactly (checked to 3e-14).  For fixed (theta, rho)
  the misfit is Phi(lb) = w/2 |d - e^{-lb} g|^2, g = pressures at beta = 1,
  available from the current solve.
- Move: propose theta' (A: global / multiscale / scale block) or w' (B) as
  before, solve once, propose lb' ~ q = Laplace approximation of
  exp(-c Phi(lb)) N(lb; m(rho), s(rho)^2) given the new pressures, accept
  with the joint MH ratio (incl. q(lb | current) / q(lb' | new)).  Exact for
  any q; no extra solve.  Not for block pCN (E8).
- Tests (tests/test_core.py::test_collapse_beta_exact): closed form equals a
  solve; with theta, w fixed the move samples the exact 1-D conditional
  (quadrature) with the Laplace q (acceptance > 0.9) and with a deliberately
  wrong q; prior preserved with scale blocks + theta-repeat; state stays
  consistent with a fresh solve under learned noise.  All 59 tests pass.
- Real-PDE check (16x16 test problem, learned noise, theta-repeat 3): move A's
  adapted step 0.0195 -> 0.0307 (1.6x; about 2.5x in s^2); move B unchanged
  (a rho change alters the pressure shape, not only its magnitude).
  Surrogate (r = 208): step 2.0x, IACT 2.0x.
- Runs: cluster/README.md "E10" (r = 57 to 4.0M, r = 208 to 6.0M solves, with
  scale blocks, theta-repeat 7, learned noise).
## 2026-10-05 - E9 scale-block pCN with learned noise: results (PR #88)

- Same budget and theta share as a7_noise (theta-repeat 7, 2.0M solves).
- r = 57: theta MPSRF < 1.2 at 1.77M solves (a7_noise 1.78M); no gain.
  Mid-run E9 is behind (equal-quality ratio 0.77-0.93), even at the end.
  Hyperparameters mix better: ESS rho 13.1k (a7_noise 8.8k), log beta 46k
  (23k), sigma_rel 90k (47k); level unchanged (948 vs 931).
- r = 208, seed 2026: not converged at 2.0M (final 1.465; a7_noise 1.674 at
  2.0M, reaching 1.465 only at 2.76M): about 1.4x.  Worst single component
  PSRF slightly worse (1.255 vs 1.206).  a7_noise extended to 6.0M crosses
  1.2 at 5.95M (final 1.198).
- Posterior identical to a7_noise: r = 57 rho 1.65 [1.40, 1.97], sigma_rel
  8.72e-4; r = 208 rho 1.457 [1.32, 1.61], sigma_rel 4.47e-4.
- Steps: fine-scale blocks 0.0156 (r = 57) and 0.0124 / 0.0156 (r = 208), 3x
  the global 0.005, as intended; the large-scale block (modes 1-20) stays at
  0.005 and is now visited 1/2 (r = 57) or 1/3 (r = 208) of the theta moves,
  so the large-scale modes, which carry the level (the slowest quantity,
  level ESS about 950), get fewer moves.  Net: no gain where those dominate.
- Root cause of the large-scale stiffness (reports/linearized/collapse_level.py):
  p(beta K) = p(K) / beta exactly, and 75% of trace(H) is the single
  pressure-magnitude direction.  Integrating log beta out of each theta move
  (closed form, no solve) doubles the global pCN step and halves the IACT in
  the surrogate (2.0x on the slowest direction).  Next: implement it.


## 2026-10-05 - E10 first run (r = 57): 13 of 64 chains frozen; fix (lb slice refresh)

- runs/cluster_kle/.../e5_soft_scale_noise_collapse (PR #91), 4.0M solves:
  theta MPSRF final 10.6, level PSRF 10.4.  About 13 chains froze during
  burn-in at Phi 5e3-1.1e5 (config scale; the healthy chains sit at 47) with
  learned sigma_rel up to 0.03 and the level at 1.2-3.4 (truth 0.83): log K
  too high, predicted pressures too small.  Moves A and B never accepted
  again in those chains (their steps adapted to the 1e-3 floor); only C, D
  and E moved them.  The posterior summaries of this run are not usable.
- Cause: the joint move proposes lb' from a Laplace (Gaussian) approximation
  q of its conditional.  Above the optimum Phi(lb) = w/2 |d - e^{-lb} g|^2
  saturates at w |d|^2 / 2 (1/(2 sigma_rel^2) = 1.25e5: the frozen chains'
  ceiling), so the conditional has a plateau but q has Gaussian tails: with
  lb more than ~0.5 above its optimum, q(lb | .) in the reverse ratio is ~0
  and every joint move is rejected.  The chain is still exact, but trapped.
  Reproduced on the 16x16 test problem: start lb_hat + 0.5 ... + 4: 0 of 300
  joint moves accepted; start lb_hat - 1 or lb_hat: 96%.  The exactness test
  of PR #90 started at lb_hat, so it never saw this.
- Fix: before move A, an exact slice-sampling update of lb at fixed (theta,
  w) from exp(-c Phi(lb)) p(lb | rho), closed form, no PDE solve (Neal 2003,
  doubling + shrinkage with the acceptance test; bracket width 2/sqrt(P)
  from (g, rho, c), never from lb).  Slice sampling has no proposal density
  that can vanish.  Same starts: 286-292 of 300 joint moves accepted.
- Test test_collapse_beta_slice_refresh: draws match the exact conditional
  (quadrature) from lb_hat and from lb_hat + 3; the trap (0 of 200) and its
  release after one refresh (> 100 of 200).  60 tests pass.
- Reruns: TAG noise_collapse2 (cluster/README.md "E10").  The r = 208
  noise_collapse run has the same flaw: stop it rather than resume.

## 2026-10-05 - housekeeping: one folder for the 2D study

- PR #91 committed `runs/cluster_kle` (the live 2D study) for the first time.
  It is a superset of the old export `runs/e5_kle_2026-09`; the only
  differences were the MPSRF summaries and comparison figures, recomputed with
  the incremental MPSRF code (PR #87).  `runs/e3_kle_2026-09` and
  `runs/e5_prior_test_kle_2026-09` were byte-identical copies of files in it.
- Removed those three folders (about 38 MB).  `runs/cluster_kle` stays: the
  cluster's sparse checkout reads the frozen kle / truth / curve inputs from
  it, so deleting it would break the runs.
- Repointed reports/linearized (lin.py, screen.py, da_eval.py, da_test.py) and
  the README; `runs/cluster_kle/STUDY.md` lists the runs and their status and
  says what to do with old references to `e5_kle_2026-09`.

## 2026-10-05 - β upgrade: post-processing of 3-move (E4) runs, no PDE solves

- Decision (Aidan): the current project keeps the 3-move hard-curve sampler;
  the soft sampler, learned noise and derived samplers move to a later paper.
  beta coverage comes from post-processing the E4 output.
- brho/beta_upgrade.py, reports/report_beta_upgrade.py.  For each saved
  (rho, theta): delta ~ N(mu, 1/P), P = |c|^2/rho^2 + 1/s(rho)^2,
  mu = (theta . c / rho) / P; log beta = log beta_bar(rho) + delta,
  theta' = theta - (delta/rho) c, so log K is unchanged (up to the 3.4% of the
  constant field outside the KLE span).  Importance weight
  w = exp(a^2/(2 rho^2 P)) / (s sqrt(P)) makes it exact for the model with the
  curve scatter restored.  Derivation in the module docstring.
- Test test_beta_upgrade_level_swap_exact (likelihood off: theta', xi_b, rho
  laws; log K unchanged when T c = 1; detects a dropped weight or a sign error).
  61 tests pass.
- Local validation (not for publication): E4 on the cluster_kle truth (kle,
  sigma_rel 2e-3), 8 chains x 100k iterations (300k solves each), 36,000 kept
  samples:

  | | beta | 90% interval | covers 2.3 |
  |---|---|---|---|
  | hard curve | 2.20 | [2.14, 2.29] | no |
  | upgraded | 2.21 | [1.48, 3.28] | yes |
  | E5 reference (cluster, 64 chains) | 2.14 | [1.43, 3.21] | yes |

  sd(log beta) 0.021 -> 0.242 (E5 0.247).  rho 1.44 [1.21, 1.73] and the
  level 0.870 unchanged (max level shift 1.7e-3).  Weight ESS 99.9%
  (sd log w 0.03): the weights are negligible here, because the data pin
  a = theta . c given rho.  Re-solving 200 upgraded samples changes Phi by
  0.03 +- 0.30 (posterior Phi sd ~ 7): the KLE-span residual is harmless.
  Freezing at one sample (the highest-posterior one, rho 1.92) gives
  2.36 [1.39, 3.98]: covers, but wider and dependent on which sample;
  per-sample is as cheap and preferred.
- To publish: run E4 on runs/cluster_kle (cheat sheet), then the report.

## 2026-10-06 - E4 (3-move) with sigma_rel matched, cluster_kle truth (PR #95)

- 64 chains, sigma_rel = 8.98e-4 (= sigma_p / ||d||, the matched value), but
  150k iterations (450k solves, burn 30k, thin 100), not the planned 667k:
  the job got `--n-total 150000` (an exported E4_ARGS, presumably).
- Not converged: theta MPSRF 2.86, worst PSRF 2.0 (56 of 57 above 1.2); level
  PSRF 1.43.  Phi reaches the noise floor (256) only around 60-75k
  iterations, after the 30k burn-in.  rho and beta PSRF 1.05.
- Posterior so far: rho 1.54 [1.31, 1.82] (truth 1.5, z = 0.33; with 2e-3 the
  r = 57 rho was biased 5-10% low); level 0.866 [0.78, 0.92] (0.835); log K
  90% coverage 88.6%; mean Phi 243 vs floor 256.
- beta: hard curve 2.23 [2.165, 2.33] (covers, at the upper edge, sd 0.05);
  upgraded 2.13 [1.39, 3.25], sd log beta 0.259 (per-sample theory 0.253);
  weight ESS 99.9% again; re-solving 200 samples changes Phi by 0.12 +- 1.07
  (posterior Phi sd 10).
- Next: extend to 667k iterations (2.0M solves) with experiments/extend_run.py
  and RESUME=1, then rerun report_sampler and report_beta_upgrade.

## 2026-10-06 - E4 (3-move) at sigma_rel 2e-3, 2.0M solves: converged; beta upgrade (PR #97)

- runs/cluster_kle_sig2e3 (store and E3 curve copied from cluster_kle: same KLE
  basis, curve and kle truth).  64 chains, 667k iterations (150k first, then
  extend_run.py: the cluster's E4_ARGS had overridden the command line),
  burn 30k, thin 100, 407,680 kept samples.
- theta MPSRF < 1.2 at iteration 390,807 (1.17M solves), final 1.116; worst
  single-theta PSRF 1.064; level PSRF 1.03.
- rho 1.469 [1.229, 1.783] (truth 1.5, z = -0.10): no visible low bias on this
  truth at 2e-3 (the linearized prediction was -5 to -10%; seed 2026 at
  r = 57 sits at the mild end).  level 0.867 [0.754, 0.942] (0.835).  log K 90%
  coverage 94.6%; mean Phi 69.6 vs floor 51.7 (tempered likelihood).
- beta: hard curve 2.208 [2.146, 2.317] (covers at the upper edge, sd 0.054);
  upgraded 2.175 [1.442, 3.264], sd log beta 0.249 (per-sample theory median
  0.240); weight ESS 99.8%; re-solving 200 samples: dPhi 0.004 +- 0.42.

## 2026-10-18 - External Darcy solver on Juno: validation and pilot for the 3D beta(rho) curve

- New (this branch): `brho/darcy_external.py` (driver for a solver kept outside the repo, contract in
  `external/README.md`), `brho/multiscale_mat.py` (multiscale whitened fields from the `.mat`
  matrices), `brho/curvefit.py` (even polynomials of degree 2, 4, 6 and PCHIP of mean log K_eff,
  leave-one-out errors), `experiments/run_external_curve.py`, `cluster/slurm/node_e3_external.sbatch`.
  Tested against a mock solver (`tests/test_external.py`); the numbers below were produced on Juno with
  the same code before it was moved into the package, so a short pilot of the repo version is still to
  be run there.
- Problem: 38x38x45, ell = 2 cells, cylinder plug (49,140 of 64,980 cells active), flux boundary
  7.6313e-6 m/s on the bottom face, pressure on the top, mu = 1e-3 Pa s. Solver: darcy_solver (C + PETSc
  3.18.1 module, CG + GAMG, rtol 1e-8), one MPI rank per solve, 64 solves at a time on one node.
- Solver self-check (Juno): uniform K0 gives Keff/K0 = 1.000000; Keff(3.7 K)/Keff(K) = 3.700000;
  reference field Keff = 1.0360e-13 against K* = 1.0360e-13 from the observed 0.480757 psi (ratio
  1.0000). Kexp = 1.036016e-13 m^2.
- Fields: the 12 subdomain files (500 local modes each) with the SP1 whitening operator (4699 modes,
  matches PHI; the SP2 file has 5313 and does not). Mean per-draw std 0.967, neighbour correlations
  0.88 / 0.88 / 0.88 along x / y / z against 0.89 / 0.89 / 0.88 for log Kref: same smoothness, cell
  ordering consistent. The 3% variance deficit is the truncated basis; the curve and the sampler must
  use the same operator so that rho means the same in both.
- Pilot (64 realizations x 6 rho, 0 failed solves, 14 s on 64 cores). Mean log(Keff/K0) (se):
  0.0002 (3e-4) at rho 0.05, 0.099 (0.007) at 1.04, 0.256 (0.016) at 2.03, -0.285 (0.066) at 3.02,
  -2.14 (0.14) at 4.01, -4.53 (0.22) at 5.0. c0 = 0 within 3e-4 in every fit. A pure quadratic does
  not fit (chi2/dof 286); degree 6 gives a = 0.128, close to the 0.1269 of the production
  beta = 1.84e-14 exp(-0.1269 rho^2) at small rho.
- Finding: for rho above about 3, log Keff is controlled by the inlet face. The boundary condition
  imposes the same flux in every bottom-face cell, so the face-average inlet pressure is dominated by the
  lowest-permeability cells. On the 6 saved fields, log Keff correlates 0.96 / 0.99 / 0.99 (rho = 3, 4, 5)
  with the log of the harmonic-mean permeability of the inlet face, against -0.33 to 0.37 for the outlet
  face (control); the offset is about 4 (ln 90 = 4.5 expected from Lz / (dz/2)). The drop of h(rho) is
  therefore a property of this flux-boundary model, not a solver artifact; it matters if the experiment
  injects differently (nozzle, injection mask).
- Solver cost: 0.5 s and 43 iterations for the uniform field; one solve at large rho took 661
  iterations and 3.1 s (CG + GAMG).
- Pre-existing, unrelated: `tests/test_core.py::test_collapse_beta_slice_refresh` fails on `main` in the
  environment used here (checked on a clean checkout).
- Next: tolerance check (`--rtol 1e-11`, `compare_external_runs.py`), then the full study (2048
  realizations x 40 rho; SE of the mean log Keff at rho = 5 is 0.038 at 2048); keep the PCHIP table as the
  baseline for the sampler and compare with the polynomials by leave-one-out error.

## 2026-10-18 - External solver: GAMG breaks down at large rho; Jacobi-preconditioned CG used (PR #121)

- Repo pilot on Juno (64 realizations x 6 rho, seed 99, same code path as the first pilot): the fields are
  bit-identical to the flat-script pilot and the solver command is identical, but 41 of 384 solves fail
  with `DIVERGED_INDEFINITE_PC` after 5-11 iterations (CG + GAMG): 4 of 64 at rho = 4.01, 37 of 64 at
  rho = 5, none below. The failure is deterministic. The first pilot's driver did not check the KSP reason
  and accepted these stopped iterates (some with a negative pressure drop) as Keff values, so large-rho
  numbers from that pilot are unreliable. Where GAMG converged, both pilots agree exactly.
- Fix: `PETSC_OPTIONS="-pc_type jacobi -ksp_max_it 100000"` (default in the sbatch). Pilot with Jacobi:
  0 failures; agrees with GAMG to <= 6.5e-8 on all solves where GAMG converged (rho 0.05-5); on the 41
  diverged solves the Jacobi/old-recorded ratio has median 0.91 (rho = 4.01) and 0.98 (rho = 5.00).
  Example (one rho = 5 field): GAMG stops with drop 9085 Pa, Jacobi converges to 16240 Pa (3636 it, 1.4 s).
- Code: runs with failed solves are saved (NaN) and their fits skipped instead of crashing in `collect()`;
  parts with a different rho grid are refused; Recovery notes in `external/README.md`.
- Open: tolerance check (`--rtol 1e-11`), then the full 2048 x 40 study.
