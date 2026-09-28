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
- ASSUMED (computeMPSRFcurve was not shared): points at thinned counts
  k = iter_burn, iter_burn + step, ...; window (k // skip, k], i.e. skip = 2
  discards the first half.  Change `a = k // skip` in brho/mpsrf.py if the
  MATLAB function differs.
- E1 blocked run 2026-09 (64 chains, 200k it., r = 57): log beta PSRF 2.10,
  rho 1.21, level 1.15, amplitude 1.11; chains settle at relerr 0.0253 vs
  noise 0.0196 (KLE truncation: the full-rank truth has 5.5% of its energy
  outside the 57 modes).  Kept: full-rank truth, 95% energy.
