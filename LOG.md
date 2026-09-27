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

## Pending decisions before E3 and E4 at full budget

1. Truth: `standardized` (default) or `raw`
2. Curve statistic: `ratio_of_mean` (default) or `mean_of_ratio`
3. Accepted: the ridge move costs one solve with the curve (3 solves per iteration)
4. Accepted: the ρ prior is truncated at the curve range, ρ ≤ 5 (1.6% of prior mass)
