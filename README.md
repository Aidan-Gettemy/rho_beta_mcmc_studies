# rho_beta_mcmc_studies
Advanced multi-parameter permeability modeling and sampling in python.

## 2D (β, ρ) experiments

This code reproduces three experiments on the 2D permeameter problem
(16×16 cells, ℓ = 0.1, 128 chessboard observations, r = 56 KLE modes):

| Short report | Experiment | Runner | Report |
|---|---|---|---|
| E2 | 5×5 grid of fixed (β, ρ), pCN on θ: every pair fits | `experiments/run_grid_study.py` | `reports/report_grid.py` |
| E3 | offline β̄(ρ) curve from global-KLE Monte Carlo | `experiments/run_beta_curve.py` | `reports/report_curve.py` |
| E4 | 3-step sampler (A pCN θ, B pCN w, C ridge) with β = β̄(ρ) | `experiments/run_curve_sampler.py` | `reports/report_sampler.py` |
| E4 check | the same sampler with the likelihood off (prior test) | `run_curve_sampler.py --prior-test 1` | `reports/report_prior_test.py` |
| all | E2 + E3 + E4 on one (β, ρ) plane, plus SUMMARY.md | – | `reports/report_combined.py` |

```
make test                 # 7 checks: scaling lemma, KLE, July data, curve, prior invariance
make quick                # whole pipeline with tiny budgets -> runs/quick/
make all                  # production: test, e2, e3, prior, e4, reports
make e2 ROOT=runs/july_raw ARGS="--truth-mode raw"   # July data exactly
```

Every config field can be overridden from the command line, for example
`--n-real 256 --n-rho 40` or `--n-total 80000`. The settings live in
`brho/config.py`.

### Layout

```
brho/                     library (no experiment logic)
  config.py               ProblemConfig, GridStudyConfig, CurveConfig, SamplerConfig, QUICK
  physics2d.py            Darcy2D: FV solve, keff (Q L / A dp with face reconstruction), obs mask
  kle.py                  KLE: field(theta), project, spatial_mean, c_one, Cholesky draws
  truth.py                make_synthetic -> Synthetic (G, K, p, d_p, sigma_p, K*_true, K*_obs, level)
  problem.py              Problem = physics + KLE + data; potential(K) -> (Phi, p_obs, Keff)
  curve.py                sweep() and BetaCurve: beta_bar(rho, K*) = K*/f(rho), band(), save/load
  samplers.py             pcn_fixed_pair (E2); ThreeStepSampler + run_chunk/load_chains (E4)
  diagnostics.py          psrf, split_rhat, mpsrf, ess, window_curve, summarize, pointwise_coverage
  io.py                   results/ figures/ tables/ under BRHO_ROOT; manifest; write_table
  plotting.py             shared figure helpers
experiments/              runners: write results/<exp>/*.npz + manifest.json only
reports/                  read results, write figures/<exp>/*.png and tables/<exp>/*.{md,csv,json}
tests/                    python tests/run_tests.py (pytest-compatible)
```

The runners only compute, and the reports only read. Each run records its full
config in `results/<exp>/manifest.json`, and the reports rebuild the problem from
that file, so a figure can never be paired with the wrong settings. E4 chains are
checkpointed every `chunk` iterations and resume when you rerun the same command.

### Figures and statistics

**Setup:** `fig_setup` (G, log K, p with the observation cells), `fig_kle_spectrum`,
`fig_truth_hist`. The table holds r, energy, σp, K\*, level, Ḡ and the scaling-lemma errors.

**E2:** `fig_grid_beta_rho` (the main 5×5 picture), `fig_grid_fit`, `fig_grid_slice`,
`fig_grid_traces`, `fig_grid_theta_norm` (‖θ‖²/r per pair, showing the weak prior
penalty), `fig_grid_fields` (a 5×5 panel of recovered log K). The table gives, per pair,
RMS/σ, mean and best Φ, acceptance, ‖θ‖²/r and the final step size.

**E3:** `fig_curve_beta` (β̄(ρ) with ±1 std and 5–95% bands, the three statistics,
and the truth), `fig_curve_f` (the ensemble Keff at β = 1 against the 2D Matheron
value 1, KLE vs Cholesky), `fig_curve_scatter`, `fig_curve_hist` (the βᵢ at ρ†),
`fig_curve_level`. The tables give the curve per ρ; the summary adds β̄(ρ†) vs β†,
the anchor β̄(0) = K\*, and the KLE–Cholesky shift.

**E4:** `fig_e4_posteriors` (ρ, β, level), `fig_e4_joint` (samples on the curve band),
`fig_e4_field`, `fig_e4_traces`, `fig_e4_convergence` (MPSRF of (ρ, level) and PSRF of
ρ, β, level, Φ, ‖θ‖² and the leading θ), `fig_e4_ppc`, `fig_e4_amplitude`,
`fig_e4_adaptation`. The `e4_stats` table gives, per quantity, the median, 90%
interval, mean, sd, truth, z, whether the interval covers the truth, PSRF,
split-R̂ and ESS. There is also a per-chain table, and the summary adds coverage,
mean Φ against the n_obs/2 floor, and the relative sd of ρ against 1/√(2r).

**Prior test:** `fig_prior_test` compares cov(θ), the law of Ḡ, ρ against the
truncated lognormal, and a QQ plot of w.

**Combined:** `fig_argument` (panel (a) E2, panel (b) E3 band + E4 cloud) and `SUMMARY.md`.

### Modelling decisions (all switchable)

- **Truth** (`--truth-mode`): the default `standardized` follows the short report
  (G centered and scaled, so β = 2.3 and ρ = 1.5 are exactly what the data see).
  `raw` reproduces the July data bit for bit (σp = 3.61e-3, K\*_obs = 3.19, Ḡ = 0.286).
- **Curve statistic** (`--statistic`, or `--curve-statistic` in E4): the default
  `ratio_of_mean` gives β̄ = K\*/mean Keff (short report). `mean_of_ratio` is
  method-report eq. 8, and `median` is also available. The raw Keff values at
  β = 1 are stored, so switching statistics never needs new solves.
- **Curve fields:** truncated global KLE with the same 56 modes as the sampler.
  The Cholesky sweep is also stored for the consistency figure.
- **ρ range:** 0.05 to 5, with the exact anchor f(0) = 1. Outside the range, β̄ is
  undefined and the proposal is rejected, so the ρ prior is truncated at 5.
  That removes 1.6% of the prior mass at s_ρ = 0.75.
- **Step C costs a solve here.** The ridge move preserves ρθ, but β̄(ρ′) ≠ β̄(ρ),
  so Φ is re-evaluated (method report eq. 11). In the hierarchical (m, σ) model
  this step is free. Each E4 iteration therefore costs three solves.
- **Likelihood:** pressure only, as in the method report. `--use-keff-term 1`
  adds the log-Keff datum.

### Runs in this repo

- `runs/july_raw/`: E2 on the July data (see `LOG.md`). Its figures, tables and small results are committed, so the reports redraw without a rerun (`BRHO_ROOT=runs/july_raw python3 reports/report_all.py`).
- `runs/quick/` is for smoke tests and is git-ignored.
