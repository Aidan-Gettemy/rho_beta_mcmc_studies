# rho_beta_mcmc_studies
Advanced multi-parameter permeability modeling and sampling in python.

## 2D (β, ρ) experiments

This code runs the experiments on the 2D permeameter problem: by default a
32×32 grid (ℓ = 0.1, so 3.2 cells per correlation length), 512 chessboard
observations and r = 57 KLE modes. The July studies used 16×16 with 128
observations (`--n 16`); their runs and tests stay reproducible.

**The story so far.** The field log K = log β + ρ G(θ) has two parametrization
degeneracies: the pressures fix the level log β + ρḠ but not its split between
β and the field mean, and they fix the product ρθ but not its split. The naive
sampler (E1) gets stuck on the β ridge. One extra datum, the measured effective
permeability K_eff, gives a soft prior tying β to ρ through the offline curve
β̄(ρ) (E3), and two moves walk the two ridges exactly (E5). At equal cost the
corrected sampler converges and the naive one does not (below). E7 checks the
multiscale (local-KLE) θ proposal inside the corrected sampler. Decisions,
results and dead ends are logged in `LOG.md`.

| Short report | Experiment | Runner | Report |
|---|---|---|---|
| E1 | naive joint (θ, β, ρ) model, pressure data only, one chain per MPI rank; blocked pCN (θ block, (ξ_β, ξ_ρ) block; default) or one global pCN move: the data pin log K but not β | `experiments/run_e1_naive.py` | `reports/report_e1.py` |
| E2 | 5×5 grid of fixed (β, ρ), pCN on θ: every pair fits | `experiments/run_grid_study.py` | `reports/report_grid.py` |
| E3 | offline β̄(ρ) curve from global-KLE Monte Carlo | `experiments/run_beta_curve.py` | `reports/report_curve.py` |
| E4 | 3-step sampler (A pCN θ, B pCN w, C ridge) with β = β̄(ρ) | `experiments/run_curve_sampler.py` | `reports/report_sampler.py` |
| E4 check | the same sampler with the likelihood off (prior test) | `run_curve_sampler.py --prior-test 1` | `reports/report_prior_test.py` |
| E5 | **the corrected sampler**: β free with the soft curve prior log β \| ρ ~ N(log β̄(ρ), s(ρ)²) from K_eff; moves A pCN θ, B pCN w (β fixed), C ridge ρθ (no solve), D level swap (β, θ along the constant field) | `experiments/run_soft_sampler.py` | `reports/report_soft.py` |
| E5 check | the same sampler with the likelihood off | `run_soft_sampler.py --prior-test 1` | `reports/report_soft_prior_test.py` |
| E7 check 1 | multiscale prior test, pure pCN, no PDE: raw / whitened / per-sample-normalized local-KLE innovation vs direct draws, including pointwise field maps | `experiments/run_ms_prior_test.py` | (writes its own figures and tables) |
| E7 check 2 | E5 prior test with the multiscale θ innovation (`--theta-innovation ms_raw` or `ms_white`) | `run_soft_sampler.py --prior-test 1 --theta-innovation ...` | `report_soft_prior_test.py --exp e5_prior_test_ms_white` |
| E7 | E5 with the whitened multiscale θ innovation in move A | `run_soft_sampler.py --theta-innovation ms_white` | `report_soft.py --exp e5_soft_ms_white` |
| convergence | MPSRF / PSRF curves in the group's MATLAB convention (`getMSPRF`, `computeMPSRFcurve`: skip 2, step 15, first 10 points dropped, threshold 1.2), per run and all runs at equal PDE solves | – | `reports/report_mpsrf.py` |
| diagnostic | spectrum of the θ posterior (marginal and given (β, ρ)) and PSRF per principal direction | – | `reports/report_spectrum.py` |
| all | E2 + E3 + E4 on one (β, ρ) plane, plus SUMMARY.md | – | `reports/report_combined.py` |

A finished run can be continued exactly from its checkpoints with
`experiments/extend_run.py --exp <run> --to <iterations>` (the extension is
recorded in the manifest) followed by the run's launcher with `RESUME=1`.

```
make test                 # 42 checks (see tests/)
make quick                # whole pipeline with tiny budgets -> runs/quick/
make all                  # production: test, e2, e3, prior, e4, reports
make e2 ROOT=runs/july_raw ARGS="--truth-mode raw --seed-scheme legacy"   # July data and seeds
make bench                # forward-solve timings on this machine
```

On a cluster, each experiment splits into a setup step, parallel tasks (one per
chain or grid pair) and a collect step. They run either as one single-node job
(`cluster/slurm/node_e2_grid.sbatch` and friends, launched through `srun` or `mpirun`) or as
array jobs (`cluster/submit_pipeline.sh`). `cluster/README.md` also covers getting
the code onto the cluster without the committed results, and bringing results back.

Every config field can be overridden from the command line, for example
`--n-real 256 --n-rho 40` or `--n-total 80000`. The settings live in
`brho/config.py`.

### Main study (September 2026): E1 vs E5 at equal cost

Settings: 32×32, ℓ = 0.1, r = 57 (95.2% energy), relative likelihood with
σ_rel = 2×10⁻³, truth drawn in the samplers' own KLE (`--truth-mode kle`: the
standardized field projected on the 57 modes; the deliberate inverse crime is
stated, and the full-rank runs are the robustness check). β† = 2.3, ρ† = 1.5.
64 chains each, 2.0M PDE solves per chain (E5: 667k iterations × 3 solves;
E1: 1M × 2). Study folder on the cluster: `runs/cluster_kle`; exported results:
`runs/e5_kle_2026-09/`.

| | E5 (soft curve prior + moves C, D) | E1 (naive blocked pCN) |
|---|---|---|
| MPSRF of all θ crosses 1.2 | 1.26M solves | not by 2.0M (1.212) |
| worst single-θ PSRF crosses 1.2 | 0.60M solves | 1.60M solves |
| PSRF / ESS of log β | 1.002 / 36,000 | 1.207 / 467 |
| ρ | 1.47 [1.23, 1.79], PSRF 1.006 | PSRF 1.031 |
| β | 2.14 [1.43, 3.21], sd(log β) 0.247 (predicted 0.244) | – |

E1 converges in principle (proper priors), but its cost is set by the level
ridge and grows as the data sharpen. The multiscale prior tests (E7 checks 1–2)
pass for the whitened innovation and fail for the raw one (its pointwise
variance collapses to 0.36 of the target along the subdomain interfaces);
per-sample normalization collapses the laws of the field's spatial mean and
variance to points. E7 itself is expected to reproduce E5: a whitened
innovation is exactly N(0, I), so move A has the same kernel as plain pCN.

### Layout

```
brho/                     library (no experiment logic)
  config.py               ProblemConfig, NaiveConfig (E1), GridStudyConfig, CurveConfig,
                          SamplerConfig (E4), SoftSamplerConfig (E5/E7), QUICK
  fv.py                   Permeameter (2D/3D): numba CSR fill (NumPy fallback), splu / CG solvers, keff, obs mask
  physics2d.py            Darcy2D = Permeameter((n, n)), kept for compatibility
  kle.py                  KLE (synthesis matrix T, G = T theta): discrete dense/kron, Gauss-Legendre Nystrom
                          (full or separable), quadrature check, save/load; FullRankGaussian (Cholesky draws)
  kernels.py              covariance kernel registry (sq-exp; separable factors)
  quadrature.py           Gauss-Legendre and midpoint rules (port of KLE_matlab)
  store.py                build-once store of KLE bases and reference fields per study
  truth.py                make_synthetic -> Synthetic (G, K, p, d_p, sigma_p, K*_true, K*_obs, level)
  problem.py              Problem = physics + KLE + data; potential(K); frozen runs (kle.npz, truth.npz)
  curve.py                sweep() and BetaCurve: beta_bar(rho, K*) = K*/f(rho), band(), save/load
  naive.py                E1: naive (theta, beta, rho) model, blocked / global pCN, checkpointed chains
  samplers.py             pcn_fixed_pair (E2); ThreeStepSampler (E4); SoftCurveSampler (E5/E7);
                          run_chunk/load_chains (checkpointed, exact resume and extension)
  multiscale.py           local-KLE assemble-and-project innovation, Sigma_ms = B B^T, whitening
                          (Pereira, "Variance control in distribution")
  mpsrf.py                MPSRF / PSRF curves, port of the group's MATLAB getMSPRF / computeMPSRFcurve
  diagnostics.py          psrf, split_rhat, mpsrf, ess, window_curve, summarize, pointwise_coverage,
                          logK_summaries (field summaries in blocks of cells), pointwise_moments
  io.py                   results/ figures/ tables/ under BRHO_ROOT; manifest; write_table
  plotting.py             shared figure helpers (incl. prior_fields: pointwise prior-test maps)
experiments/              runners: write results/<exp>/*.npz + manifest.json only
reports/                  read results, write figures/<exp>/*.png and tables/<exp>/*.{md,csv,json}
tests/                    python tests/run_tests.py (pytest-compatible)
benchmarks/               bench_solver.py: timings by grid size, backend and solver
cluster/                  env.sh, SLURM templates, submit_pipeline.sh, export_results.sh, README
runs/                     exported results (figures, tables, manifests, frozen KLE/truth; no chains)
```

The runners only compute, and the reports only read. Each run records its full
config in `results/<exp>/manifest.json`, and the reports rebuild the problem from
that file, so a figure can never be paired with the wrong settings. The setup step
also saves the run's KLE basis and synthetic data (`kle.npz`, `truth.npz`), and every
chain and report loads them. This matters because the eigenvectors inside repeated
eigenvalues depend on rounding, so they differ across machines and thread counts
(`cluster/README.md`). E1, E4, E5 and E7 chains are checkpointed and resume
when you rerun the same command (on the cluster: resubmit with `RESUME=1`); a
rerun that requests different settings stops and asks for `--fresh`. Reports
read only finished checkpoints (`chain<N>.npz`), so they can run while a job is
still writing.

### The KLE: discrete or Nyström

The field is G = Tθ with T = φ(x_cells)·√λ (`brho/kle.py`). There are two families of builders:

- **Discrete** (`dense`, `kron`): the eigenvectors of the covariance matrix on the cell
  centers. This is the July method and stays the default up to 4,096 cells, so those
  runs reproduce exactly.
- **Nyström** (`nystrom`, `nystrom_sep`): a port of `KLE_matlab`. It solves the
  Fredholm eigenproblem ∫C(x,y)φ(y)dy = λφ(x) on a Gauss–Legendre grid with `--kle-nq`
  nodes per direction, then extends the eigenfunctions to the simulation cells with
  the Nyström formula. For a separable kernel, `nystrom_sep` solves the same weighted
  problem as a tensor product of 1D problems. It is identical to the full solver
  (tested to 1e-13) and far faster: your ℓ = 0.022, nq = 80 example takes 0.24 s
  instead of 93 s, with 1,025 modes; 30³ at ℓ = 0.2 takes 0.02 s instead of 8.5 s.
  It is the default above 4,096 cells.

How the two relate, as checked in `tests/test_kle.py`:

- The discrete KLE is exactly the Nyström method with the midpoint rule at the cell
  centers (they agree to 1e-12).
- Gauss–Legendre converges fast: at ℓ = 0.1, nq = 20 gives eigenvalues to 5e-7 and
  nq = 30 to machine precision. At ℓ = 0.022, nq = 80 is within 7e-5. A good rule
  is nq ≈ 3/ℓ.
- Every separable build refines the rule to 1.5·nq, records how far the kept
  eigenvalues move (`nq_check` in the manifest), and warns above 1e-4.
- Modes tied with the last kept one (the (i, j) / (j, i) pairs on a square) are kept
  together, so the truncated field keeps the kernel's symmetry.

**Build once, then load.** Setup steps take the KLE basis and the reference field
from the study's store (`$BRHO_ROOT/store/`), keyed by exactly the settings that
determine them, and build them only if they are missing. E2, E3 and E4 of one study
therefore share one basis and one truth. E4 refuses a curve built with a different
basis, and each run freezes its own copy (`kle.npz`, `truth.npz`). To compare against
MATLAB, run `benchmarks/kle_compare_matlab.py`.

### Speed and scale

The forward solver (`brho/fv.py`) fills a fixed CSR pattern with a numba kernel,
following `simulators_tools._assemble_slab_coo`; without numba it uses the same
values from vectorized NumPy. The linear solver is picked per dimension. Measured
here without numba (`benchmarks/bench_solver.py`), total per forward solve:

| grid | July setting (spsolve) | new default | solver |
|---|---|---|---|
| 50×50 | 6.7 ms | 5.0 ms | splu, MMD_AT_PLUS_A |
| 100×100 | 32 ms | 22–27 ms | splu, MMD_AT_PLUS_A |
| 200×200 | 201 ms | 156 ms | splu, MMD_AT_PLUS_A |
| 20³ | 368 ms | 25 ms | Jacobi-scaled CG |
| 30³ | 5.5 s | 0.14 s | Jacobi-scaled CG |

The CSR fill is 1–2% of a solve at these sizes, so numba's gain shows up mainly
in the smaller grids and in Python overhead. Run the benchmark on the cluster to
see it.

### Figures and statistics

**Setup:** `fig_setup` (G, log K, p with the observation cells), `fig_kle_spectrum`,
`fig_truth_hist`. The table holds r, energy, σp, K\*, level, Ḡ and the scaling-lemma errors.

**E1** (figures in `e1_blocked/` and `e1_global/`, comparison in `e1_compare/`): `fig_e1_acceptance` (running and per-chain acceptance and step sizes, per block),
`fig_e1_relerr` (‖d − p_obs‖/‖d‖ per iteration against the noise level), `fig_e1_traces`
(β and ρ drift), `fig_e1_plane` (each chain's (β, ρ) samples, over the E2 grid),
`fig_e1_identified` (log β and ρ against the level and amplitude of log K, which the
data do identify; the key number is the spread of chain means relative to the prior
sd), `fig_e1_pressure` and `fig_e1_logK` (fields rebuilt from saved samples with very
different (β, ρ)), `fig_e1_fit`. The tables give per-chain summaries; `e1_summary.json`
holds the PSRF, ESS and prior-relative spreads. The rebuilt samples are saved in
`results/e1_<move>/pfield_samples.npz`. With both moves run, `fig_e1_compare_traces` (log β, ρ, level and relative error against PDE solves) and `fig_e1_compare_identifiability` (spread over prior sd and PSRF per quantity) compare them at equal compute.

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

**E5 / E7** (`e5_soft/`, `e5_soft_ms_white/`): `fig_e5_posteriors` (ρ, β against the
soft prior, level), `fig_e5_joint` (samples in the soft curve band), `fig_e5_traces`,
`fig_e5_field` (true / mean / sd of log K and a slice with the 90% band),
`fig_e5_moves` (acceptance per move and step adaptation). `e5_stats` gives median,
90% interval, truth, coverage, PSRF, split-R̂ and ESS; `e5_summary.json` adds the
pointwise coverage of log K and the moves' steps.

**E5 / E7 prior tests** (`e5_prior_test*/`): `fig_e5_prior_test` (sorted diag cov(θ),
QQ of w and of ξ_β, ρ histogram, laws of the spatial mean and empirical variance of G,
all against direct prior draws) and `fig_e5_prior_fields` (pointwise maps: thinned
sample fields, pointwise mean, pointwise variance and its ratio to diag(TTᵀ), with
the subdomain interfaces for a multiscale innovation).

**Multiscale prior test** (`ms_prior_test/`): `fig_ms_prior_test` (diag cov(θ), laws of
the spatial mean and empirical variance) and `fig_ms_prior_fields` (the pointwise maps),
rows reference / raw / whitened / normalized; the table gives the cov(θ) error, θ
variance range, spatial-mean sd, field variance, pointwise variance ratio range.

**Convergence** (`<run>/fig_mpsrf_*`, `fig_psrf_*`, and `mpsrf_compare/`): MPSRF and
worst-component PSRF of θ and of (θ, log β, log ρ) against iteration, the final PSRF
per component, and all runs on one axis against PDE solves per chain, with the
crossing of 1.2 marked. **Spectrum** (`<run>/fig_spectrum`, `spectrum_compare/`).

**Combined:** `fig_argument` (panel (a) E2, panel (b) E3 band + E4 cloud) and `SUMMARY.md`.

### Modelling decisions (all switchable)

- **Truth** (`--truth-mode`): the default `standardized` follows the short report
  (G centered and scaled, so β = 2.3 and ρ = 1.5 are exactly what the data see).
  `kle` (the September main study) projects that field on the samplers' KLE, so
  there is no truncation floor; `kle_raw` projects the raw draw.
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
- **β in E5/E7:** free, with the soft curve prior log β | ρ ~ N(log β̄(ρ), s(ρ)²),
  s² = (E3 realization scatter of log Keff)² + σ_e² (report.pdf, Remark 3); the
  pressures are the only likelihood term, so K\* is used once. E4's hard
  β = β̄(ρ) gave β an sd of 0.05 around a value 5% off the truth.
- **θ proposal:** plain pCN in every sampler (the group's standard; nothing
  posterior-informed, so E1 and E5 are compared on equal terms).
  `--theta-innovation` (E5 only) swaps in the multiscale innovation, `ms_white`
  (correct, law N(0, I)) or `ms_raw` (wrong law, kept for the prior test), with
  `--ms-blocks` (default 2) and `--ms-local-energy` (0.95).
- **Likelihood:** pressure only, as in the method report (`--use-keff-term 1`
  adds the log-Keff datum to E4). The default is the **relative** form,
  Φ = ‖d − G‖²/(2σ_rel²‖d‖²) = relerr²/(2σ_rel²), using the global normalized L2
  norm. σ_rel is either given directly (`--sigma-rel`) or derived from a pair
  (`--rel-eps` ε, `--rel-n-eff` n_eff) as σ_rel = ε/√n_eff; the default is
  ε = 0.02, n_eff = 128, so σ_rel = 1.77e-3. At the truth Φ ≈ (n_eff/2)(noise/ε)²
  on any grid (73 on 16×16, 61 on 32×32), where the absolute form jumps from 64
  to 256. `--likelihood absolute` gives Φ = ‖d − G‖²/(2σp²). Runs whose manifests
  predate this option (the July runs) load as `absolute`.

### Runs in this repo

Each folder holds figures, tables, manifests and the frozen KLE basis and truth,
but no chains (those stay on the cluster). Export a study with
`bash cluster/export_results.sh $BRHO_ROOT <study folder name>`; the second
argument is the folder under `runs/`, not the git branch.

- `runs/e5_kle_2026-09/`: **the main study** (truth `kle`, σ_rel = 2e-3): E3 curve, E1
  and E5 at 2.0M solves, their MPSRF comparison, the θ spectra, the E5 prior test, and
  the multiscale prior tests (standalone, `ms_raw`, `ms_white`).
- `runs/e5_3d_2026-10/`: **the 3D study** (unit cube 16³, ℓ = 0.2, r = 78, truth `kle`,
  σ_rel = 2e-3): E3 curve, E5 prior test, E1 and E5 at 2.0M solves, MPSRF comparison.
- `runs/e3_kle_2026-09/`, `runs/e5_prior_test_kle_2026-09/`: the E3 curve and E5 prior
  test of the main study as first exported.
- `runs/e1_blocked_2026-09/`, `runs/e2_grid_2026-09/`, `runs/e3_curve_2026-09/`,
  `runs/e4_sampler_2026-09/`, `runs/e4_prior_test_2026-09/`: the full-rank-truth
  (`standardized`) study, 32×32: E1, E2, E3, E4 (the robustness check).
- `runs/july_raw/`: E2 on the July data (see `LOG.md`). Its figures, tables, small results and frozen KLE basis are committed, so the reports redraw without a rerun (`BRHO_ROOT=runs/july_raw python3 reports/report_all.py`).
- `runs/quick/` is for smoke tests and is git-ignored.
