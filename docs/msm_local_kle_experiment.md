# Multiscale (local-KLE) pCN experiment: global pCN vs. primal, whitened and primal/dual MSM samplers

Status: **specification, agreed with Aidan on 2026-11-02; amended the same day (below).**
The Dirichlet-left solver option and its tests (Section 9) are merged. The samplers'
building blocks are in `brho/msm_local.py`; the production sampler is not written yet.

**Sigma_ms rank test (Section 6, step 2), 2026-11-02: the primal grid FAILS** with 4 local
modes per block (r_+ = 58 of 64, six exactly missing directions); the dual grid passes
(r_+ = 64). Numbers: `LOG.md` (2026-11-02) and
`runs/msm_local_kle_2026-11/tables/msm_rank_test/`.
**Decision (Aidan, 2026-11-02):** keep 4 local modes per block and split experiment 3 into
**3a** (whitened with the pseudo-inverse, null directions dropped) and **3b** (whitened with
the report's null-space term). So there are six samplers: 1, 2, 3a, 3b, 4, 5. Experiments
that need the primal grid run with the rank-deficient primal Sigma_ms (r_+ = 58) as
documented in Sections 4-7. A guard in the prior-test and production scripts stops the study
if the primal r_+ is anything but 58 or the dual r_+ anything but 64, since that would mean
the setup changed and the decision must be revisited.

## 1. Question

How do the samplers that propose in *local* KLE coefficients (all subdomains at once,
then projected onto the global KLE) compare with plain global pCN, on a problem with
no hierarchy (no beta, no rho)?

- Experiment 3b (whitened, with the null-space term) must reproduce experiment 1 (global
  pCN) within Monte Carlo error, because the whitened innovation then has exactly the
  global law (`brho/multiscale.py`, Pereira's report). It is a correctness check, not a
  speed-up. Experiment 3a drops the six null directions (the primal Sigma_ms has rank 58),
  so its prior has zero variance along them; comparing 3a with 3b measures what those
  six directions cost.
- Experiment 2 (raw) is expected to show the variance deficit the report predicts.
- Experiments 4 and 5 (raw primal + raw dual) test whether alternating between a
  primal and a dual decomposition removes the bias in the pointwise log-permeability
  standard deviation. The raw proposals are known not to preserve the prior; these
  experiments are run anyway, as a test of that idea.

## 2. Model problem

| item | value |
|---|---|
| domain, grid | unit square, 32 x 32 cells (N = 1024) |
| prior | log kappa = G(theta) = T theta, theta ~ N(0, I_r), squared-exponential kernel exp(-\|x-y\|^2 / (2 ell^2)), ell = 0.1 (`brho/kernels.py`) |
| stochastic dimension | r = 64 (KLE truncated at exactly 64 modes; no beta, no rho: equivalently rho = 1 and no level shift) |
| KLE | `brho/kle.py`, `auto` -> dense discrete KLE for N <= 4096. The builder truncates by energy and keeps tied eigenvalues together; there is no fixed-r option. Checked with the repo's own `build_kle` (ell = 0.1, 32 x 32): `energy = 0.966` gives r = 64 (modes 63 and 64 are an exact tied pair, mode 65 is separate by 24%, so r = 64 does not cut a tie group). For comparison `energy = 0.95` gives r = 57. |
| boundary conditions | pressure p = 1 on the left face, p = 0 on the right face (Dirichlet); no flow on top and bottom |
| forward solver | `brho/fv.py` `Permeameter(..., bc="dirichlet")` (Section 9) |
| observations | pressure at every other cell, a checkerboard of 512 of the 1024 cell centers (`Permeameter.obs_mask`) |
| likelihood | relative-error Gaussian, Phi(theta) = \|\|d - O p\|\|^2 / (2 sigma_rel^2 \|\|d\|\|^2), sigma_rel = 0.003 |

### Data

1. Draw theta_true ~ N(0, I_64) (fixed seed). The truth is therefore in the span of
   the sampler's KLE (the "inverse crime" is deliberate, as in the E5/E7 runs).
2. Solve the forward problem for kappa_true = exp(T theta_true).
3. Observe on the 512 checkerboard cells and add relative noise at sigma_rel = 0.003.
   The noise is generated with the same convention the repo uses for
   `likelihood = "relative"`, so that the data are consistent with the likelihood.
   The resulting Phi floor (Phi at theta_true) is recorded in the manifest before any
   chain runs.

## 3. Common run settings

| item | value |
|---|---|
| chains | 64 per experiment, independent prior draws as starting states, seeds via `SeedSequence([seed0, chain])`, the **same seeds across experiments** so the runs are paired |
| iterations per chain | N = 100,000 |
| adaptation | multiplicative step-size adaptation (`StepAdapter`) toward 25% acceptance, for the first M = 10,000 iterations, then frozen |
| pCN | s_new = sqrt(1 - s^2) x_old + s xi; acceptance min(1, exp(Phi - Phi')), likelihood only |
| one forward solve per iteration | yes, in every experiment |

## 4. The six samplers (experiments 1, 2, 3a, 3b, 4, 5)

Notation: B is the assemble-and-project matrix of `brho/multiscale.py` (column
(k, j) = projection onto the global KLE of the local mode sqrt(mu_j^k) psi_j^k of block k,
extended by zero), P is the L2 projection onto the global KLE, c are local KLE
coefficients, p is the number of local coefficients. Sigma_ms = B B^T = V D V^T,
V_+ the eigenvectors with d_i > tau d_1 (tau = 1e-10), V_0 the rest, and
W_+ = V_+ diag(d_+^(-1/2)) V_+^T. For the primal grid with 4 modes per block
r_+ = 58 and V_0 has 6 columns.

1. **Global pCN (control).** State theta in R^64, innovation N(0, I_64), one step size.
2. **Primal MSM, raw.** 4 x 4 blocks (16 subdomains), 4 local modes per block, so
   p = 64. The chain state is the local coefficient vector c in R^p:
   c' = sqrt(1 - s^2) c + s eta, eta ~ N(0, I_p), **all blocks proposed simultaneously
   with one step size** (no per-block adaptation).
   To evaluate the likelihood, the piecewise (discontinuous at block interfaces) local
   field is built from c', projected onto the global KLE, which gives
   theta = B c' in global coordinates, then log kappa = T theta, kappa = exp(log kappa),
   forward solve, relative misfit, Phi. Acceptance uses the likelihood ratio only.
   The target in c is N(0, I_p) x likelihood, i.e. the implied prior on theta is
   N(0, B B^T), not N(0, I_r). This is "what the method is"; it is not corrected.
3. **Primal MSM, whitened** (Pereira's report, `multiscale_variance_control.pdf`),
   with the whitening placed in the forward map. Because the primal Sigma_ms is
   rank-deficient (r_+ = 58), there are two variants, both with all blocks proposed
   simultaneously and one step size:
   - **3a, pseudo-inverse (null directions dropped).** State c in R^64,
     theta = W_+ B c. The implied prior on theta is N(0, V_+ V_+^T): exactly whitened on
     the 58 reachable directions, zero variance along the 6 null directions
     (cov error sqrt(6/64) = 0.306; pointwise prior variance 0.80-1.00 of the target,
     mean 0.93; computed exactly). The chain can never move theta along V_0.
   - **3b, with the null-space term.** State (c, zeta), zeta in R^6, both stepped by pCN
     with the same s: c' = sqrt(1 - s^2) c + s eta, zeta' = sqrt(1 - s^2) zeta + s nu,
     theta = W_+ B c + V_0 zeta. This is the report's whitened innovation
     xi = W B eta + V_0 zeta, so theta has law N(0, I_64) exactly (operator check
     ~1e-15) and the theta-dynamics are those of global pCN.
   In both, `theta_from_c(..., whiten=True)` (exact whitening) still refuses a
   rank-deficient Sigma_ms; 3a and 3b are explicit modes ("pinv", "augmented").
4. **Primal + dual mixture, raw.** The chain state is the global theta. Each iteration,
   with probability 0.5 use the primal grid and with probability 0.5 the dual grid.
   For the chosen grid g: project the current global field onto g's local KLE
   coefficients (c_g = Pi_g theta, lossy because of the local truncation), take the
   pCN step in c_g (all blocks of g simultaneously), form theta' = P(local field of c'),
   and accept on the likelihood ratio only. **Two step sizes, one per grid, adapted
   independently** to 25%, both frozen after M.
5. **Primal then dual, raw, in sequence.** Phase 1: K = 50,000 iterations of the
   experiment-2 sampler (state c_primal). Stop. For each chain, project the final
   global field onto the dual local basis, c_dual(0) = Pi_dual theta_final. Phase 2:
   K = 50,000 iterations of the same sampler on the dual grid (state c_dual).
   Adaptation runs for M = 10,000 iterations at the start of each phase and is
   restarted in phase 2 with a fresh step size (the dual grid is a different move).
   The projection at the switch is lossy, so Phi may jump there; Phi just before and
   just after the switch is recorded.

### Grids

- **Primal:** 4 x 4 blocks of 8 x 8 cells; 4 local modes per block; p_primal = 64.
- **Dual:** vertices at the centers of the primal blocks, i.e. interfaces at
  (i + 1/2)/4, which is the half-block shift used by `brho/blockpcn.py`. This gives a
  5 x 5 tiling: 9 interior blocks, 12 half-size edge blocks and 4 quarter-size corner
  blocks. Each block gets its own local dimension from the **95% local-energy
  criterion** (with tied eigenvalues kept together, as in `local_truncation`). The
  resulting block dimensions and p_dual are reported before any run.

## 5. Quantities saved (per experiment, per chain; 64 chains)

| quantity | schedule |
|---|---|
| relative error \|\|d - O p\|\| / \|\|d\|\| and Phi | every iteration |
| global theta (for the MSM runs: the theta obtained by projecting the local field onto the global KLE; for experiment 1 the state) | every 50 iterations (from 2026-11-03; the first production runs saved every 100), from iteration 0 |
| local coefficients c | every 50 iterations (from 2026-11-03; the first production runs saved every 100), from iteration 0. Experiments 2, 3a, 3b: the state c (3b also zeta, 6 numbers). Experiment 4: both Pi_primal theta and Pi_dual theta of the current state. Experiment 5: c_primal in phase 1, c_dual in phase 2 |
| acceptance rate | one value per chain per block of 500 iterations after M (experiment 5: after M within each phase; experiment 4: per grid) |
| step sizes | at the end of adaptation, plus the adaptation history |
| Phi just before / after the phase switch | experiment 5 |

### Diagnostics computed from the saved data

1. **Acceptance-rate histograms**: all chains, blocks of 500 iterations after M; pooled
   and per block.
2. **MPSRF** (`brho/mpsrf.py`, Felipe's MATLAB convention), curve and final value, on
   the saved samples after M:
   - on the global theta (all experiments);
   - on the local coefficients c (experiments 2, 3a, 3b; 3b on the joint (c, zeta); experiment 4 on both projections;
     experiment 5 per phase, because the dimension differs between phases, and
     theta across the full run).
   With 1,000 saved samples per chain the sample count exceeds every dimension
   (64 for theta and for the primal c; p_dual to be reported).
   The primal B is square (64 x 64) but has rank 58, so theta = B c (experiments 2, 3a,
   and phase 1 of 5) lives in a 58-dimensional subspace and the within-chain covariance
   of theta is singular. MPSRF on theta for those runs is therefore computed in range
   coordinates V_+^T theta (58 numbers; V_+ from the Sigma_ms eigendecomposition), and
   the full 64-dim theta MPSRF is not reported for them. For 3b and experiment 1 theta
   is full rank and is used as is. The invariance of MPSRF under invertible linear
   maps no longer gives a c-versus-theta consistency check for 2 and 3a (B is not
   invertible), so none is claimed.
   **How MPSRF/PSRF are computed (group convention, from 2026-11-03; also written into every
   `post_summary.json`).** `brho/mpsrf.py` (getMSPRF / computeMPSRFcurve). The window **starts at
   iteration 1,000** of the run (of the phase for the second phase of experiment 5), so the adaptation
   period is not discarded, and grows to the end of the run; every 2nd saved sample (skip 2), a curve
   point every 15 saved samples. With m chains and n samples per chain: W = mean within-chain covariance
   (divisor n-1), B/n = covariance of the chain means (divisor m-1), MPSRF = sqrt((n-1)/n + (m+1)/m
   lambda_max(W^-1 B/n)); PSRF_i = sqrt(V_ii/W_ii), V = (n-1)/n W + (m+1)/m B/n, reported PSRF = max_i
   PSRF_i. Convergence iteration = first point (after the first 10) below 1.2, interpolated. Not
   split-chain. The first production runs (saved every 100 iterations) used a window starting after M
   (10,900 effective) and are to be recomputed with this convention.
3. **Posterior fields.** A random subset (about 5,000, same count for every experiment)
   of the saved post-M global thetas is turned into log kappa = T theta and solved forward
   for the pressure. From these: pointwise mean and pointwise standard deviation of
   log kappa and of the pressure. For experiment 5 an equal number of samples comes
   from each phase (about 2,500 each, post-M of that phase); the results are shown
   per phase and pooled. The standard-deviation maps of experiments 2-5 are shown
   next to experiment 1, the posterior mean next to the truth.

## 6. Tests before any sampling (in this order)

1. **Solver tests** (Section 9): `tests/test_dirichlet.py`.
2. **Sigma_ms rank test.** Sigma_ms = B B^T must **not** be rank deficient. Exactly:
   1. Build the primal blocks, local KLEs (4 modes per block) and B (64 x 64), and the
      dual blocks, local KLEs (95% energy) and B_dual (64 x p_dual), from the same
      global KLE (r = 64) used by all experiments.
   2. Form Sigma_ms = B B^T and its eigendecomposition Sigma_ms = V D V^T (descending d).
   3. Report, for each grid: p, the singular values of B, the eigenvalues d_1 and
      d_min, the condition number d_1 / d_min, and r_+ = #{d_i > tau d_1} with
      tau = 1e-10 (the report's tolerance).
   4. For the primal grid also report the operator check
      \|\|W Sigma_ms W^T + V_0 V_0^T - I\|\|_F / sqrt(r) (machine precision expected).
   5. **Pass:** r_+ = r = 64 for the primal grid and for the dual grid. **Fail:** r_+ < r
      for either grid. A failure stops the study before any sampling and is reported
      with the numbers above; nothing is "fixed" silently (for example, by adding a
      coarse move or changing the local dimension).
      **Outcome (2026-11-02):** primal FAILS (58), dual passes (64). By decision of Aidan
      the study proceeds with 4 modes per block and the 3a/3b split; the production and
      prior-test scripts guard that primal r_+ == 58 and dual r_+ == 64 exactly.
3. **Prior tests** (Phi = 0, no solves), experiments 2-5, against direct sampling of
   theta ~ N(0, I_64), as in `experiments/run_ms_prior_test.py`:
   Frobenius error of cov(theta) - I, the law of the spatial mean of the field and the
   law of its empirical variance. Experiments 1, 2, 3a, 3b, 4, 5. Expected: 1 and 3b pass; 3a shows
   the six-direction deficit (cov error about 0.306); 2, 4, 5 show the raw variance deficit. All results are reported before the posterior runs.
4. **Short pilot** of each experiment (small N) to check the pipeline and the Phi
   floor, before the production runs.

## 7. Expectations (to be compared with the results, not assumed)

- Experiment 1 and experiment 3: same posterior within Monte Carlo error; MPSRF
  similar.
- Experiment 2: too-narrow pointwise standard deviation of log kappa relative to experiment 1.
- Experiments 4 and 5: whether the alternation reduces that bias is the open question;
  the proposals are not prior-preserving, so exact agreement with experiment 1 is not
  expected.
- Misfit: the truth is inside the KLE span, so the relative error should reach the noise
  level (sigma_rel = 0.003) for the samplers that target the correct posterior.

## 8. Cluster plan

Six self-contained sbatch scripts (experiments 1, 2, 3a, 3b, 4, 5), one per experiment, in `cluster/slurm/` (all run flags
written in the script itself, so rereading it shows how the run was made; a copy per
experiment), one chain per CPU (64 CPUs), each script activating the conda environment
explicitly (on Juno: `module load miniconda/24.11.1`, then
`source activate /home/arg210006/.conda/envs/gd24_numba`). Output under
`runs/msm_local_kle_2026-11/<experiment>/`. Cost: 64 chains x 100,000 iterations x one
32 x 32 forward solve per iteration per experiment.

## 9. Dirichlet-left solver (first piece of code)

The existing `Permeameter` prescribes an influx on the left face. A new opt-in option,
`Permeameter(shape, bc="dirichlet")` (and the same keyword on `Darcy2D`), prescribes
p = 1 on the left and p = 0 on the right. The default (`bc="influx"`) path is not
changed. Discretization: each boundary cell gets a half-cell transmissibility
T = (area / h_x) 2 K_cell to its Dirichlet face, added to the diagonal, with
T p_face on the right-hand side. The matrix stays symmetric positive definite.
The effective permeability is Keff = Q / (p_left - p_right) with
Q = sum over left cells of T (p_left - p_cell).

Tests (`tests/test_dirichlet.py`, run by `python3 tests/run_tests.py`):

| test | what it checks |
|---|---|
| constant K | p = 1 - x at the cell centers to machine precision, Keff = K |
| K = K(y) only | p = 1 - x exactly; Keff = arithmetic mean of K(y) |
| K = K(x) layered | Keff = harmonic mean exactly; p profile equals the analytic piecewise-linear series solution |
| independent reference | explicit per-face assembly (small n) reproduces the solver's matrix and pressure |
| conservation | left flux = right flux; residual A p - b ~ 1e-10 or better; 0 <= p <= 1 (maximum principle) |
| scaling | p(cK) = p(K), Keff(cK) = c Keff(K) |
| solvers / backends | splu, spsolve and cg agree; numba and NumPy fills agree (if numba is installed) |
| convergence | on a smooth K(x, y), Keff converges at second order under mesh refinement |
| 3D and shape | small cube: constant-K check; 32 x 32 has a 512-cell checkerboard `obs_mask` |
| regression | the default influx path is unchanged (existing `test_fv.py`) |
