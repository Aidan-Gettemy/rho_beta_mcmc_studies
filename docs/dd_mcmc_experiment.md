# Domain-decomposed MCMC with spline interfaces (experiment 7)

Competing method for the multi-scale comparison in `runs/msm_local_kle_2026-11`: the domain-decomposed Bayesian
inversion of Xu, Liao & Li (2022, arXiv:2211.04026). Interface pressure is estimated from the data, frozen, and used
as the Dirichlet condition of independent subdomain MCMC runs with local KLEs. Local samples are projected onto the
global KLE basis exactly as in the MSM method (`theta = B c_all`, Definition 2 / Theorem 1 of the paper).

## Method as implemented (`brho/msm_dd.py`)
- **Interface pressure.** The paper's 1D active-learning GP is replaced by one 2D thin-plate spline through all 512
  checkerboard sensor values (`scipy.interpolate.RBFInterpolator`, smoothing 1e-3, 60 neighbours). Reason: a 1D GP on
  the sensors next to an interface sees an h/2 offset sawtooth (3.7% error on vertical interfaces), and the paper's
  delta_tol = 1e-7 does not carry over to this pressure scale. Spline error against the truth face pressure
  p_face = (K_a p_a + K_b p_b)/(K_a + K_b): mean 0.36-0.38%, worst 0.6% over the 24 primal segments (`summary.json`).
- **Local problem.** Same 32x32 cell-centred FV discretisation; block sides on interfaces get Dirichlet faces with
  T = 2K, outer sides keep the permeameter BCs (p = 1 left, p = 0 right, no flow top/bottom). Local likelihood = global
  Phi restricted to the block's sensors with the global weight w, so sum Phi_i = Phi when the interface data are exact.
- **Sampler.** pCN with likelihood-only acceptance, step adapted to 25% acceptance for the first M iterations, then
  frozen (`StepAdapter`). Seeds `SeedSequence([seed0, grid_tag, block, chain])`, grid_tag primal 0, dual 1.
- **Global fields.** One saved post-M state per subdomain, independent draws per subdomain, assembled `theta = B c_all`.

## Tests
- **A.** Primal 4x4 grid, 6 local modes per block (95% local energy), 64 chains per subdomain, 50,000 iterations,
  M = 5,000. 5,000 assembled global fields.
- **B.** Primal and dual (5x5, 95% local-energy rule, p = 114) grids, 50,000 iterations each. 5,000 + 5,000 assembled
  fields pooled for the pointwise mean and sd. Primal chains are shared with test A (run once, post-processed twice).
- **MPSRF.** Per-subdomain MPSRF with the repo's convention (window from iteration 1,000, skip 2, step 15 saved
  samples, first 10 curve points ignored, interpolated crossing at 1.2, W divisor n-1, max PSRF reported), then the
  sup over subdomains at every curve point. Converged = first iteration at which the sup curve is below 1.2.
- States are saved every 50 iterations. Field plots: contourf, jet.

## Run
```
sbatch cluster/slurm/msm_exp7_dd_primal.sbatch                                   # chains + test A
sbatch --dependency=afterok:<id> cluster/slurm/msm_exp7_dd_dual.sbatch           # chains + test B
```
Outputs: `exp7_dd_primal/{chains.npz,summary.json,post/}`, `exp7_dd_dual/`, `exp7_dd_pooled/post/`.
Local check: `python tests/run_tests.py test_msm_dd`.

## Known caveat (smoke runs, 8 chains x 6,000 iterations)
Frozen-Dirichlet decomposition does not enforce flux continuity, so the assembled fields fit the data globally worse
than the local fits suggest: global relerr ~0.17 (local 0.01-0.09; global pCN plateau 0.0204). Using the exact truth
face pressures instead of the spline does not change this (0.164 vs 0.161), so it is a property of the decomposition,
not of the spline. Smoke posterior sd was about 1.3x the global one, and the mean rms error about 0.7 vs 0.415.
The production runs are the real test.
