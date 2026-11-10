# Domain decomposition with interface flux data (experiment 8)

Idealized extension of `docs/dd_mcmc_experiment.md`: extra data, measured flux across the interfaces, are added to
the likelihood so that independently sampled subdomains are forced to agree on the flux across their common faces.
Question: with such data, how far does domain decomposition become feasible (global-solve misfit of assembled fields,
log-k error and spread against the global sampler, and how fast the small independent subdomains converge)?

## Flux data (`brho/msm_flux.py`)
- Faces: the interface lines of both decompositions, n = 32: face indices 4, 8, 12, 16, 20, 24, 28 in x and in y
  (primal 8/16/24, dual 4/12/20/28). Every other face along each line (even along-line index): 14 lines x 16 = 224 data.
- Flux through a face between cells a (lower/left) and b, in the +x / +y direction: q = T_ab (p_a - p_b),
  T_ab = 2 K_a K_b / (K_a + K_b), the finite-volume flux of the forward model.
- Noise as for the pressure data: sigma_q = 0.003 ||q_true|| / sqrt(224), from its own random stream
  (SeedSequence([2026, 99])), so the pressure data and all earlier results are unchanged.
- Likelihood term of the same global-norm form as the pressure term:
  Phi_q = w_q/2 ||q_obs - q(K)||^2, w_q = 1/(sigma_rel_q^2 ||q_obs||^2).
  **sigma_rel_q = 0.08, not 0.003.** Reason: the blocks' interface pressure is a spline fitted to the pressure data
  (error ~0.4%), and the one-sided block flux computed from the true permeability and the spline faces differs from the
  true flux by 7.1% (primal) / 7.7% (dual) in relative L2 on the measured faces (check on the seed-2026 truth).
  At 0.3% weight the true permeability would have Phi_q ~ 280 per block set and the flux term would fight the spline
  error instead of enforcing continuity. The same level is used for the global reference, so the three runs share one
  likelihood definition. Both numbers are flags (`--flux-noise-rel`, `--flux-sigma-rel`).

## Samplers
- **Global + flux (`run_msm_chains.py --kind global --flux-noise-rel 0.003`)**: the usual global pCN with potential
  Phi_p + Phi_q from one forward solve (`FluxGlobalData`). Reference for everything below.
- **DD + flux (`run_dd_chains.py --grid primal|dual --flux-noise-rel 0.003`)**: each block adds Phi_q,i for the measured
  faces on its own interface sides, using the block's one-sided flux 2 K_cell (p_face - p_cell) with the spline face
  pressure as Dirichlet value. The primal blocks see the primal lines (96 data, each counted by both neighbours), the dual
  blocks the dual lines (128). Blocks stay independent.

## Runs (`cluster/slurm/msm_exp8_*.sbatch`)
`msm_exp8_global_flux` (64 chains, 100k, M = 10k), `msm_exp8_ddflux_primal` (test A), `msm_exp8_ddflux_dual`
(test B, pooled with the primal chains). DD: 64 chains per subdomain, 50k iterations, M = 5k, states every 50. The
iteration budget and the MPSRF start are to be revisited (the flux blocks are expected to converge in about 4k
iterations; `--mpsrf-start` and `--adapt` are flags of the scripts).

Post-processing as in experiment 7, plus the flux misfit of the assembled global fields
(`assembled_flux_relerr`: ||q_obs - q(assembled)|| / ||q_obs|| from the full forward solve).

## Tests
`python tests/run_tests.py test_msm_dd`: flux layout and noise, global potential at the truth, and exact face pressures
+ true K reproduce the true flux on every measured face (so the block flux potentials add up to the noise-only value).

## Protocol
Same two-stage protocol as `docs/dd_mcmc_experiment.md` (pilot of 10,000 iterations to find the step size, then restart
from dispersed prior draws with the step size frozen, 1,000 iterations burned, MPSRF every 100 iterations from iteration
1,000). Budgets: 50,000 per subdomain, 100,000 for the global problem. `msm_exp8_*` scripts replace the earlier single-stage
settings (M = 5,000 / 10,000).
