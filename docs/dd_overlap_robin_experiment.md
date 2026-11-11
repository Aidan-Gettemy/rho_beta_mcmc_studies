# Robin interface condition and overlapping subdomains (prototype, experiment 9)

Variants of the domain-decomposed sampler of `docs/dd_mcmc_experiment.md` / `docs/dd_flux_experiment.md` that try to
reduce the inconsistency between independently sampled blocks while keeping their fast convergence.

## Robin interface condition (`--robin-alpha`, `brho/msm_dd.py: BlockSolver, Block.set_robin`)
On an interface face the block uses alpha p + K dp/dn = alpha g + h, g the spline pressure, h = -(outward Darcy flux)
from the flux data (cubic interpolation along the line, `FluxData.line_flux`). In the finite-volume scheme a face becomes
a link of conductance 1/(1/T + 1/alpha), T = 2K, to the pressure g + h/alpha; alpha -> infinity is the Dirichlet face of
experiments 7/8. Sides that do not lie on a flux line stay Dirichlet. The flux enters the boundary condition, so there is
no flux penalty (`--flux-sigma-rel` is unused).

## Overlapping subdomains (`--overlap-halo`, `brho/msm_dd_overlap.py`)
Every subdomain is extended by `halo` cells towards each interior interface; local KLE (95% energy), local solve and
interface condition live on the extended rectangle, only the core's sensors enter the likelihood, assembly uses the
core part of the local modes (theta = B c_all as before). halo = 4 puts the extended boundaries of the primal blocks on
the dual flux lines (and vice versa), so Robin works there; halo = 2 has no flux lines.

## Deterministic check (`experiments/diag_interface_conditions.py`, true K, sandbox truth)
Interior pressure error and local sensor misfit are the same for every condition (0.0016-0.0018, ~noise level); only
the interface flux differs: Dirichlet 7.1% (measured faces) / 5.9% (unmeasured), Robin (all faces, cubic flux, alpha = 1)
2.4% / 3.1%, alpha = 0.1: 0.6% / 3.3%. Linearised sd/prior sd of the local coefficients under the study's wide
likelihood weight: 0.85 -> 0.77 (alpha = 1) -> 0.66 (alpha = 0.1); under the per-sensor noise weight 0.47 -> 0.13 -> 0.07.

## Pilot (sandbox, 8 chains x 10,000 iterations, M = 3,000, primal, flux data 0.3%)
Study likelihood (global-norm weight): rms error of the posterior mean of log k / mean sd / sensor misfit:
Dirichlet + flux penalty 0.470 / 0.531 / 0.093; Robin alpha = 1 0.445 / 0.583 / 0.111; halo 2 0.642 / 0.696 / 0.170;
halo 4 0.461 / 0.515 / 0.099; halo 4 + Robin 0.585 / 0.620 / 0.136. No clear gain.

Per-sensor likelihood (`--sigma-rel 0.00013258 --data-relerr 0.003`, weight 1/sigma_p^2, same data):
no flux 0.923 / 0.502; penalty (0.08) 0.740 / 0.369; penalty at the noise level (`--flux-sigma-rel 0.00020045`)
0.138 / 0.004 (overconfident); Robin alpha = 1 0.260 / 0.062; **Robin alpha = 0.1 0.079 / 0.026, sensor misfit 0.0082,
flux misfit 0.052, sup-MPSRF below 1.2 from 6,300 iterations**; halo 4 variants 0.25-0.44, not converged (sup-MPSRF 8-10
after 10,000 iterations, 11-17 local modes). Global pCN at this likelihood does not converge in 20,000 iterations
(12 chains, acceptance 2-4%, MPSRF 12-47), so there is no valid global reference yet.
