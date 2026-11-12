# Plain Gibbs for the Robin flux datum (experiment 13) and the noise-level global reference (experiment 14)

## Model
Robin DD (docs/dd_overlap_robin_experiment.md) with the interface flux h as an unknown of each block chain, prior
h ~ N(h_hat, Sigma0): h_hat = the interpolated flux data, Sigma0 = (eps * rms(h_hat))^2 * exp(-|i-j|/L) along a side
(independent between sides), eps = `--h-prior-rel`, L = `--h-corr-len` (4 cells). For fixed local coefficients c the
block sensor pressures are linear in h, p_s(c, h) = p_s(c, h0) + S (h - h0), so the conditional is Gaussian in closed form:
    P = Sigma0^-1 + w S^T S,   mean = P^-1 [Sigma0^-1 h_hat + w S^T (d - p_s(c, h0) + S h0)],   h | c ~ N(mean, P^-1).
One iteration = a pCN step for c given h (one local solve) followed by an exact draw of h given c (`brho/msm_dd_gibbs.py`,
`--h-prior-rel` in `experiments/run_dd_chains.py`). One target, so the ordinary MPSRF applies. Each block has its own
copy of h (a face shared by two blocks is drawn separately for each; no joint draw across blocks).

## Sandbox pilot (primal, 8 chains x 20,000 it., sandbox truth)
eps 0 (h fixed) / 3% / 20% / 70%: mean error 0.079 / 0.084 / 0.208 / 0.322, mean sd 0.026 / 0.035 / 0.143 / 0.355,
median |e|/sd 2.21 / 1.33 / 0.81 / 0.56, final sup-MPSRF 1.07 / 1.15 / 1.79 / 4.36. The sd follows the linearised
marginal-likelihood estimate; the mean gets worse as h is freed; mixing slows.

## Production (cluster)
`msm_exp13_dd_gibbs{03,10}_{primal,dual}.sbatch`: eps = 3% and 10%, 64 chains, pilot 10k + restart 100k per subdomain,
burn 1k; the dual script also writes the pooled output `exp13_dd_gibbs{03,10}_pooled`.
`msm_exp14_global_sharp_fluxnoise_10M.sbatch`: global pCN, flux at its noise level, `--flux-sigma-rel 0.00020045`
(= 0.003/sqrt(224): per-datum sd equal to the noise), 10M iterations, pilot 100k with 500-iteration windows.

## Flux weight: correction
The global-norm flux weight w_q = 1/(sigma_rel_q^2 ||q_obs||^2) with sigma_rel_q = 0.08 means a per-datum sd of
0.08 ||q_obs|| = 0.08 sqrt(224) rms(q) ~ 1.2 rms flux, i.e. almost uninformative. The exp. 9/11/12 "global + flux"
references therefore use the flux data very weakly; the matching reference for a DD run that trusts the flux data at its
noise level is exp. 14.
