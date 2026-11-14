# Exp. 18 -- noise scaling of the non-overlapping DD (does f x more precise data recover the global sampler's sharpness?)

Claim to test (LOG, 2026-11-13): the DD loses information on the coupling modes, equivalent to about 6.5x more data noise
(linearised global posterior with noise x 6.5 has the DD's sd of 0.03; `scratchpad/lin_halo.py`).  Untested converse: the DD with
6.5x more precise data reaches the global sampler's sd and error at the nominal noise (exp. 14: error 0.0045).

## Design
Non-overlapping DD, matched to the best settings so far (exp. 17 without overlap): Robin alpha 0.1 + plain Gibbs for h (sd 0.03 x rms
face flux; measured faces: the flux noise), interior-flux likelihood, standard 14 flux lines (224 faces), noise-level flux weights,
primal 10 modes / dual 99% local energy, truth seed 2026.  ALL noise levels divided by f (data noise, likelihood sigma_rel, flux noise,
flux weight, Gibbs prior sd on the measured faces):

| f | data noise | sigma_rel | flux weight |
|---|---|---|---|
| 1 (matched control) | 0.003 | 0.00013258 | 0.00020045 |
| 3 | 0.001 | 4.4194e-5 | 6.682e-5 |
| 6.5 | 0.00046154 | 2.0397e-5 | 3.0838e-5 |

Chains must be longer for sharper data (smaller pCN steps): f = 1: 64 chains, 10k pilot + 100k; f = 3: 32 chains, 30k + 1M;
f = 6.5: 32 chains, 50k + 3M, step floor 1e-4 and adaptation window 500 (new flags `--s-min`, `--adapt-every`; defaults unchanged).
Scripts: `cluster/gen_noise_sbatch.py` -> `cluster/slurm/msm_exp18_noise_f{1,3,6p5}_{primal,dual}.sbatch`, `cluster/submit_noise_scaling.sh`.

## How to read it
- f = 1 is the first matched non-overlapping control for exp. 17 (no earlier run had identical settings).
- If DD error and sd at f = 6.5 approach the global's (0.0045 error) the information-loss picture is quantitatively right;
  if they stay well above, other causes (truncation, coupling that no amount of data fixes) matter.  The linearised sd scaling
  with f can be compared with the observed one (sd ~ f^-a).
- Check MPSRF (< 1.2) before reading errors; extend the chains if not converged.
