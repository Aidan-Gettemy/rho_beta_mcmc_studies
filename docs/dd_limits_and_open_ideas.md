# Domain-decomposed MCMC: what it can and cannot do (note of 2026-11-13, to come back to)

Setting: 32x32 Darcy, 512 sensors, 64-mode global KLE (ell 0.1), 4x4 primal / 5x5 dual blocks. All numbers below are from the
sandbox or cluster runs listed in LOG.md (sandbox and cluster truths differ; never mix them).

## Advantages (supported by the runs)
- Block chains converge fast and in parallel; wall time is set by one block, not the whole field.
- Best interface treatment found: Robin coupling with Gibbs updates of the interface flux (non-overlapping 10/0.99: about 0.03).
- Overlap halo 2 at 99% energy: 0.021 (dual) / 0.025 (primal) on the cluster, pooled 0.018 (exp. 17), but UNCONVERGED (MPSRF about 2)
  and without a matched non-overlapping control. Global references: same-basis 0.0105, 64-mode global pCN 0.0045.

## Limits (supported by the runs)
- 95% local energy: truncation floor 0.097-0.17. 99%: limited by information and by mixing (step shrinks with modes and sensors).
- Neumann variant, blended (partition-of-unity) assembly: no gain (blend within +-8% of core assembly, sandbox, halo 2).
- Information limit: error about 0.03-0.04 even with the exact interface; the global problem needs a noise multiplier of about 6.5
  to reach the DD's sd of 0.03, so the data-per-mode ratio alone does not explain the gap.

## Why (the reasoning to keep)
1. The likelihood is local (exact pressure + flux on a block boundary decouples the forward solve), but the PRIOR is not. The global
   posterior has 64 coefficients informed by all 512 sensors. The DD represents them as 96-160 local coefficients: each long-wavelength
   mode is learned about 16 times from about 1/16 of the data, and nothing forces the copies to agree.
2. Every fix restores consistency by coupling the blocks (overlap, Robin, neighbour conditioning), and that brings back the global
   problem's stiffness: information shared between blocks and slow mixing are two sides of the same posterior correlation.
3. (User, 2026-11-13) Making the DD work would need the long-range correlations known a priori; but changing a long-range mode changes
   the field in every subdomain, so every subproblem has to be re-solved and checked. Even with cheap subproblems the total work of one
   such move is a full set of block solves, i.e. the cost of a global solve. Parallelism can cut the wall time of that move, not the work;
   at this size the global forward solve is about 1 ms per iteration, so there is nothing to gain here.

## Checks done on the neighbour-conditioned (DDKLE, Panunzio-Cottereau-Puel) idea (2026-11-13)
Script: `experiments/dd_frozen_directions_check.py` (prior only). Fraction v of block prior variance left after conditioning on the 8 neighbours:
- Exact 64-mode prior: primal 6 modes, 32 of 96 directions with v < 0.01; 10 modes, 136 of 160 (72 exactly determined); dual 99%: all 190.
  Given all other blocks every direction is determined (64 degrees of freedom shared by all blocks).
- DDKLE target (manual's G): fine at 95% (min v 0.02) but not a valid covariance at 99% and above (min eigenvalue -0.026 at 10 modes).
Posterior check on stored sandbox chains (6 modes, non-overlapping; 8 chains x 100 saved states, so rough):
- Stage-1 posteriors are close to Gaussian (chi2 quantile ratios 0.96-1.06 sharp likelihood; max |skew| 0.3-0.6).
- With the sharp likelihood the stage-1 posterior sd is only 1-2% of the prior sd, so marginalising over it differs little from a frozen state.
- Prior-data conflict z^2 (expected about 6): 506 and 1038 for stage-2 blocks, 2804 for a block with 6 earlier neighbours, 12841 for a block
  with 8 earlier neighbours. The neighbours' truncated coefficients do not predict the block interior: conditioning turns the block interior
  into an interpolation problem. Not done: the same check at 10/0.99 (needs sandbox non-overlapping 99% chains, about 15 min).
The sequential/one-pass schemes discussed (frozen state, or Gaussian summary of the earlier posterior) are different models from the global
posterior: earlier blocks never see later data. Not built.

## Open items
- Matched non-overlapping control at 99% on the cluster; extend the four unconverged 99% exp. 17 runs; intermediate energy (97-98%).
- Larger-field test (the one place DD could win on speed: global posterior dimension and pCN step grow with area, block cost does not;
  the error floor remains, so the claim would be speed at a given accuracy).
- Larger data noise (truncation would matter less) and the 10/0.99 conflict check above.
