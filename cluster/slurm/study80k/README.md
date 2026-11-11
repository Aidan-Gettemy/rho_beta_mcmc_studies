# Study s80k: pCN vs GLAM (one anchor / many anchors) vs masked GLAM, 80000 iterations

Real problem (external Darcy solver + GPU tracer, 6001-dim whitened state, 24 chains, random initialization, `--seed0 2026` in every arm so all
arms start from the same 24 states).  GLAM exchanges are NOT stopped in any arm: GLAM is framed as a population-based optimization
heuristic, not as exact MCMC.  Only arm 1 is the correctness reference.

| arm | tag (results/e6_glam_<tag>) | script | what it is |
|---|---|---|---|
| 1 | `s80k_pcn`   | `s80k_1_pcn.sbatch`   | plain pCN, no exchanges: baseline and goal; source of the mask |
| 2 | `s80k_glam1` | `s80k_2_glam1.sbatch` | GLAM, one anchor (restart all chains from the best state, radius 0.02) |
| 3 | `s80k_glamN` | `s80k_3_glamN.sbatch` | GLAM, many anchors (r24m3 settings) |
| 4 | `s80k_glam1_mask` | (later) | arm 2 + `--restart-keep-mask` |
| 5 | `s80k_glamN_mask` | (later) | arm 3 + `--restart-keep-mask` + `--selection-constrained-only` |

Identical in all arms: `--n-iter 80000`, `--fixed-steps` (NO step-size adaptation; `--window1 0 --window2 80000`), likelihood, prior, 24 ranks, seed.

**Before submitting, fill in the step sizes** (`FILL_IN` in `GLAM_FLAGS`; the script refuses to start otherwise).  On Juno:
`python -c "import json; print(json.load(open('results/e6_glam_r24m3/adaptation.json'))['beta'])"` gives `[beta_G, beta_L]` (same for `e6_glam_pcn`).
Arm 1 takes beta_G from the pcn run; arms 2-3 take beta_G and beta_L from r24m3 (see the note on comparability in the hand-over message).
Submit from the repo root:  `sbatch cluster/slurm/study80k/s80k_1_pcn.sbatch` (same for 2 and 3; they are independent).

Storage: the code fixes the layout `results/e6_glam_<tag>`, `figures/e6_glam_<tag>`, `tables/e6_glam_<tag>`, so the shared `s80k_` tag prefix
is what groups the study; the post-processing scripts work unchanged.  Scripts live here; the final comparison report goes to `reports/glam_study80k/`.

Resume: all arms requeue and resume (`GLAM_RESUME=1` after a time-out).  Arms 2-5 resume from the last exchange snapshot; arm 1 from its
`--snapshot-every 500` snapshots (at most 500 iterations lost).  A finished arm-1 run can be extended with a larger `--n-iter` and `GLAM_RESUME=1`.
A run that already holds chains refuses to start without `GLAM_RESUME=1`.

Next: after arm 1, `python experiments/coef_variance_by_index.py --exp-tag s80k_pcn --start-iteration <tail start>` writes `keep_mask.npz` for arms 4-5
(branch `transplant`).
