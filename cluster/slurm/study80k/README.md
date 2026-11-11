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

Identical in all arms: `--n-iter 80000 --window1 14000 --window2 66000`, step-size adaptation, likelihood, prior, 24 ranks, seed.
Submit from the repo root:  `sbatch cluster/slurm/study80k/s80k_1_pcn.sbatch` (same for 2 and 3; they are independent).

Storage: the code fixes the layout `results/e6_glam_<tag>`, `figures/e6_glam_<tag>`, `tables/e6_glam_<tag>`, so the shared `s80k_` tag prefix
is what groups the study; the post-processing scripts work unchanged.  Scripts live here; the final comparison report goes to `reports/glam_study80k/`.

Resume: arms 2-5 requeue and resume from the last exchange snapshot (`GLAM_RESUME=1` after a time-out).  Arm 1 CANNOT be resumed (no snapshots) and
refuses to overwrite an existing run.

Next: after arm 1, `python experiments/coef_variance_by_index.py --exp-tag s80k_pcn --start-iteration <tail start>` writes `keep_mask.npz` for arms 4-5
(branch `transplant`).
