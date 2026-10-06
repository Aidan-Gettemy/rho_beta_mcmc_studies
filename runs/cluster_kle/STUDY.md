# runs/cluster_kle: the main 2D study (read this if confused)

32×32, ℓ = 0.1, r = 57 KLE modes, truth `kle` (seed 2026), β† = 2.3, ρ† = 1.5,
relative likelihood with σ_rel = 2e-3 (fixed) or learned.

## This is the only 2D folder now

- **On the cluster** this folder is the live study: `BRHO_ROOT=$PWD/runs/cluster_kle`.
  The chains are git-ignored and stay there; figures, tables, manifests and the frozen
  `kle.npz` / `truth.npz` / `curve.npz` inputs are committed.
- **Keep it in your sparse checkout** (`git sparse-checkout add runs/cluster_kle`). Never
  delete it from the repo: the cluster runs read the frozen inputs from it.
- **Retired 2026-10-05:** `runs/e5_kle_2026-09/` (the old export of this same study),
  `runs/e3_kle_2026-09/` and `runs/e5_prior_test_kle_2026-09/` (byte-identical earlier
  exports). Everything they held is here. If an old note, slide or LOG entry says
  `e5_kle_2026-09`, read `cluster_kle`. Only the MPSRF summaries differ: they were
  recomputed here with the incremental MPSRF code (PR #87).
- No more `export_results.sh` for this study: commit it directly
  (cluster/CHEATSHEET.md section 6).

## Runs and their status

| run folder | what | status |
|---|---|---|
| `e5_soft` | E5, global pCN, 2.0M solves | done |
| `e5_soft_a4`, `_a7`, `_a14` | θ-repeat 4 / 7 / 14 | done |
| `e5_soft_ms_white` | E7, whitened multiscale innovation | done |
| `e5_soft_block` | E8, 2×2 primal + 3×3 dual blocks | done (appendix) |
| `e5_soft_a7_noise` | θ-repeat 7, learned σ_rel | done |
| `e5_soft_scale_noise` | E9, scale blocks, learned σ_rel | done |
| `e5_soft_scale_noise_collapse` | E10, first run | **superseded: about 13 of 64 chains froze (LOG 2026-10-05). Do not use or resume.** |
| `e5_soft_scale_noise_collapse2` | E10 with the log β slice refresh (PR #92) | not planned (next-steps line, realignment 2026-10-05) |
| `e1_blocked`, `e1_blocked_a7` | E1 naive sampler | done |
| `e4_sampler` | E4 (3-move, hard curve), sigma_rel matched 8.98e-4, + beta upgrade | 450k solves (150k it.), not converged (theta MPSRF 2.86): extend |

The other studies: `runs/cluster_l005` (64×64, r = 208, seed 2026), `runs/cluster_l005_s2074`
(and the `_s2063` / `_s2086` stubs), `runs/e5_3d_2026-10` (3D).
