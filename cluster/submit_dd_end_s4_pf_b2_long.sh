#!/bin/bash
# flux 2 x 2 ell 0.05, longer DD restarts (240k); overwrites the 60k results in place (they stay in the git history); run from the repo root
set -euo pipefail
R=runs/msm_local_kle_2026-11/dd_end/l0.05/pf
rm -rf $R/l-4-DD_t2026 $R/l-4-DD_t2026_pilot $R/l-4-DD+_t2026 $R/l-4-DD+_t2026_pilot
P=$(sbatch --parsable cluster/slurm/dd_end_s4_t2026_l0.05_pf_b2_primal_long.sbatch)
D=$(sbatch --parsable cluster/slurm/dd_end_s4_t2026_l0.05_pf_b2_dual_long.sbatch)
sbatch --dependency=afterok:$P:$D cluster/slurm/dd_end_s4_t2026_l0.05_pf_b2_pooled_long.sbatch
echo "primal $P, dual $D, pooled waits for both"
