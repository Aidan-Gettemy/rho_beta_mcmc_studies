#!/bin/bash
# resubmit the flux 2 x 2 ell 0.05 DD runs as three jobs; the pooled post waits for the other two (run from the repo root)
set -euo pipefail
rm -rf runs/msm_local_kle_2026-11/dd_end/l0.05/pf/l-4-DD_t2026 runs/msm_local_kle_2026-11/dd_end/l0.05/pf/l-4-DD_t2026_pilot
P=$(sbatch --parsable cluster/slurm/dd_end_s4_t2026_l0.05_pf_b2_primal.sbatch)
D=$(sbatch --parsable cluster/slurm/dd_end_s4_t2026_l0.05_pf_b2_dual.sbatch)
sbatch --dependency=afterok:$P:$D cluster/slurm/dd_end_s4_t2026_l0.05_pf_b2_pooled.sbatch
echo "primal $P, dual $D, pooled waits for both"
