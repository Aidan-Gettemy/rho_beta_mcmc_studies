#!/bin/bash
# submit the dd_end stage 7 jobs (run from the repo root)
set -euo pipefail
J_dd_end_s7_t2026_c_e0_5=$(sbatch --parsable cluster/slurm/dd_end_s7_t2026_c_e0.5.sbatch)
J_dd_end_s7_t2026_c_e2=$(sbatch --parsable cluster/slurm/dd_end_s7_t2026_c_e2.sbatch)
J_dd_end_s7_t2026_d_e0_5=$(sbatch --parsable cluster/slurm/dd_end_s7_t2026_d_e0.5.sbatch)
J_dd_end_s7_t2026_d_e2=$(sbatch --parsable cluster/slurm/dd_end_s7_t2026_d_e2.sbatch)
sbatch --dependency=afterok:$J_dd_end_s7_t2026_c_e0_5 cluster/slurm/dd_end_s7_t2026_l0.1_p_b4_e0.5.sbatch
sbatch --dependency=afterok:$J_dd_end_s7_t2026_c_e2 cluster/slurm/dd_end_s7_t2026_l0.1_p_b4_e2.sbatch
sbatch --dependency=afterok:$J_dd_end_s7_t2026_d_e0_5 cluster/slurm/dd_end_s7_t2026_l0.1_pf_b4_e0.5.sbatch
sbatch --dependency=afterok:$J_dd_end_s7_t2026_d_e2 cluster/slurm/dd_end_s7_t2026_l0.1_pf_b4_e2.sbatch
