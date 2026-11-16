#!/bin/bash
# submit the dd_end stage 3 jobs (run from the repo root)
set -euo pipefail
sbatch cluster/slurm/dd_end_s3_t2026_l0.2_p_b2.sbatch
sbatch cluster/slurm/dd_end_s3_t2026_l0.2_p_b4.sbatch
sbatch cluster/slurm/dd_end_s3_t2026_l0.2_pf_b2.sbatch
sbatch cluster/slurm/dd_end_s3_t2026_l0.2_pf_b4.sbatch
sbatch cluster/slurm/dd_end_s3_t2026_l0.1_p_b2.sbatch
sbatch cluster/slurm/dd_end_s3_t2026_l0.1_p_b4.sbatch
sbatch cluster/slurm/dd_end_s3_t2026_l0.1_p_b8.sbatch
sbatch cluster/slurm/dd_end_s3_t2026_l0.1_pf_b2.sbatch
sbatch cluster/slurm/dd_end_s3_t2026_l0.1_pf_b4.sbatch
sbatch cluster/slurm/dd_end_s3_t2026_l0.1_pf_b8.sbatch
