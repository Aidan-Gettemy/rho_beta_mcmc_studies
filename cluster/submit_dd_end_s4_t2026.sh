#!/bin/bash
# submit the dd_end stage 4 jobs (run from the repo root)
set -euo pipefail
sbatch cluster/slurm/dd_end_s4_t2026_k-4.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_l0.05_p_b2.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_k-16.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_l0.05_p_b4.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_k-64.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_l0.05_p_b8.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_l-4.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_l0.05_pf_b2.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_l-16.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_l0.05_pf_b4.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_l-64.sbatch
sbatch cluster/slurm/dd_end_s4_t2026_l0.05_pf_b8.sbatch
