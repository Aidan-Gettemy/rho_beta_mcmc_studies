#!/bin/bash
# submit the dd_end stage 2 jobs (run from the repo root)
set -euo pipefail
sbatch cluster/slurm/dd_end_s2_t2026_a.sbatch
sbatch cluster/slurm/dd_end_s2_t2026_b.sbatch
sbatch cluster/slurm/dd_end_s2_t2026_c.sbatch
sbatch cluster/slurm/dd_end_s2_t2026_d.sbatch
sbatch cluster/slurm/dd_end_s2_t2026_e.sbatch
sbatch cluster/slurm/dd_end_s2_t2026_f.sbatch
