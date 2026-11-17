#!/bin/bash
# submit the dd_end stage 6 jobs (run from the repo root)
set -euo pipefail
sbatch cluster/slurm/dd_end_s6_t2027_c.sbatch
sbatch cluster/slurm/dd_end_s6_t2027_l0.1_p_b4.sbatch
sbatch cluster/slurm/dd_end_s6_t2027_d.sbatch
sbatch cluster/slurm/dd_end_s6_t2027_l0.1_pf_b4.sbatch
