#!/bin/bash
# submit the dd_end stage 8 jobs (run from the repo root)
set -euo pipefail
sbatch cluster/slurm/dd_end_s8_t2026_l-16-DD-H.sbatch
sbatch cluster/slurm/dd_end_s8_t2026_l-16-DD-H4.sbatch
