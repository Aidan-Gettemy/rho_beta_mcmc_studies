#!/bin/bash
# Submit the paper-MSM study (docs/paper_msm_experiment.md).  Repo root: bash cluster/submit_paper_msm.sh   (DRY=1 prints only)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
JOBS=(
  paper_e0_global_all
  paper_e1_global_one
  paper_e2_raw_all
  paper_e3_proj_all
  paper_e4_proj_one
  paper_e5_smooth_all
  paper_e6_smooth_one
  paper_e7_smooth_sub
  paper_prior
)
for j in "${JOBS[@]}"; do
    f=cluster/slurm/$j.sbatch
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
    if [ -n "${DRY:-}" ]; then echo "sbatch $f"; else echo -n "$j: "; sbatch "$f"; fi
done
