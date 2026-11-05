#!/bin/bash
# Submit the nine jobs of the grid-refinement study (docs/grid_arms_experiment.md).
# Run from the repo root:   bash cluster/submit_grid_arms.sh        (add DRY=1 to only print the sbatch commands)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
JOBS=(
  grid_arm1_check_n16
  grid_arm1_check_n32
  grid_arm1_check_n64
  grid_arm2_fixed_n16
  grid_arm2_fixed_n32
  grid_arm2_fixed_n64
  grid_arm3_abs_n16
  grid_arm3_abs_n32
  grid_arm3_abs_n64
)
for j in "${JOBS[@]}"; do
    f=cluster/slurm/$j.sbatch
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
    if [ -n "${DRY:-}" ]; then echo "sbatch $f"; else echo -n "$j: "; sbatch "$f"; fi
done
