#!/bin/bash
# Submit all jobs of the MSM second-round plan (docs/msm_experiment_plan_2026-11-03.md).
# Run from the repo root:   bash cluster/submit_msm_round2.sh        (add DRY=1 to only print the sbatch commands)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
JOBS=(
  msm_exp1_global_s50
  msm_exp2_primal_raw_s50
  msm_exp3a_primal_pinv_s50
  msm_exp3b_primal_aug_s50
  msm_exp6a_alt_L1
  msm_exp6b_alt_L10
  msm_exp6c_alt_L100
  msm_exp6d_alt_L1_nooffset
  msm_exp5b_sequence_exact
  msm_exp2_6_primal_raw_d6
  msm_exp2_8_primal_raw_d8
  msm_exp4_8_mixture_d8
  msm_exp4b_mixture_innovation
  msm_post_rerun_exp4_exp5
)
for j in "${JOBS[@]}"; do
    f=cluster/slurm/$j.sbatch
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
    if [ -n "${DRY:-}" ]; then echo "sbatch $f"; else echo -n "$j: "; sbatch "$f"; fi
done
