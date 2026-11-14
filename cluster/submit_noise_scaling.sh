#!/bin/bash
# Submit the six exp. 18 jobs (independent).  Run from the repo root:  bash cluster/submit_noise_scaling.sh
set -euo pipefail
sbatch cluster/slurm/msm_exp18_noise_f1_primal.sbatch
sbatch cluster/slurm/msm_exp18_noise_f1_dual.sbatch
sbatch cluster/slurm/msm_exp18_noise_f3_primal.sbatch
sbatch cluster/slurm/msm_exp18_noise_f3_dual.sbatch
sbatch cluster/slurm/msm_exp18_noise_f6p5_primal.sbatch
sbatch cluster/slurm/msm_exp18_noise_f6p5_dual.sbatch
