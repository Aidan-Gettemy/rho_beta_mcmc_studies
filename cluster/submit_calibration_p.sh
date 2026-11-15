#!/bin/bash
# submit the pressure-only calibration jobs (run from the repo root)
sbatch cluster/slurm/cal_l0.2_p_e0.0093.sbatch
sbatch cluster/slurm/cal_l0.2_p_e0.0124.sbatch
sbatch cluster/slurm/cal_l0.2_p_e0.0161.sbatch
sbatch cluster/slurm/cal_l0.1_p_e0.0248.sbatch
sbatch cluster/slurm/cal_l0.1_p_e0.033.sbatch
sbatch cluster/slurm/cal_l0.1_p_e0.0429.sbatch
sbatch cluster/slurm/cal_l0.05_p_e0.024.sbatch
sbatch cluster/slurm/cal_l0.05_p_e0.032.sbatch
sbatch cluster/slurm/cal_l0.05_p_e0.0416.sbatch
