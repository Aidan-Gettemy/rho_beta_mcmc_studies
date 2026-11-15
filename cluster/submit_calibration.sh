#!/bin/bash
# submit the Phase 0b calibration jobs (run from the repo root)
sbatch cluster/slurm/cal_l0.05_p_e0.01.sbatch
sbatch cluster/slurm/cal_l0.05_p_e0.0145.sbatch
sbatch cluster/slurm/cal_l0.05_p_e0.02.sbatch
sbatch cluster/slurm/cal_l0.1_p_e0.011.sbatch
