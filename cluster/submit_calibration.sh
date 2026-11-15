#!/bin/bash
# submit the Phase 0b calibration jobs (run from the repo root)
sbatch cluster/slurm/cal_l0.05_pf_e0.1.sbatch
sbatch cluster/slurm/cal_l0.05_pf_e0.2.sbatch
sbatch cluster/slurm/cal_l0.05_pf_e0.4.sbatch
