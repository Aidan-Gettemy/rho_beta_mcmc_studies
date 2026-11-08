#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24            # the 24 chains run as threads of one process (ThreadWorld); more cores = faster
#SBATCH --mem=64G
#SBATCH --partition=dev-a30           # EDIT: any CPU partition works, no GPU needed (add nothing for --gres)
#SBATCH --time=02:00:00
#SBATCH -J glam-gauss
#SBATCH -o logs/glam-gauss-%j.out
# Synthetic d=6001 Gaussian test of the r24m3 ideas (experiments/run_glam_gaussian.py).  Pure numpy.
# Usage:  sbatch cluster/slurm/glam_gaussian.sbatch                      (runs g1, stages, r24m3b, three seeds each)
#         sbatch --export=ALL,SEEDS="0",PRESETS="r24m3b" cluster/slurm/glam_gaussian.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC
source cluster/env.sh
set +u; [ "${CONDA_PREFIX:-}" = "$BRHO_CONDA_ENV" ] || { echo "ERROR: wrong env '${CONDA_PREFIX:-none}'"; exit 1; }; set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
SEEDS="${SEEDS:-0 1 2}"
PRESETS="${PRESETS:-base stages r24m3b}"
for seed in $SEEDS; do
  for pre in $PRESETS; do
    flag=""; [ "$pre" = "base" ] || flag="--preset $pre"
    python experiments/run_glam_gaussian.py --n-iter 20000 --seed "$seed" $flag --tag "${pre}_s${seed}" --out-root runs/gaussian &
  done
  wait
done
