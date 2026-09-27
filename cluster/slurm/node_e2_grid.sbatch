#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=25          # one rank per (beta, rho) pair (5x5 grid)
#SBATCH --mem=32G
#SBATCH --partition=g1mig             # your partition
#SBATCH --time=02:00:00
#SBATCH -J brho-e2
#SBATCH -o logs/brho-e2-%j.out
#
# E2, the 5x5 grid study, as ONE single-node job:
#   1. setup    one process: KLE basis + reference field (from the study
#               store, or built once), manifest, numba compiled into the cache
#   2. pairs    one rank per pair, all in parallel (srun, or mpirun below)
#   3. collect  merge the 25 pair files, draw the E2 figures and tables
# Submit from the repo root:   sbatch cluster/slurm/node_e2_grid.sbatch
# Results: $BRHO_ROOT/{results,figures,tables}/e2_grid  (BRHO_ROOT: cluster/env.sh)
# If the partition only hands out whole nodes, ask for all 64 tasks: ranks
# 25..63 then find no pair and exit immediately.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID    # node-local, per job
echo "BRHO_ROOT=$BRHO_ROOT  tasks=$SLURM_NTASKS  node=$(hostname)"

# 1. setup: one process may use every core for the KLE build (BLAS threads)
OMP_NUM_THREADS=$SLURM_NTASKS MKL_NUM_THREADS=$SLURM_NTASKS OPENBLAS_NUM_THREADS=$SLURM_NTASKS \
    python experiments/run_grid_study.py --setup-only $PROBLEM_ARGS $E2_ARGS

# 2. pairs: rank r runs pair r (one thread each; env.sh pins threads to 1)
srun --ntasks="$SLURM_NTASKS" --cpu-bind=cores --output="logs/brho-e2-%j-rank%t.out" \
    python experiments/run_grid_study.py --pair-index env
# (equivalent with your usual launcher:)
# mpirun --bind-to core --map-by core -np "$SLURM_NTASKS" \
#     python experiments/run_grid_study.py --pair-index env

# 3. collect + reports
python experiments/run_grid_study.py --collect
python reports/report_setup.py
python reports/report_grid.py
echo "done: figures in $BRHO_ROOT/figures/e2_grid"
