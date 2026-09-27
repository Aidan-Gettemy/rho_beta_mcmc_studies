#!/bin/bash
#SBATCH --nodes=1                     # one node
#SBATCH --ntasks-per-node=64          # one MPI rank = one chain
#SBATCH --mem=64G
#SBATCH --partition=g1mig             # your partition
#SBATCH --time=08:00:00
#SBATCH -J brho-e1
#SBATCH -o logs/brho-e1-%j.out
#
# E1: the naive (theta, beta, rho) model sampled by standard global pCN,
# 64 independent chains via mpi4py (experiments/run_e1_naive.py), then the
# post-processing (reports/report_e1.py) on one process.
#
# Submit from the repo root:      sbatch cluster/slurm/mpi_e1_naive.sbatch
# Continue after a time limit:    RESUME=1 sbatch cluster/slurm/mpi_e1_naive.sbatch
# Start over in the same study:   FRESH=1 sbatch cluster/slurm/mpi_e1_naive.sbatch
# Budget: E1_ARGS below (one PDE solve per iteration; ~1 ms at 16x16, so
# 200k iterations take a few minutes per chain).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh                 # conda env, BRHO_ROOT, PROBLEM_ARGS

# --- pin one thread per rank: avoid 64x64 thread oversubscription ---------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
# --- node-local numba cache; rank 0 compiles it before the others start --
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID

E1_ARGS="${E1_ARGS:---n-iter 200000 --n-burn 20000 --thin 200 --checkpoint-every 10000 --log-every 20000}"
MODE=""
[[ "${RESUME:-0}" == 1 ]] && MODE="--resume"
[[ "${FRESH:-0}" == 1 ]] && MODE="--fresh"
echo "BRHO_ROOT=$BRHO_ROOT  ranks=$SLURM_NTASKS  node=$(hostname)  mode=${MODE:-new}"

EXEC=experiments/run_e1_naive.py
if [[ "$MODE" == "--resume" ]]; then
    mpirun --bind-to core --map-by core -np "$SLURM_NTASKS" python3 $EXEC --resume
else
    mpirun --bind-to core --map-by core -np "$SLURM_NTASKS" \
        python3 $EXEC $MODE $PROBLEM_ARGS $E1_ARGS
fi

# --- post-processing (one process) ---------------------------------------
python3 reports/report_e1.py
echo "figures: $BRHO_ROOT/figures/e1_naive   tables: $BRHO_ROOT/tables/e1_naive"
