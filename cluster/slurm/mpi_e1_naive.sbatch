#!/bin/bash
#SBATCH --nodes=1                     # one node
#SBATCH --ntasks-per-node=64          # one MPI rank = one chain
#SBATCH --mem=64G
#SBATCH --partition=g1mig             # your partition
#SBATCH --time=12:00:00
#SBATCH -J brho-e1
#SBATCH -o logs/brho-e1-%j.out
#
# E1: the naive (theta, beta, rho) model sampled by pCN, 64 independent
# chains via mpi4py (experiments/run_e1_naive.py), then the post-processing
# (reports/report_e1.py) on one process.
#   MOVE=blocked (default)  A) pCN on theta, B) pCN on (xi_beta, xi_rho)
#   MOVE=global             one pCN move on everything, one step size
#   MOVE=both               blocked, then global (equal PDE-solve budgets),
#                           then the side-by-side comparison figures
#
# Submit from the repo root:      sbatch cluster/slurm/mpi_e1_naive.sbatch
#                                 MOVE=both sbatch cluster/slurm/mpi_e1_naive.sbatch
# Continue after a time limit:    RESUME=1 [MOVE=...] sbatch cluster/slurm/mpi_e1_naive.sbatch
# Start over in the same study:   FRESH=1 [MOVE=...] sbatch cluster/slurm/mpi_e1_naive.sbatch
# Budgets (E1_ARGS_BLOCKED / E1_ARGS_GLOBAL below): the blocked move costs 2
# PDE solves per iteration, the global move 1, so the defaults give both the
# same 400k solves per chain (~1 ms per solve at 16x16).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh                 # conda env, BRHO_ROOT, PROBLEM_ARGS

# --- pin one thread per rank: avoid 64x64 thread oversubscription ---------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
# --- node-local numba cache; rank 0 compiles it before the others start --
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID

E1_ARGS_BLOCKED="${E1_ARGS_BLOCKED:---n-iter 200000 --n-burn 20000 --thin 200 --checkpoint-every 10000 --log-every 20000}"
E1_ARGS_GLOBAL="${E1_ARGS_GLOBAL:---n-iter 400000 --n-burn 40000 --thin 400 --checkpoint-every 20000 --log-every 40000}"
MOVE="${MOVE:-blocked}"
MODE=""
[[ "${RESUME:-0}" == 1 ]] && MODE="--resume"
[[ "${FRESH:-0}" == 1 ]] && MODE="--fresh"
case "$MOVE" in blocked|global) MOVES="$MOVE" ;; both) MOVES="blocked global" ;;
     *) echo "MOVE must be blocked, global or both"; exit 1 ;; esac
echo "BRHO_ROOT=$BRHO_ROOT  ranks=$SLURM_NTASKS  node=$(hostname)  moves=$MOVES  mode=${MODE:-new}"

EXEC=experiments/run_e1_naive.py
for mv in $MOVES; do
    if [[ "$MODE" == "--resume" ]]; then
        ARGS="--resume"
    else
        [[ "$mv" == blocked ]] && BUDGET="$E1_ARGS_BLOCKED" || BUDGET="$E1_ARGS_GLOBAL"
        ARGS="$MODE $PROBLEM_ARGS $BUDGET"
    fi
    mpirun --bind-to core --map-by core -np "$SLURM_NTASKS" \
        python3 $EXEC --move "$mv" $ARGS
done

# --- post-processing (one process): each move, plus the comparison if both exist
python3 reports/report_e1.py --move all
echo "figures: $BRHO_ROOT/figures/e1_{blocked,global,compare}"
