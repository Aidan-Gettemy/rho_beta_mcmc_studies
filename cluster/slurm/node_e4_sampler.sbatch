#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64          # one rank per chain
#SBATCH --mem=64G
#SBATCH --partition=g1mig             # your partition
#SBATCH --time=08:00:00
#SBATCH -J brho-e4
#SBATCH -o logs/brho-e4-%j.out
#
# E4, the 3-step curve sampler, as ONE single-node job (needs E3 finished):
#   1. setup    one process: basis + truth from the store (checked against the
#               E3 curve), frozen copies, manifest, numba compiled
#   2. chains   E4_CHAINS chains over the node's ranks; with more chains than
#               ranks each rank runs its share one after the other
#   3. reports  convergence, posteriors, fields, tables
# Submit from the repo root:   sbatch cluster/slurm/node_e4_sampler.sbatch
# Likelihood-off check first:  PRIOR_TEST=1 sbatch cluster/slurm/node_e4_sampler.sbatch
#
# Time limit: chains checkpoint every `chunk` iterations.  If the job hits the
# limit, submit it again with RESUME=1: the setup step is skipped and every
# chain continues from its checkpoint.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
PT="${PRIOR_TEST:-0}"
NCH="${E4_CHAINS:-$SLURM_NTASKS}"
echo "BRHO_ROOT=$BRHO_ROOT  tasks=$SLURM_NTASKS  chains=$NCH  prior_test=$PT"

if [[ "${RESUME:-0}" != 1 ]]; then
    OMP_NUM_THREADS=$SLURM_NTASKS MKL_NUM_THREADS=$SLURM_NTASKS OPENBLAS_NUM_THREADS=$SLURM_NTASKS \
        python experiments/run_curve_sampler.py --setup-only --n-chains "$NCH" \
        --prior-test "$PT" $PROBLEM_ARGS $E4_ARGS
else    # resuming: keep the run, only compile numba into this job's cache
    python experiments/run_curve_sampler.py --warmup-only --prior-test "$PT"
fi

srun --ntasks="$SLURM_NTASKS" --cpu-bind=cores --output="logs/brho-e4-%j-rank%t.out" \
    python experiments/run_curve_sampler.py --chain-id env --prior-test "$PT"
# mpirun --bind-to core --map-by core -np "$SLURM_NTASKS" \
#     python experiments/run_curve_sampler.py --chain-id env --prior-test "$PT"

python experiments/run_curve_sampler.py --status --prior-test "$PT" || true
if [[ "$PT" == 1 ]]; then python reports/report_prior_test.py
else python reports/report_sampler.py && python reports/report_combined.py; fi
