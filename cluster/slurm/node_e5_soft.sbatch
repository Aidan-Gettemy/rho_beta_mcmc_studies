#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64          # one rank per chain
#SBATCH --mem=64G
#SBATCH --partition=g1mig             # your partition
#SBATCH --time=08:00:00
#SBATCH -J brho-e5
#SBATCH -o logs/brho-e5-%j.out
#
# E5: beta FREE with the soft curve prior, 4-move sampler (A: pCN theta,
# B: pCN w, C: ridge, D: beta-theta swap), as ONE single-node job.
# Needs E3 run IN THE SAME STUDY (same BRHO_ROOT, same --truth-mode): K*
# comes from the truth.  Decisions (LOG.md, 2026-09-28): --truth-mode kle for
# the main study, sigma_rel unchanged, no posterior-informed proposal.
#   1. setup    basis + truth from the store, checked against the E3 curve
#   2. chains   E5_CHAINS chains (default: one per rank)
#   3. reports  report_soft (or report_soft_prior_test), report_mpsrf
# Submit from the repo root, study folder via BRHO_ROOT, e.g.
#   BRHO_ROOT=$PWD/runs/cluster_kle PRIOR_TEST=1 sbatch cluster/slurm/node_e5_soft.sbatch
#   BRHO_ROOT=$PWD/runs/cluster_kle sbatch cluster/slurm/node_e5_soft.sbatch
# Time limit: chains checkpoint every `chunk` iterations; resubmit with RESUME=1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
PT="${PRIOR_TEST:-0}"
NCH="${E5_CHAINS:-$SLURM_NTASKS}"
echo "BRHO_ROOT=$BRHO_ROOT  tasks=$SLURM_NTASKS  chains=$NCH  prior_test=$PT"

if [[ "${RESUME:-0}" != 1 ]]; then
    OMP_NUM_THREADS=$SLURM_NTASKS MKL_NUM_THREADS=$SLURM_NTASKS OPENBLAS_NUM_THREADS=$SLURM_NTASKS \
        python experiments/run_soft_sampler.py --setup-only --n-chains "$NCH" \
        --prior-test "$PT" $PROBLEM_ARGS ${E5_ARGS:---n-total 150000 --n-burn 30000 --thin 100}
else    # resuming: keep the run, only compile numba into this job's cache
    python experiments/run_soft_sampler.py --warmup-only --prior-test "$PT"
fi

srun --ntasks="$SLURM_NTASKS" --cpu-bind=cores --output="logs/brho-e5-%j-rank%t.out" \
    python experiments/run_soft_sampler.py --chain-id env --prior-test "$PT"
# mpirun --bind-to core --map-by core -np "$SLURM_NTASKS" \
#     python experiments/run_soft_sampler.py --chain-id env --prior-test "$PT"

python experiments/run_soft_sampler.py --status --prior-test "$PT" || true
if [[ "$PT" == 1 ]]; then python reports/report_soft_prior_test.py
else python reports/report_soft.py
     python reports/report_mpsrf.py        # MPSRF/PSRF, MATLAB style, all runs in this study
fi
