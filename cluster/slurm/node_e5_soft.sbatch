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
# Multiscale theta innovation (E7, brho/multiscale.py): INNOV=ms_raw | ms_white
# (default global); each gets its own run folder, e.g.
#   BRHO_ROOT=$PWD/runs/cluster_kle PRIOR_TEST=1 INNOV=ms_raw   sbatch cluster/slurm/node_e5_soft.sbatch
#   BRHO_ROOT=$PWD/runs/cluster_kle PRIOR_TEST=1 INNOV=ms_white sbatch cluster/slurm/node_e5_soft.sbatch
#   BRHO_ROOT=$PWD/runs/cluster_kle              INNOV=ms_white sbatch cluster/slurm/node_e5_soft.sbatch
# Block (subdomain) pCN for theta (E8, brho/blockpcn.py): INNOV=block (sweep,
# 16 solves / iteration in 2D) or INNOV=block_random, e.g.
#   BRHO_ROOT=$PWD/runs/cluster_kle PRIOR_TEST=1 INNOV=block sbatch cluster/slurm/node_e5_soft.sbatch
#   BRHO_ROOT=$PWD/runs/cluster_kle INNOV=block E5_ARGS="--n-total 125000 --n-burn 10000 --thin 25" \
#       sbatch cluster/slurm/node_e5_soft.sbatch
# Variants of one innovation get their own folder with TAG (passed as --exp-tag),
# e.g. block pCN without the coarse move (E8a) / with 2-cell dual halos (E8b):
#   TAG=nocoarse INNOV=block E5_ARGS="... --block-coarse 0" sbatch ...
#   TAG=nocoarse_dualhalo INNOV=block E5_ARGS="... --block-coarse 0 --block-dual-halo 0.0625" sbatch ...
# Time limit: chains checkpoint every `chunk` iterations; resubmit with RESUME=1
# (and the same PRIOR_TEST / INNOV).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
PT="${PRIOR_TEST:-0}"
INNOV="${INNOV:-global}"
TAG="${TAG:-}"                         # run-folder suffix for variants, e.g. TAG=nocoarse
SEL="--prior-test $PT --theta-innovation $INNOV${TAG:+ --exp-tag $TAG}"
NCH="${E5_CHAINS:-$SLURM_NTASKS}"
echo "BRHO_ROOT=$BRHO_ROOT  tasks=$SLURM_NTASKS  chains=$NCH  prior_test=$PT  innovation=$INNOV  tag=$TAG"

if [[ "${RESUME:-0}" != 1 ]]; then
    OMP_NUM_THREADS=$SLURM_NTASKS MKL_NUM_THREADS=$SLURM_NTASKS OPENBLAS_NUM_THREADS=$SLURM_NTASKS \
        python experiments/run_soft_sampler.py --setup-only --n-chains "$NCH" \
        $SEL $PROBLEM_ARGS ${E5_ARGS:---n-total 150000 --n-burn 30000 --thin 100}
else    # resuming: keep the run, only compile numba into this job's cache
    python experiments/run_soft_sampler.py --warmup-only $SEL
fi

srun --ntasks="$SLURM_NTASKS" --cpu-bind=cores --output="logs/brho-e5-%j-rank%t.out" \
    python experiments/run_soft_sampler.py --chain-id env $SEL
# mpirun --bind-to core --map-by core -np "$SLURM_NTASKS" \
#     python experiments/run_soft_sampler.py --chain-id env --prior-test "$PT"

python experiments/run_soft_sampler.py --status $SEL || true
EXP=$(python -c "from brho import io; print(io.exp_soft($PT == 1, '$INNOV', '$TAG'))")
if [[ "$PT" == 1 ]]; then python reports/report_soft_prior_test.py --exp "$EXP"
else python reports/report_soft.py --exp "$EXP"
     python reports/report_mpsrf.py        # MPSRF/PSRF, MATLAB style, all runs in this study
fi
