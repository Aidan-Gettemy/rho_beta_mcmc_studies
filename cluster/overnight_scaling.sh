#!/bin/bash
# Scaling study (2026-10-01): does the block-pCN gain grow with the number of
# subdomains once r is large?  64x64, ell = 0.05 (3.2 cells per ell, as in the
# main study), r = 208.  At equal cost (SOLVES PDE solves per chain, 64 chains):
#   e5   E5          global pCN theta           3 solves / it
#   b2   E8 2x2      block pCN, 2x2 + 3x3 dual  16 solves / it
#   b3   E8 3x3      block pCN, 3x3 + 4x4 dual  28 solves / it
# Chain of jobs: E3 curve -> one setup job -> one node job per run.
# The node jobs run with RESUME=1, so a preempted / requeued / timed-out job
# continues from its checkpoints instead of re-running setup (which would
# delete the chains).  Resubmit a node job the same way to continue it.
#
#   BRHO_ROOT=$PWD/runs/cluster_l005 PART=cpu-preempt TIME=12:00:00 bash cluster/overnight_scaling.sh
#
# Options (environment):
#   BRHO_ROOT     study folder (default runs/cluster_l005); use a NEW one per truth
#   PROBLEM_ARGS  problem (default 64x64, ell 0.05, truth kle, sigma_rel 2e-3);
#                 add e.g. --truth-seed 2074 for another truth
#   RUNS          which runs (default "e5 b2 b3")
#   SOLVES        PDE solves per chain (default 2000000); e.g. 500000 for stubs
# Representative truth (2026-10-02, reports/linearized/screen.py): seed 2074
# (informed energy at the 49.5th prior percentile); the original seed 2026 is
# at the 0.3rd.  Stub truths: 2063 (9.8th), 2086 (93.4th).
set -euo pipefail
cd "$(dirname "$0")/.."
PART="${PART:-cpu-preempt}"
TIME="${TIME:-12:00:00}"
export BRHO_ROOT="${BRHO_ROOT:-$PWD/runs/cluster_l005}"
export PROBLEM_ARGS="${PROBLEM_ARGS:---n 64 --ell 0.05 --truth-mode kle --sigma-rel 2e-3}"
RUNS="${RUNS:-e5 b2 b3}"
SOLVES="${SOLVES:-2000000}"
mkdir -p logs "$BRHO_ROOT"
echo "study $BRHO_ROOT; problem $PROBLEM_ARGS; runs $RUNS; $SOLVES solves / chain; partition $PART; node time $TIME"

J3=$(sbatch --parsable --partition="$PART" --export=ALL cluster/slurm/node_e3_curve.sbatch)
echo "E3 curve: job $J3"

RUN="python experiments/run_soft_sampler.py --setup-only --n-chains 64 $PROBLEM_ARGS"
SETUP="source cluster/env.sh; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; true"
JOBS=()
for run in $RUNS; do
    case $run in
        e5) n=$(( SOLVES / 3 ));  SETUP="$SETUP && $RUN --n-total $n --n-burn 30000 --thin 100"
            JOBS+=("global:") ;;
        b2) n=$(( SOLVES / 16 )); SETUP="$SETUP && $RUN --theta-innovation block --n-total $n --n-burn 10000 --thin 25"
            SETUP="$SETUP && python experiments/run_block_prior_test.py --from-exp e5_soft_block --blocks 2"
            JOBS+=("block:") ;;
        b3) n=$(( SOLVES / 28 )); SETUP="$SETUP && $RUN --theta-innovation block --exp-tag 3x3 --block-n 3 --n-total $n --n-burn 5000 --thin 15"
            SETUP="$SETUP && python experiments/run_block_prior_test.py --from-exp e5_soft_block_3x3 --blocks 3"
            JOBS+=("block:3x3") ;;
        *) echo "unknown run $run (e5 | b2 | b3)"; exit 1 ;;
    esac
done
JS=$(sbatch --parsable --partition="$PART" --dependency=afterok:"$J3" --ntasks=1 \
     --cpus-per-task=4 --mem=24G --time=01:00:00 -J brho-setup \
     -o logs/brho-setup-%j.out --export=ALL --wrap "$SETUP")
echo "setup ($RUNS): job $JS"

for v in "${JOBS[@]}"; do
    INNOV=${v%%:*}; TAG=${v#*:}
    J=$(sbatch --parsable --partition="$PART" --time="$TIME" --dependency=afterok:"$JS" \
        --export=ALL,RESUME=1,INNOV="$INNOV",TAG="$TAG" cluster/slurm/node_e5_soft.sbatch)
    echo "chains innovation=$INNOV tag=${TAG:-none}: job $J"
done
echo "check: squeue -u \$USER; tail logs/brho-e5-*.out"
echo "after the first results commit of a study inside the repo: git sparse-checkout add ${BRHO_ROOT#$PWD/}"
