#!/bin/bash
# Scaling study (2026-10-01): does the block-pCN gain grow with the number of
# subdomains once r is large?  64x64, ell = 0.05 (3.2 cells per ell, as in the
# main study), r = 208.  At equal cost (2.0M PDE solves per chain, 64 chains):
#   E5          global pCN theta           3 solves / it   667,000 it
#   E8 2x2      block pCN, 2x2 + 3x3 dual  16 solves / it  125,000 it
#   E8 3x3      block pCN, 3x3 + 4x4 dual  28 solves / it   71,500 it
# Chain of jobs: E3 curve -> one setup job (all three runs) -> three node jobs.
# The node jobs run with RESUME=1, so a preempted / requeued / timed-out job
# continues from its checkpoints instead of re-running setup (which would
# delete the chains).  Resubmit a node job the same way to continue it.
#
#   PART=cpu-preempt TIME=12:00:00 bash cluster/overnight_scaling.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PART="${PART:-cpu-preempt}"
TIME="${TIME:-12:00:00}"
export BRHO_ROOT="${BRHO_ROOT:-$PWD/runs/cluster_l005}"
export PROBLEM_ARGS="--n 64 --ell 0.05 --truth-mode kle --sigma-rel 2e-3"
mkdir -p logs "$BRHO_ROOT"
echo "study $BRHO_ROOT; problem $PROBLEM_ARGS; partition $PART; node time $TIME"

J3=$(sbatch --parsable --partition="$PART" --export=ALL cluster/slurm/node_e3_curve.sbatch)
echo "E3 curve: job $J3"

RUN="python experiments/run_soft_sampler.py --setup-only --n-chains 64 $PROBLEM_ARGS"
SETUP="source cluster/env.sh; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; \
$RUN --n-total 667000 --n-burn 30000 --thin 100 && \
$RUN --theta-innovation block --n-total 125000 --n-burn 10000 --thin 25 && \
$RUN --theta-innovation block --exp-tag 3x3 --block-n 3 --n-total 71500 --n-burn 5000 --thin 15 && \
python experiments/run_block_prior_test.py --from-exp e5_soft_block --blocks 2 && \
python experiments/run_block_prior_test.py --from-exp e5_soft_block_3x3 --blocks 3"
JS=$(sbatch --parsable --partition="$PART" --dependency=afterok:"$J3" --ntasks=1 \
     --cpus-per-task=4 --mem=24G --time=01:00:00 -J brho-setup \
     -o logs/brho-setup-%j.out --export=ALL --wrap "$SETUP")
echo "setup (E5, E8 2x2, E8 3x3; block prior tests): job $JS"

for v in "global:" "block:" "block:3x3"; do
    INNOV=${v%%:*}; TAG=${v#*:}
    J=$(sbatch --parsable --partition="$PART" --time="$TIME" --dependency=afterok:"$JS" \
        --export=ALL,RESUME=1,INNOV="$INNOV",TAG="$TAG" cluster/slurm/node_e5_soft.sbatch)
    echo "chains innovation=$INNOV tag=${TAG:-none}: job $J"
done
echo "check: squeue -u \$USER; tail logs/brho-e5-*.out"
