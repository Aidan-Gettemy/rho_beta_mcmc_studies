#!/bin/bash
# Extend the six s80k runs from 80000 to 100000 iterations (+20000).  Run from the repo root on Juno:
#       bash cluster/slurm/study80k/s80k_extend.sh            # submits all six
#       ARMS="1 3" bash cluster/slurm/study80k/s80k_extend.sh    # only some arms (numbers as in the file names: 1 2 2b 3 4 5)
# Each job is the original arm script, unchanged except N_ITER=100000 (--n-iter and --window2 both follow it) and GLAM_RESUME=1:
# every rank continues from its newest snapshot (pCN: per-chain snapshot at iteration 80000; GLAM arms: the last exchange
# snapshot, a few hundred iterations before 80000, so the last stage before 80000 is run again with the saved random stream),
# the same flags, step sizes, masks and seeds.  Chain parts after the snapshot are rewritten; earlier ones are kept.
# Checked first (tests/test_glam.py::test_a_finished_glam_run_can_be_extended...): the extended run equals one uninterrupted 100000-iteration run.
# BEFORE submitting, check that the snapshots exist:  ls results/e6_glam_<tag>/anchor_state | tail -4   (pCN: chain_r*_ite80000.pkl).
# 20000 iterations take about a quarter of the 80000 (24 h limit used there), so 8 h are asked for here.  Afterwards re-run the post jobs
# (s80k_post_*.sbatch) and s80k_mpsrf_blocks.sbatch; they read the whole chain, so the tail from 40000 now has 60000 iterations.
set -euo pipefail
cd "$(dirname "$0")/../../.."
declare -A SCRIPT=( [1]=s80k_1_pcn [2]=s80k_2_glam1 [2b]=s80k_2b_glam1_best [3]=s80k_3_glamN [4]=s80k_4_glam1_mask [5]=s80k_5_glamN_mask )
for a in ${ARMS:-1 2 2b 3 4 5}; do
  sbatch --time="${EXTEND_TIME:-08:00:00}" --export=ALL,GLAM_RESUME=1,N_ITER=100000 "cluster/slurm/study80k/${SCRIPT[$a]}.sbatch"
done
