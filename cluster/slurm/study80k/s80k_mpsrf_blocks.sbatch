#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=h200
#SBATCH --time=02:00:00               # per arm: thinned samples are loaded once (24 chains x 4000 x 6001, float32 = 2.3 GB) and each 500-dimensional block curve took about 3 s in the post jobs
#SBATCH -J glam-mpsrf-blocks-s80k
#SBATCH -o logs/glam-mpsrf-blocks-s80k-%j.out
#
# MPSRF OF EVERY BLOCK, for all six arms of study s80k.  The post jobs computed MPSRF for rho, the first 100 coordinates and block 0 only.
# This adds block:1 .. block:11, with exactly the same samples and window (stride 10, from iteration 40000, burn 10, step 10).
# glam_mpsrf.py rewrites tables/e6_glam_<tag>/mpsrf.json and mpsrf_curves.csv, so the three original specs are listed first: nothing is lost.
# CPU only (no forward solves).  Submit from the repo root:   sbatch cluster/slurm/study80k/s80k_mpsrf_blocks.sbatch
# One arm only:   sbatch --export=ALL,TAGS="s80k_pcn" cluster/slurm/study80k/s80k_mpsrf_blocks.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TAGS="${TAGS:-s80k_pcn s80k_glam1 s80k_glam1_best s80k_glamN s80k_glam1_mask s80k_glamN_mask}"
DIMS=(rho first:100 block:0 block:1 block:2 block:3 block:4 block:5 block:6 block:7 block:8 block:9 block:10 block:11)
FLAGS=(
  --stride 10
  --start-iteration 40000
  --skip 1
  --step 10
  --burn-samples 10
  --level 1.2
  --ignore-first 10
  --max-minutes 30
)

export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC
source cluster/env.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1
mkdir -p logs
echo "python: $(which python)   tags: $TAGS"

FAILED=()
for t in $TAGS; do
  echo "=== $(date +%T)  $t"
  python experiments/glam_mpsrf.py --exp-tag "$t" --dims "${DIMS[@]}" "${FLAGS[@]}" || FAILED+=("$t")
done
echo "=== $(date +%T)  done"
if [ "${#FAILED[@]}" -gt 0 ]; then echo "FAILED: ${FAILED[*]}"; exit 1; fi
