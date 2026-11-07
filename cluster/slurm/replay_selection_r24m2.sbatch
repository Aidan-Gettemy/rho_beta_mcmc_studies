#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # numpy BLAS threads; the uncapped settings do 1000+ rounds of (pool x selected x 6000) matrix products
#SBATCH --mem=32G
#SBATCH --partition=dev-a30           # no GPU is used (no --gres); if this partition refuses a CPU-only job, override at submit time: sbatch --partition=<cpu partition> ...
#SBATCH --time=03:00:00               # EDIT: about 16 uncapped settings of a few minutes each, plus the capped ones (seconds)
#SBATCH -J glam-replay-r24m2
#SBATCH -o logs/glam-replay-r24m2-%j.out
#
# OFFLINE REPLAY OF THE GLAM ARCHIVE SELECTION on the anchor set a run saved (no solver, no MPI, no chains).
# Self-contained record: every flag is written out below.  Submit from the repo root:   sbatch cluster/slurm/replay_selection_r24m2.sbatch
# Needs <results>/anchor_state/anchor_state_ite*_global.pkl (written by the run) and its manifest.json.  Output:
#   tables/e6_glam_<tag>/replay_selection.csv and .md   (one row per setting; the first row, "baseline", is the run's own setting)
# The pool is the saved archive (r24m2: about 1200 anchors), so a replay can only THIN it: what a different regularization
# (eps), gain scale, minimum distance or cap would have kept.  Read the columns in experiments/replay_archive_selection.py.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

GLAM_TAG=r24m2                        # EDIT: the --exp-tag of the run (results/e6_glam_<tag>)

REPLAY_FLAGS=(
  --eps 1e-6,1e-2,3e-2,1e-1           # covariance_regularization of the log-det gain (run: 1e-6)
  --scale 1,10                        # logdet_gain_scale (run: 10)
  --mind 3,12                         # minimum_anchor_distance (run: 3)
  --cap none,60                       # max_anchors; none = unlimited (run: none)
  --minscore 0.5                      # minimum_diversity_score (run: 0.5)
  --pool-size 0                       # 0 = the whole saved archive; e.g. 600 for a quick first pass
  --seed 0
)

# ------------------------------------------------------------------ environment (pinned: an exported BRHO_CONDA_ENV in the submitting shell is ignored)
export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC
source cluster/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs

echo "python: $(which python)   CONDA_PREFIX=${CONDA_PREFIX:-none}"
[ "${CONDA_PREFIX:-}" = "$BRHO_CONDA_ENV" ] || { echo "ERROR: active environment is '${CONDA_PREFIX:-none}', expected $BRHO_CONDA_ENV; job stopped."; exit 1; }
[ -z "${BRHO_CONDA_FAILED:-}" ] || { echo "ERROR: the conda environment $BRHO_CONDA_ENV did not activate (see the WARNING above); job stopped."; exit 1; }
RUN="$BRHO_ROOT/results/e6_glam_$GLAM_TAG"
[ -d "$RUN/anchor_state" ] || { echo "ERROR: $RUN/anchor_state not found (check GLAM_TAG and BRHO_ROOT=$BRHO_ROOT)"; exit 1; }
ls "$RUN/anchor_state" | tail -3

python experiments/replay_archive_selection.py --exp-tag "$GLAM_TAG" "${REPLAY_FLAGS[@]}"
echo "all steps done"
