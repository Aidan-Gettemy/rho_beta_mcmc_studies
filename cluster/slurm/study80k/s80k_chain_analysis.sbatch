#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h200
#SBATCH --time=08:00:00               # per arm: 24 chains x (the 60000-iteration tail, 1 GB of coefficients each) for the variance, then only the traces for the exchanges
#SBATCH -J glam-chain-analysis-s80k
#SBATCH -o logs/glam-chain-analysis-s80k-%j.out
#
# CHAIN-LEVEL ANALYSES of the six s80k arms, run where the chains are; only small tables are written (CSV / JSON in tables/e6_glam_<tag>/):
#   variance_by_index.csv, variance_summary.json   pooled variance of every coordinate over the tail (iterations 40000-100000, every 10th, 24 chains),
#                                                  split by one common partition: the 12-chain mask (installed from the anchor = constrained,
#                                                  keep own = unconstrained), the partition arms 4 and 5 were run with
#   exchange_phi.csv, leakage_summary.json         per exchange: Phi of the chains before the restart and of the restart states, the anchor Phi
#                                                  (from exchanges.pkl where it has the exchange), and restart Phi - anchor Phi (the leakage check)
# CPU only (no forward solves).  Submit from the repo root:   sbatch cluster/slurm/study80k/s80k_chain_analysis.sbatch
# One arm only:   sbatch --export=ALL,TAGS="s80k_glam1_mask" cluster/slurm/study80k/s80k_chain_analysis.sbatch
#
# exchanges.pkl: the extension from 80000 to 100000 started each run's exchange list empty and rewrote the file, so it now holds the exchanges since
# about iteration 79000.  For arms 2 and 3 the 80000-iteration file is in git (commit 9af63ef) and is added; for arms 2b, 4 and 5 the earlier
# exchanges are not available, so the anchor Phi is only known there from about iteration 79000 on (the Phi of the chains is known for every exchange).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TAGS="${TAGS:-s80k_pcn s80k_glam1 s80k_glam1_best s80k_glamN s80k_glam1_mask s80k_glamN_mask}"
MASK=runs/cluster/tables/e6_glam_s80k_pcn/keep_mask_12chains.npz     # the partition; needs the sparse checkout to contain runs/cluster/tables/e6_glam_s80k_pcn
FLAGS=(
  --mask "$MASK"
  --n-sub 12
  --chains 24
  --tail-start 40000
  --stride 10
  --n-iter 100000
)

export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC
source cluster/env.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1
mkdir -p logs
[ -s "$MASK" ] || { echo "missing $MASK: git sparse-checkout add runs/cluster/tables/e6_glam_s80k_pcn; git restore --source=origin/main --worktree -- $MASK"; exit 1; }
echo "python: $(which python)   tags: $TAGS"

OLD_DIR=$(mktemp -d)
FAILED=()
for t in $TAGS; do
  echo "=== $t   $(date +%T)"
  EXTRA=()
  case "$t" in
    s80k_glam1|s80k_glamN)               # 80000-iteration exchanges of arms 2 and 3 (committed in the first results PR)
      git show "9af63ef:runs/cluster/results/e6_${t/s80k_/glam_s80k_}/exchanges.pkl" > "$OLD_DIR/$t.pkl" 2>/dev/null \
        && EXTRA=(--old-exchanges "$OLD_DIR/$t.pkl") || echo "  no old exchanges.pkl for $t in 9af63ef" ;;
  esac
  python experiments/s80k_chain_analysis.py --exp-tag "$t" "${FLAGS[@]}" "${EXTRA[@]}" || FAILED+=("$t")
done
rm -rf "$OLD_DIR"
echo "failed arms: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
