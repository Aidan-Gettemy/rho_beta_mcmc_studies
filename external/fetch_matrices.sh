#!/usr/bin/env bash
# Copy the whitening operator and the KLE / subdomain matrix files (m500 version)
# from the cluster to this computer. Run this ON YOUR LOCAL COMPUTER.
#
#     export REMOTE_USER=... REMOTE_HOST=... REMOTE_BASE=/path/on/cluster/matrices   # no defaults on purpose
#     ./fetch_matrices.sh --list      # 1) show which remote files match (check this first)
#     ./fetch_matrices.sh --size      # 2) total size of what will be copied
#     ./fetch_matrices.sh --dry-run   # 3) rsync's own preview, copies nothing
#     ./fetch_matrices.sh             # 4) do the copy (safe to re-run; resumes)

set -euo pipefail

# ---------------- settings (environment variables) ----------------
REMOTE_USER="${REMOTE_USER:?set REMOTE_USER}"
REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST}"
LOCAL_DIR="${LOCAL_DIR:-external/matrices}"
# --------------------------------------------

REMOTE_BASE="${REMOTE_BASE:?set REMOTE_BASE (folder on the cluster that holds the matrices)}"
TAG="38x38x45_38x38x45_l2.0x2.0x2.0_m500_IND"

# Single file.
WHITENING="${WHITENING:-whitening_operator_SP1.npz}"   # check the remote name with --list; 4699 modes, matches phi_matrix_full (SP2_2mm has 5313: wrong)

# Subdomain files. On the cluster each of these is "<prefix>_<n>.mat" for the
# subdomains n = 1..12 (and possibly a "_full" file), so we match "<prefix>_*"
# instead of typing out every name. That also works if numbering starts at 0.
PREFIXES=(
  "SG_elext_${TAG}"          # extended subdomain cells
  "SG_elem_${TAG}"           # central subdomain cells
  "SG_ext_${TAG}"            # extension subdomain cells
  "KLE_matrix_${TAG}"        # KLE subdomain matrix files
  "CondMatDim_${TAG}"        # KLE subdomain matrix dimensions
  "phi_matrix_full_${TAG}"   # KLE full matrix file
  "CondMat_${TAG}"           # conditioning subdomain matrix files
  "pdom_${TAG}"              # local domain elements
  "pnode_${TAG}"             # local conditioned elements
)

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
MODE="${1:-copy}"

# Remote shell patterns (expanded on the cluster, not here).
remote_patterns="'${REMOTE_BASE}/${WHITENING}'"
for p in "${PREFIXES[@]}"; do remote_patterns+=" '${REMOTE_BASE}/${p}'_*"; done

if [[ "$MODE" == "--list" ]]; then
  ssh "$REMOTE" "ls -1d $remote_patterns 2>&1"
  echo
  echo "Counts per prefix (expect 12 subdomain files for most):"
  for p in "${PREFIXES[@]}"; do
    n=$(ssh "$REMOTE" "ls -1d '${REMOTE_BASE}/${p}'_* 2>/dev/null | wc -l")
    printf "  %-70s %s\n" "$p" "$n"
  done
  exit 0
fi

if [[ "$MODE" == "--size" ]]; then
  ssh "$REMOTE" "du -ch $remote_patterns 2>&1 | tail -n 1"
  exit 0
fi

RSYNC_OPTS=(-av --partial --progress)
[[ "$MODE" == "--dry-run" ]] && RSYNC_OPTS+=(--dry-run)

# One rsync call from the base folder: copy only the whitening file and the
# "<prefix>_*" matches (files, or folders with their contents); skip the rest.
FILTERS=(--include="${WHITENING}")
for p in "${PREFIXES[@]}"; do
  FILTERS+=(--include="${p}_*" --include="${p}_*/***")
done
FILTERS+=(--exclude="*")

mkdir -p "$LOCAL_DIR"
rsync "${RSYNC_OPTS[@]}" "${FILTERS[@]}" "${REMOTE}:${REMOTE_BASE}/" "${LOCAL_DIR}/"

echo "Done. Files are in ${LOCAL_DIR}"
