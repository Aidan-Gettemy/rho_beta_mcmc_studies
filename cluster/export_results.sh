#!/bin/bash
# Copy the shareable part of a study into the repo, leaving raw chains out.
#
#   bash cluster/export_results.sh <BRHO_ROOT of the study> <name>
#   e.g. bash cluster/export_results.sh /scratch/$USER/brho/grid16 grid16_2026-10
#
# Copies figures/, tables/, SUMMARY.md, every manifest.json, the frozen
# kle.npz / truth.npz and the small collected results (grid.npz, curve_*.npz)
# into runs/<name>/.  Chain checkpoints and per-pair files stay where they are.
# Then commit on a branch:
#   git checkout -b results-<name>
#   git add --sparse runs/<name>        # --sparse only needed in a sparse clone
#   git commit -m "results: <name>" && git push -u origin results-<name>
set -euo pipefail
src="${1:?study root (its BRHO_ROOT)}"; name="${2:?name for runs/<name>}"
if [[ "$name" == results-* || "$name" == */* ]]; then
    echo "WARNING: '$name' looks like a branch name or a path; the second argument is"
    echo "  the study folder under runs/ (e.g. cluster_kle), not the git branch."
    echo "  Continuing in 5 s -- Ctrl-C to stop."; sleep 5
fi
cd "$(dirname "$0")/.."
dst="runs/$name"
mkdir -p "$dst"
for d in figures tables; do [[ -d "$src/$d" ]] && cp -r "$src/$d" "$dst/"; done
[[ -f "$src/SUMMARY.md" ]] && cp "$src/SUMMARY.md" "$dst/"
( cd "$src" && find results \( -name manifest.json -o -name kle.npz -o -name truth.npz \
      -o -name grid.npz -o -name 'curve*.npz' \) -print0 ) |
  while IFS= read -r -d '' f; do mkdir -p "$dst/$(dirname "$f")"; cp "$src/$f" "$dst/$f"; done
du -sh "$dst"
echo "exported to $dst (raw chains and pair files left in $src)"
