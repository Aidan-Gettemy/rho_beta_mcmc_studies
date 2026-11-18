#!/bin/bash
# Stage only the small result files of finished dd_end runs (never raw chains, never pilots).
#
#   bash cluster/return_results.sh [--dry-run] <run dir> [<run dir> ...]
#   bash cluster/return_results.sh $(ls -d runs/msm_local_kle_2026-11/dd_end/l0.1/*/*_t202[78])
#
# Per run dir it adds summary.json and everything under post/ and post_pooled/ (figures, post_summary.json, fields.npz),
# one file at a time, and refuses chains.npz and *.chk / *_pilot paths.  Missing pieces are reported, not fatal.
# Run it on a results branch, then:  git commit -m "results: ..." && git push -u origin <branch>
set -uo pipefail
dry=0; [[ "${1:-}" == "--dry-run" ]] && { dry=1; shift; }
[[ $# -eq 0 ]] && { echo "usage: $0 [--dry-run] <run dir> ..."; exit 1; }
cd "$(git rev-parse --show-toplevel)"
added=0; skipped=0; missing=0
for d in "$@"; do
    d="${d%/}"
    if [[ "$d" == *_pilot* ]]; then echo "skip pilot: $d"; skipped=$((skipped+1)); continue; fi
    if [[ ! -d "$d" ]]; then echo "MISSING run dir: $d"; missing=$((missing+1)); continue; fi
    [[ -f "$d/post/post_summary.json" ]] || { echo "MISSING post/post_summary.json in $d"; missing=$((missing+1)); }
    while IFS= read -r f; do
        case "$f" in *chains.npz|*.chk|*_pilot*) skipped=$((skipped+1)); continue;; esac
        if (( dry )); then echo "would add $f"; else git add --sparse -f -- "$f"; fi
        added=$((added+1))
    done < <( { [[ -f "$d/summary.json" ]] && echo "$d/summary.json"; find "$d/post" "$d/post_pooled" -type f 2>/dev/null; } | sort )
done
echo "files added: $added, skipped: $skipped, missing: $missing"
if (( ! dry )); then
    n=$(git diff --cached --name-only | grep -c -E "chains\.npz|_pilot" || true)
    echo "staged chains/pilot files (must be 0): $n"
    echo "staged size:"; git diff --cached --name-only | xargs -r du -ch | tail -1
fi
