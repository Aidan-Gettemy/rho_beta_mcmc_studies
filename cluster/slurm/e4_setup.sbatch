#!/bin/bash
#SBATCH --job-name=brho-e4-setup
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
# Freezes a copy of the E3 curve, writes truth + manifest.  Needs E3 done.
# Set PRIOR_TEST=1 to set up the likelihood-off check instead.
set -euo pipefail
source cluster/env.sh
python experiments/run_curve_sampler.py --setup-only --n-chains "$E4_CHAINS" \
    --prior-test "${PRIOR_TEST:-0}" $PROBLEM_ARGS $E4_ARGS
