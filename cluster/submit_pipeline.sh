#!/bin/bash
# Submit the whole study as one dependency chain (run from the repo root):
#
#   E2 setup -> E2 pairs (array) -> E2 collect + reports ----------------+
#   E3 curve + report -> E4 setup -> E4 chains (array) -> all reports <--+
#
# Usage:  bash cluster/submit_pipeline.sh            # everything
#         bash cluster/submit_pipeline.sh e4         # only E4 (+ reports)
#         PRIOR_TEST=1 bash cluster/submit_pipeline.sh e4   # likelihood-off check
# Settings live in cluster/env.sh.  Check progress: squeue -u $USER and
#   BRHO_ROOT=... python experiments/run_curve_sampler.py --status
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh
mkdir -p logs
what="${1:-all}"
S="sbatch --parsable $SLURM_ACCOUNT_ARGS --export=ALL"
dep=""

if [[ "$what" == all || "$what" == e2 ]]; then
    n_pairs=$(python - <<'EOF'
import sys; sys.path.insert(0, ".")
from brho import config as C
g = C.GridStudyConfig(); print(len(g.betas) * len(g.rhos))
EOF
)
    j1=$($S cluster/slurm/e2_setup.sbatch)
    j2=$($S --dependency=afterok:$j1 --array=0-$((n_pairs - 1)) cluster/slurm/e2_pair.sbatch)
    j3=$($S --dependency=afterok:$j2 cluster/slurm/e2_collect.sbatch)
    echo "E2: setup $j1, pairs $j2 (0-$((n_pairs - 1))), collect $j3"
    dep="$j3"
fi

if [[ "$what" == all || "$what" == e3 || "$what" == e4 ]]; then
    if [[ "$what" != e4 ]]; then
        j4=$($S cluster/slurm/e3_curve.sbatch)
        echo "E3: curve $j4"
        e3dep="--dependency=afterok:$j4"
    else
        e3dep=""                                   # reuse the existing curve
    fi
    if [[ "$what" != e3 ]]; then
        j5=$($S $e3dep cluster/slurm/e4_setup.sbatch)
        j6=$($S --dependency=afterok:$j5 --array=0-$((E4_CHAINS - 1)) cluster/slurm/e4_chain.sbatch)
        echo "E4: setup $j5, chains $j6 (0-$((E4_CHAINS - 1)))"
        dep="${dep:+$dep:}$j6"
    fi
fi

if [[ -n "$dep" ]]; then
    j7=$($S --dependency=afterany:$dep cluster/slurm/reports.sbatch)
    echo "reports: $j7 (after $dep)"
fi
