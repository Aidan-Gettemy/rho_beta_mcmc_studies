#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64          # one rank per chain
#SBATCH --mem=64G
#SBATCH --partition=g1mig
#SBATCH --time=08:00:00
#SBATCH -J brho-frozen-rho
#SBATCH -o logs/brho-frozen-rho-%j.out
#
# Frozen-rho baseline for the level split (reports/frozen_rho_pcn.py), on an
# existing curve-sampler run (default e4_sampler_ab):
#   1. setup    best-fit state (lowest-Phi kept sample), the same one frozen_beta_pcn.py uses
#   2. chains   64 chains, pCN on theta + pCN on log beta, rho fixed; FROZEN_SOLVES per chain
#   3. report   tables/<exp>/frozen_rho.{md,json}, figures/<exp>/fig_frozen_rho.png
#   4. frozen (rho, theta) pCN on beta with FROZEN_CHAINS chains, then fig_beta_methods.png
# Submit from the repo root with BRHO_ROOT set to the study, e.g.
#   BRHO_ROOT=$PWD/runs/cluster_kle_sig2e3 sbatch cluster/slurm/node_frozen_rho.sbatch
# Time limit hit: chains checkpoint every 50k iterations; resubmit with RESUME=1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID
EXP="${FROZEN_EXP:-e4_sampler_ab}"
SOLVES="${FROZEN_SOLVES:-1000000}"
NCH="${FROZEN_CHAINS:-$SLURM_NTASKS}"
echo "BRHO_ROOT=$BRHO_ROOT  exp=$EXP  chains=$NCH  solves/chain=$SOLVES"

if [[ "${RESUME:-0}" != 1 ]]; then
    python reports/frozen_rho_pcn.py --exp "$EXP" --setup --chains "$NCH" --solves "$SOLVES"
fi
srun --ntasks="$SLURM_NTASKS" --cpu-bind=cores --output="logs/brho-frozen-rho-%j-rank%t.out" \
    python reports/frozen_rho_pcn.py --exp "$EXP" --chain-id env
python reports/frozen_rho_pcn.py --exp "$EXP" --report
python reports/frozen_beta_pcn.py --exp "$EXP" --chains "$NCH"
python reports/plot_beta_methods.py --exp "$EXP"
