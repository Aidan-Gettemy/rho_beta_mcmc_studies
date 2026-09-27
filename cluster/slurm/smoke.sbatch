#!/bin/bash
#SBATCH --job-name=brho-smoke
#SBATCH --partition=g1mig             # your partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# Run first on a new cluster/environment: tests, solver benchmark, and the
# whole pipeline with tiny budgets (into runs/quick, not BRHO_ROOT).
set -euo pipefail
source cluster/env.sh
python tests/run_tests.py
python benchmarks/bench_solver.py --budget 0.5
make quick
