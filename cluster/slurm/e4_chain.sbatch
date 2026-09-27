#!/bin/bash
#SBATCH --job-name=brho-e4-chain
#SBATCH --partition=g1mig             # your partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
# One 3-step-sampler chain per array task (submit with --array=0-(E4_CHAINS-1);
# submit_pipeline.sh does this).  Chains checkpoint every `chunk` iterations:
# if a task hits its time limit, resubmit the same array and it resumes.
set -euo pipefail
source cluster/env.sh
python experiments/run_curve_sampler.py --chain-id env --prior-test "${PRIOR_TEST:-0}"
