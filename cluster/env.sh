# Sourced by every job script.  Edit once for your cluster.

# --- software environment (pick what your cluster uses) -------------------
# The job scripts run with `set -u` (unset variables are errors).  Lmod and
# conda's activate scripts read variables that may be unset (e.g. Lmod's
# "LD_PRELOAD: unbound variable"), so relax -u around them only:
set +u
# e.g. the numba environment from your MCMC jobs.  On Ganymede the conda
# module depends on python38, which needs the gnu9 toolchain loaded first
# (`module spider python38` shows the prerequisite on other clusters):
# module purge
# module load gnu9/9.4.0
# module load conda/22.11.1
# source activate $HOME/.conda/envs/gd24_numba
set -u
# record which Python the job got (should be inside the conda env):
# echo "python: $(which python)"; python -c "import numba; print('numba', numba.__version__)"

# --- where results go (one folder per study keeps runs apart) -------------
export BRHO_ROOT="${BRHO_ROOT:-$PWD/runs/cluster}"   # e.g. /scratch/$USER/brho/grid32

# --- one thread per process: parallelism comes from array tasks ----------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# numba caches compiled kernels next to the source; point it somewhere
# writable and shared if the repo is read-only on compute nodes:
# export NUMBA_CACHE_DIR="$HOME/.cache/numba"

# --- scheduling ------------------------------------------------------------
# This file runs INSIDE a job, after SLURM has scheduled it, so it cannot set
# a job's partition, time or memory.  Those come from (highest first):
#   1. the sbatch command line:   sbatch --partition=g1mig script.sbatch
#   2. SBATCH_* variables in the shell you submit from, e.g. in ~/.bashrc:
#        export SBATCH_PARTITION=g1mig
#   3. the #SBATCH lines of each script (all set --partition=g1mig).
# SLURM_ACCOUNT_ARGS is only read by cluster/submit_pipeline.sh, which calls
# sbatch itself: extra options for every job it submits.
export SLURM_ACCOUNT_ARGS=""              # e.g. "--account=myproj"

# --- study settings (passed to the setup steps) ---------------------------
# Each line only sets a DEFAULT: a value given when you submit wins, e.g.
#   E2_ARGS="--n-iter 20000" sbatch cluster/slurm/node_e2_grid.sbatch
# (sbatch passes your shell's variables into the job; "${X:-default}" keeps them).
export PROBLEM_ARGS="${PROBLEM_ARGS:---truth-mode standardized}"
# likelihood (brho/problem.py): relative by default, sigma_rel = rel_eps/sqrt(rel_n_eff)
#   = 0.02/sqrt(128) = 1.77e-3; the same form as MCMCpar.py (sigma = s, Phi =
#   relerr^2/(2 s^2)), so "--sigma-rel s" matches it.  Set it directly, or
#   via the pair "--rel-eps 0.02 --rel-n-eff 128"; "--likelihood absolute" for
#   the July form.
# KLE (brho/kle.py): default "auto" = dense discrete KLE up to 4096 cells,
# Gauss-Legendre Nystrom (separable path) above.  For a Nystrom study at any
# size, e.g. 100x100 with ell = 0.05:
#   PROBLEM_ARGS="--n 100 --ell 0.05 --kle-method nystrom_sep --kle-nq 60"
# kle-nq ~ 3/ell converges the eigenvalues to ~1e-10 for sq-exp; the setup
# log and manifest report the check (nq_check).
export E2_ARGS="${E2_ARGS:---n-iter 50000 --n-burn 10000}"   # per-pair pCN budget
export E3_ARGS="${E3_ARGS:-}"                      # e.g. "--n-real 512 --n-rho 40"
# E4 chains: array jobs (submit_pipeline.sh, e4_setup) default to 16; the
# single-node launcher node_e4_sampler.sbatch defaults to one per rank.
export E4_CHAINS="${E4_CHAINS:-}"
export E4_ARGS="${E4_ARGS:---n-total 51000 --n-burn 15000 --thin 30}"
