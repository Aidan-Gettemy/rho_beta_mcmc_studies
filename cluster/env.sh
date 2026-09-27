# Sourced by every job script.  Edit once for your cluster.

# --- software environment (pick what your cluster uses) -------------------
# e.g. the numba environment from your MCMC jobs:
# module load conda/22.11.1
# source activate $HOME/.conda/envs/gd24_numba

# --- where results go (one folder per study keeps runs apart) -------------
export BRHO_ROOT="${BRHO_ROOT:-$PWD/runs/cluster}"   # e.g. /scratch/$USER/brho/grid32

# --- one thread per process: parallelism comes from array tasks ----------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# numba caches compiled kernels next to the source; point it somewhere
# writable and shared if the repo is read-only on compute nodes:
# export NUMBA_CACHE_DIR="$HOME/.cache/numba"

# --- scheduler defaults used by the templates -----------------------------
export SLURM_ACCOUNT_ARGS=""              # e.g. "--account=myproj --partition=standard"

# --- study settings (passed to the setup steps) ---------------------------
export PROBLEM_ARGS="--truth-mode standardized"
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
export E2_ARGS=""                                  # grid study overrides
export E3_ARGS=""                                  # e.g. "--n-real 512 --n-rho 40"
export E4_CHAINS=16
export E4_ARGS="--n-total 51000 --n-burn 15000 --thin 30"
