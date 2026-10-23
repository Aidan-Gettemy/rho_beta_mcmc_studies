#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8                    # <= the ranks of the run (about 1.5 GiB of GPU memory each on an A30)
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=dev-a30
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -J brho-glam-post
#SBATCH -o logs/brho-glam-post-%j.out
#
# Post-processing of a GLAM run: diagnostics (RE traces, acceptance histograms), strided samples + forward model on
# them (best field, pointwise statistics, per-z-layer concentration), field figures, and the MPSRF cost estimate.
# Submit from the repo root; the run is selected by its --exp-tag:
#   GLAM_TAG=med sbatch cluster/slurm/node_glam_post.sbatch
# Options (environment variables, all optional):
#   STRIDE=50            iterations between the saved samples (the forward model runs once per sample)
#   POST_ARGS="..."      extra arguments of experiments/glam_postprocess.py     (e.g. "--best-scope tail")
#   FIELD_ARGS="..."     extra arguments of experiments/plot_glam_fields.py     (e.g. "--conc-layers 5 15 25 35 --nsd 2")
#   MPSRF_ARGS="..."     arguments of experiments/glam_mpsrf.py (default: --estimate only).  A real MPSRF run is
#                        expensive: read the estimate first, then e.g.  MPSRF_ARGS="--psrf-only"  or  "--dims rho first:100"
#   BRHO_ROOT=...        study folder holding results/e6_glam_<tag> (default runs/cluster, see cluster/env.sh)
#   BRHO_CONDA_ENV=...   conda environment (default darcyMCMC)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${GLAM_TAG:?set GLAM_TAG to the --exp-tag of the run, e.g. GLAM_TAG=med sbatch cluster/slurm/node_glam_post.sbatch}"
export BRHO_CONDA_ENV="${BRHO_CONDA_ENV-/home/arg210006/.conda/envs/darcyMCMC}"
source cluster/env.sh
set -a; source external/solver.local; set +a
export TRACER_DIR="${TRACER_DIR:-$SOLVER_DIR/transport}"
set +u
[ -n "${SOLVER_ENV:-}" ] && source "$SOLVER_ENV"
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
export UCX_TLS="${UCX_TLS:-self,sm}"
export PETSC_OPTIONS="$EXT_PETSC_OPTIONS ${PETSC_OPTIONS:-}"
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1
mkdir -p logs "$GLAM_WORK"

echo "python: $(which python)   CONDA_PREFIX=${CONDA_PREFIX:-none}"
[ -z "${BRHO_CONDA_FAILED:-}" ] || { echo "ERROR: the conda environment $BRHO_CONDA_ENV did not activate (see the WARNING above); job stopped."; exit 1; }
python -c "import numpy, mpi4py, cupy; print('numpy', numpy.__version__, 'mpi4py', mpi4py.__version__, 'cupy', cupy.__version__)" \
  || { echo "ERROR: numpy/mpi4py/cupy not importable in $(which python)"; exit 1; }
RUN="$BRHO_ROOT/results/e6_glam_$GLAM_TAG"
[ -d "$RUN" ] || { echo "ERROR: $RUN not found (check GLAM_TAG and BRHO_ROOT=$BRHO_ROOT)"; exit 1; }
echo "run: $RUN   chain files: $(ls "$RUN" | wc -l)   ranks: $SLURM_NTASKS"

# launcher from the MPI library that mpi4py was built against (see node_glam.sbatch)
MPILIB=$(python -c "from mpi4py import MPI; print(MPI.Get_library_version())" | head -3 | tr '\n' ' ')
if [ -n "${MPI_LAUNCH:-}" ]; then LAUNCH="$MPI_LAUNCH"
elif echo "$MPILIB" | grep -qi "mpich" && [ -x "${CONDA_PREFIX:-/none}/bin/mpiexec" ]; then LAUNCH="$CONDA_PREFIX/bin/mpiexec"
elif echo "$MPILIB" | grep -qi "mpich"; then LAUNCH="srun --mpi=pmi2"
else LAUNCH="mpirun"; fi
echo "launcher: $LAUNCH"

STRIDE="${STRIDE:-50}"
step() { echo; echo "=== $(date +%T)  $* ==="; }

step "1/4 diagnostics: RE traces and acceptance histograms (no solver)"
python experiments/plot_glam_diagnostics.py --exp-tag "$GLAM_TAG"

step "2/4 samples (stride $STRIDE) and forward model on them: best field, statistics, per-z concentration"
$LAUNCH -n "$SLURM_NTASKS" python experiments/glam_postprocess.py --exp-tag "$GLAM_TAG" --stride "$STRIDE" \
    --fields --forward external ${POST_ARGS:-}

step "3/4 field figures"
python experiments/plot_glam_fields.py --exp-tag "$GLAM_TAG" ${FIELD_ARGS:-}

step "4/4 MPSRF (${MPSRF_ARGS:---estimate})"
python experiments/glam_mpsrf.py --exp-tag "$GLAM_TAG" ${MPSRF_ARGS:---estimate}

echo; echo "done: figures in $BRHO_ROOT/figures/e6_glam_$GLAM_TAG, tables/arrays under $RUN/post"
