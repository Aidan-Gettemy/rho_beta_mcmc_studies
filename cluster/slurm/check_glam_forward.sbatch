#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2                    # 2 ranks: also tests the solver launched from inside MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -J brho-glamcheck
#SBATCH -o logs/brho-glamcheck-%j.out
#
# First real-world check of the GLAM forward model (solver + GPU tracer + likelihood), 2 ranks, 1 GPU:
#   sbatch cluster/slurm/check_glam_forward.sbatch
# Expect: the reference field reproduces the data (Phi near 0), prior draws have large Phi, no solver failures.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# the GPU tracer and CuPy live in the darcyMCMC environment (cupy 13.4.1, petsc4py 3.21.4); override with BRHO_CONDA_ENV=...
export BRHO_CONDA_ENV="${BRHO_CONDA_ENV-/home/arg210006/.conda/envs/darcyMCMC}"
source cluster/env.sh
set -a; source external/solver.local; set +a
export TRACER_DIR="${TRACER_DIR:-$SOLVER_DIR/transport}"      # transpTracer.py + rwPETSc.py live in the solver repository
set +u
[ -n "${SOLVER_ENV:-}" ] && source "$SOLVER_ENV"
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
export UCX_TLS="${UCX_TLS:-self,sm}"
export PETSC_OPTIONS="$EXT_PETSC_OPTIONS ${PETSC_OPTIONS:-}"
mkdir -p logs "$GLAM_WORK"
echo "python: $(which python)"
python -c "import mpi4py; print('mpi4py', mpi4py.__version__)"
python -c "import cupy; print('cupy', cupy.__version__)"
python -c "import petsc4py; print('petsc4py', petsc4py.__version__)" || echo "WARNING: petsc4py not importable"
nvidia-smi --query-gpu=name,memory.total --format=csv
MPILIB=$(python -c "from mpi4py import MPI; print(MPI.Get_library_version())" 2>/dev/null | head -3 | tr '\n' ' ')
if [ -n "${MPI_LAUNCH:-}" ]; then LAUNCH="$MPI_LAUNCH"
elif echo "$MPILIB" | grep -qi "mpich" && [ -x "${CONDA_PREFIX:-/none}/bin/mpiexec" ]; then LAUNCH="$CONDA_PREFIX/bin/mpiexec"
elif echo "$MPILIB" | grep -qi "mpich"; then LAUNCH="srun --mpi=pmi2"
else LAUNCH="mpirun"; fi
echo "launcher: $LAUNCH"
PHI_NPY="$EXT_MATRICES/phi_full_f32.npy"
export EXT_PHI_NPY="$PHI_NPY"
$LAUNCH -n "$SLURM_NTASKS" python experiments/check_glam_forward.py --cfg "$EXT_CFG" --matrix-dir "$EXT_MATRICES" \
    --tag "$EXT_TAG" --whitening "$EXT_WHITENING" --phi-npy "$PHI_NPY" --fits-json "$GLAM_FITS" \
    --conc "$EXT_DATA_CONC" --pdrop "$EXT_REF_PDROP" --work "$GLAM_WORK" --phi-mode "${PHI_MODE:-gpu}" \
    --ref-field "$EXT_REF_FIELD" ${TRACER_DIR:+--tracer-dir "$TRACER_DIR"}
