#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64                   # one GLAM chain per CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G                    # every rank holds its own float32 PHI (1.2 GB) -> ~77 GB for 64 ranks (+ CuPy/solver)
#SBATCH --partition=h200              # 64 cores, 384 GB, 2 x H200; h100 (4 x H100, 512 GB) or a30 (128 cores, 8 x A30) also possible
#SBATCH --gres=gpu:2                  # --gres=gpu:<n> is required for GPU access; the 64 ranks share these GPUs round-robin
#SBATCH --time=24:00:00                   # partition limit 2 days
#SBATCH -J brho-glam
#SBATCH -o logs/brho-glam-%j.out
#
# GLAM (global-local anchored multistage sampling) on top of the offline PCHIP beta(rho) curve.
# Submit from the repo root; settings come from cluster/env.sh, GLAM options via GLAM_ARGS:
#   GLAM_ARGS="--forward toy --exp-tag rehearsal --n-iter 2000 --window1 600 --window2 600 --burnin-length 100 --global-stage-length 100 --local-stage-length 100" PHI_MODE=ram  sbatch --partition=normal --gres=none --time=00:30:00 cluster/slurm/node_glam.sbatch   # dress rehearsal, no solver/tracer
#   Small test that starts quickly (8 chains on one A30, shared partition):
#   GLAM_ARGS="--exp-tag small --n-iter 400 --window1 150 --window2 150 --burnin-length 50 --global-stage-length 50 --local-stage-length 50" \
#     sbatch --partition=dev-a30 --gres=gpu:1 --ntasks=8 --mem=32G --time=01:00:00 cluster/slurm/node_glam.sbatch
#   GLAM_ARGS="--exp-tag run1"                                    sbatch cluster/slurm/node_glam.sbatch                   # real run
#   GLAM_ARGS="--exp-tag run1 --resume"                           sbatch cluster/slurm/node_glam.sbatch                   # continue after a time-out
# PHI_MODE=ram uses the CPU projection (6x slower at 64 processes, see cluster/slurm/bench_projection.sbatch).
# Then post-process (strided samples; add --fields to re-run the forward model on them):
#   mpirun -np 64 python experiments/glam_postprocess.py --exp-tag run1 --stride 50 --fields
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# the GPU tracer and CuPy live in the darcyMCMC environment (cupy 13.4.1, petsc4py 3.21.4); override with BRHO_CONDA_ENV=...
export BRHO_CONDA_ENV="${BRHO_CONDA_ENV-/home/arg210006/.conda/envs/darcyMCMC}"
source cluster/env.sh
set -a; source external/solver.local; set +a          # SOLVER_DIR, TRACER_DIR, optional SOLVER_ENV
export TRACER_DIR="${TRACER_DIR:-$SOLVER_DIR/transport}"      # transpTracer.py + rwPETSc.py live in the solver repository
set +u
[ -n "${SOLVER_ENV:-}" ] && source "$SOLVER_ENV"
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
export UCX_TLS="${UCX_TLS:-self,sm}"      # MPICH/UCX on one node: shared memory only (also silences "cuDeviceGetCount failed" on CPU nodes)
export PETSC_OPTIONS="$EXT_PETSC_OPTIONS ${PETSC_OPTIONS:-}"
mkdir -p logs "$GLAM_WORK"
echo "python: $(which python)"; python -c "import numpy, scipy; print('numpy', numpy.__version__)"
python -c "import mpi4py; print('mpi4py', mpi4py.__version__)" || echo "WARNING: mpi4py not importable"
[ "${PHI_MODE:-gpu}" = gpu ] && { python -c "import cupy; print('cupy', cupy.__version__)" || echo "WARNING: cupy not importable (needed for --phi-mode gpu and the tracer)"; }
[ -f "$GLAM_FITS" ] || { echo "GLAM_FITS=$GLAM_FITS not found: point it at the E3 curve study's fits.json, e.g. GLAM_FITS=../brho_runs/tables/e3_external/fits.json sbatch ..."; exit 1; }
echo "PETSC_OPTIONS=$PETSC_OPTIONS  ranks=$SLURM_NTASKS"

PHI_NPY="$EXT_MATRICES/phi_full_f32.npy"
if [ ! -f "$PHI_NPY" ]; then
  python experiments/run_external_curve.py --cache-phi "$EXT_MATRICES/phi_matrix_full_${EXT_TAG}.mat" "$PHI_NPY"
fi
export EXT_PHI_NPY="$PHI_NPY"

# Launcher: it must come from the same MPI library that mpi4py was built against (a mismatch gives
# "Runtime environment uses unsupported PMI version PMIx" for MPICH-built mpi4py under OpenMPI's mpirun).
MPILIB=$(python -c "from mpi4py import MPI; print(MPI.Get_library_version())" | head -3 | tr '\n' ' ')
echo "mpi4py library: $MPILIB"
if [ -n "${MPI_LAUNCH:-}" ]; then LAUNCH="$MPI_LAUNCH"                          # explicit override, e.g. MPI_LAUNCH="srun --mpi=pmi2"
elif echo "$MPILIB" | grep -qi "mpich" && [ -x "${CONDA_PREFIX:-/none}/bin/mpiexec" ]; then LAUNCH="$CONDA_PREFIX/bin/mpiexec"
elif echo "$MPILIB" | grep -qi "mpich"; then LAUNCH="srun --mpi=pmi2"
else LAUNCH="mpirun"; fi
echo "launcher: $LAUNCH   (mpirun is $(which mpirun 2>/dev/null || echo none))"
# preflight: mpi4py must see the world the launcher creates (if not, every process would think it is rank 0)
$LAUNCH -n 4 python -c "from mpi4py import MPI; c = MPI.COMM_WORLD; assert c.Get_size() == 4, c.Get_size(); print('mpi4py ok, rank', c.Get_rank(), 'of', c.Get_size())"

# GPU memory over time (every 30 s) next to the job log: tells how many ranks fit on a GPU
( while true; do date +%T; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null; sleep 30; done > "logs/brho-glam-${SLURM_JOB_ID:-local}.gpumem" ) &
GPUMON=$!
trap 'kill $GPUMON 2>/dev/null || true' EXIT

$LAUNCH -n "$SLURM_NTASKS" python experiments/run_glam.py --cfg "$EXT_CFG" --matrix-dir "$EXT_MATRICES" \
    --tag "$EXT_TAG" --whitening "$EXT_WHITENING" --phi-npy "$PHI_NPY" --fits-json "$GLAM_FITS" \
    --conc "$EXT_DATA_CONC" --pdrop "$EXT_REF_PDROP" --work "$GLAM_WORK" --phi-mode "${PHI_MODE:-gpu}" ${TRACER_DIR:+--tracer-dir "$TRACER_DIR"} $GLAM_ARGS
