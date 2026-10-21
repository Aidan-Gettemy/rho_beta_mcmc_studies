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
#   GLAM_ARGS="--forward toy --exp-tag rehearsal --n-iter 2000"  sbatch --partition=normal --gres=none --time=00:30:00 cluster/slurm/node_glam.sbatch   # dress rehearsal, no solver/tracer
#   GLAM_ARGS="--exp-tag run1"                                    sbatch cluster/slurm/node_glam.sbatch                   # real run
#   GLAM_ARGS="--exp-tag run1 --resume"                           sbatch cluster/slurm/node_glam.sbatch                   # continue after a time-out
# PHI_MODE=ram uses the CPU projection (6x slower at 64 processes, see cluster/slurm/bench_projection.sbatch).
# Then post-process (strided samples; add --fields to re-run the forward model on them):
#   mpirun -np 64 python experiments/glam_postprocess.py --exp-tag run1 --stride 50 --fields
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
set -a; source external/solver.local; set +a          # SOLVER_DIR, TRACER_DIR, optional SOLVER_ENV
set +u
[ -n "${SOLVER_ENV:-}" ] && source "$SOLVER_ENV"
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
export PETSC_OPTIONS="$EXT_PETSC_OPTIONS ${PETSC_OPTIONS:-}"
mkdir -p logs "$GLAM_WORK"
[ -f "$GLAM_FITS" ] || { echo "GLAM_FITS=$GLAM_FITS not found: point it at the E3 curve study's fits.json, e.g. GLAM_FITS=../brho_runs/tables/e3_external/fits.json sbatch ..."; exit 1; }
echo "PETSC_OPTIONS=$PETSC_OPTIONS  ranks=$SLURM_NTASKS"

PHI_NPY="$EXT_MATRICES/phi_full_f32.npy"
if [ ! -f "$PHI_NPY" ]; then
  python experiments/run_external_curve.py --cache-phi "$EXT_MATRICES/phi_matrix_full_${EXT_TAG}.mat" "$PHI_NPY"
fi
export EXT_PHI_NPY="$PHI_NPY"

mpirun -np "$SLURM_NTASKS" python experiments/run_glam.py --cfg "$EXT_CFG" --matrix-dir "$EXT_MATRICES" \
    --tag "$EXT_TAG" --whitening "$EXT_WHITENING" --phi-npy "$PHI_NPY" --fits-json "$GLAM_FITS" \
    --conc "$EXT_DATA_CONC" --pdrop "$EXT_REF_PDROP" --work "$GLAM_WORK" --phi-mode "${PHI_MODE:-gpu}" ${TRACER_DIR:+--tracer-dir "$TRACER_DIR"} $GLAM_ARGS
