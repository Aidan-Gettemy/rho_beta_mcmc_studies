#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G                    # Juno's default is 64 GB; the shared PHI cache alone is 1.2 GB
#SBATCH --partition=normal            # Juno (UTD): 64-core nodes; change for another cluster
#SBATCH --time=04:00:00
#SBATCH -J brho-e3ext
#SBATCH -o logs/brho-e3ext-%j.out
#
# E3 with an external Darcy solver: offline beta(rho) curve for the 3D 38x38x45 problem on ONE
# node, 64 realizations at a time, one single-rank solve per core (no MPI between solves).
# The solver is not in this repository: external/solver.local says where it is (external/README.md).
#
# Submit from the repo root (settings come from cluster/env.sh, overridable on the command line):
#   sbatch cluster/slurm/node_e3_external.sbatch                          # full study
#   E3EXT_ARGS="--n-real 64 --n-rho 6 --exp-tag pilot" ALLOW_FEW_RHO=1 \
#       sbatch --time=00:30:00 cluster/slurm/node_e3_external.sbatch      # pilot: timing and spread
#   E3EXT_ARGS="--n-real 64 --n-rho 6 --exp-tag tight --rtol 1e-11" ALLOW_FEW_RHO=1 \
#       sbatch --time=00:30:00 cluster/slurm/node_e3_external.sbatch      # solver-accuracy check
# Run once, interactively on a compute node, before the first study:
#   python experiments/run_external_curve.py --selfcheck --cfg $EXT_CFG --ref-field $EXT_REF_FIELD \
#       --ref-pdrop $EXT_REF_PDROP
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
set -a; source external/solver.local; set +a          # SOLVER_DIR and, optionally, SOLVER_ENV
set +u
[ -n "${SOLVER_ENV:-}" ] && source "$SOLVER_ENV"      # e.g. module loads + PETSC_DIR for the solver
set -u
# One thread per solve and per BLAS call: the 64 cores are 64 separate processes.  Each solve is a
# single MPI rank, so keep OpenMPI off the InfiniBand/UCX path (avoids "max locked memory" errors).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader

# PETSc options for the solver: defaults from env.sh first, anything exported at submit time after them.
export PETSC_OPTIONS="$EXT_PETSC_OPTIONS ${PETSC_OPTIONS:-}"
echo "PETSC_OPTIONS=$PETSC_OPTIONS"

PHI_NPY="$EXT_MATRICES/phi_full_f32.npy"
if [ ! -f "$PHI_NPY" ]; then
  python experiments/run_external_curve.py --cache-phi "$EXT_MATRICES/phi_matrix_full_${EXT_TAG}.mat" "$PHI_NPY"
fi

python experiments/run_external_curve.py --cfg "$EXT_CFG" --matrix-dir "$EXT_MATRICES" --tag "$EXT_TAG" \
    --whitening "$EXT_WHITENING" --phi-npy "$PHI_NPY" --workers "$SLURM_CPUS_PER_TASK" \
    --Kexp "$EXT_KEXP" $E3EXT_ARGS
python experiments/check_external_fields.py --ref-field "$EXT_REF_FIELD" $(echo "$E3EXT_ARGS" | grep -o -- '--exp-tag [^ ]*')
