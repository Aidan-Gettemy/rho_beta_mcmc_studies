#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # one process: LSQR projection, then the solver and tracer on the fitted fields
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G                     # PHI (1.2 GB float32) on the device, a copy on the host while loading
#SBATCH --partition=dev-a30
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00               # EDIT: 3 rho values x 2 levels = 6 projections (a few minutes each on the CPU) + 7 forward solves
#SBATCH -J glam-refproj
#SBATCH -o logs/glam-refproj-%j.out
#
# MODEL-SPACE CHECK WITH THE TRUE FIELD.  Can the sampler's parametrization (12 blocks x 500 coefficients, whitened projection,
# log K = log beta(rho) + rho * YP) represent the reference field?  For each rho and each level (the offline curve's
# beta(rho), and the true beta) the reference field is projected onto the model space by least squares, the forward model
# (solver + GPU tracer) is run on the fitted field, and Phi is reported: the floor a sampler can reach.  Also prints the
# unprojected reference field's Phi, and (if GAUSS_FIELD is set) checks Kref = beta * exp(rho * G).
# Self-contained record: every flag is written out below.  Copy it per experiment, edit the lines marked EDIT, submit from the repo root:
#       sbatch cluster/slurm/glam_reference_projection.sbatch
# Results: tables/e6_glam_<tag>/reference_projection.json, results/e6_glam_<tag>/reference_projection.npz (theta*, w of every row).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------------------------ settings
CHECK_TAG=refcheck                                               # EDIT: outputs go to *_e6_glam_<tag>
export EXT_CFG=external/DATA.cfg
export EXT_MATRICES=external/matrices
export EXT_TAG=38x38x45_38x38x45_l2.0x2.0x2.0_m500_IND
export EXT_WHITENING=$EXT_MATRICES/whitening_operator_SP1.npz
export EXT_DATA_CONC=external/data/Syn_2mm_TC.dat
export EXT_REF_PDROP=external/data/Syn_2mm_pDrop.dat
export EXT_REF_FIELD=external/data/Kref.dat                       # reference permeability K [m^2], solver order
export GLAM_FITS=runs/juno_external_2026-10/tables/e3_external/fits.json
export EXT_PETSC_OPTIONS="-pc_type jacobi -ksp_max_it 100000"
GAUSS_FIELD=""                                                   # EDIT (optional): the Gaussian field file G of the reference, same ordering as Kref
PHI_MODE=gpu
N_SUB=12
SOLVER_RTOL=1e-8

CHECK_FLAGS=(
  --ref-beta 9.869232667160132e-14    # beta the reference field was generated with (0.1 darcy)
  --ref-rho 1.0                       # rho the reference field was generated with
  --rhos 0.8 1.0 1.25                 # EDIT: the rho values to project at (the sampler's best rho so far was 0.79)
  --lsqr-iter 300                     # LSQR iterations of the projection
  --damp 0.0                          # 0 = minimum-norm least squares; > 0 = Tikhonov damped fit
  --sigma-conc 0.001                  # likelihood sigmas used for the reported Phi
  --sigma-dp 0.002
  --mu-logrho 0.0                     # log rho = mu + s * w, for the saved starting states
  --s-logrho 0.75
)
# add `--no-forward` to the flags to skip the solver and tracer (projection only)

# ------------------------------------------------------------------ environment (pinned: an exported BRHO_CONDA_ENV in the submitting shell is ignored)
export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC
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
[ "${CONDA_PREFIX:-}" = "$BRHO_CONDA_ENV" ] || { echo "ERROR: active environment is '${CONDA_PREFIX:-none}', expected $BRHO_CONDA_ENV; job stopped."; exit 1; }
[ -z "${BRHO_CONDA_FAILED:-}" ] || { echo "ERROR: the conda environment $BRHO_CONDA_ENV did not activate (see the WARNING above); job stopped."; exit 1; }
python -c "import numpy, scipy, cupy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cupy', cupy.__version__)" \
  || { echo "ERROR: numpy/scipy/cupy not importable in $(which python)"; exit 1; }
python -c "import petsc4py; print('petsc4py', petsc4py.__version__)" || echo "WARNING: petsc4py not importable"
nvidia-smi --query-gpu=name,memory.total --format=csv
[ -f "$GLAM_FITS" ] || { echo "ERROR: GLAM_FITS=$GLAM_FITS not found"; exit 1; }
[ -f "$EXT_REF_FIELD" ] || { echo "ERROR: reference field $EXT_REF_FIELD not found"; exit 1; }

PHI_NPY="$EXT_MATRICES/phi_full_f32.npy"
[ -f "$PHI_NPY" ] || python experiments/run_external_curve.py --cache-phi "$EXT_MATRICES/phi_matrix_full_${EXT_TAG}.mat" "$PHI_NPY"
export EXT_PHI_NPY="$PHI_NPY"

python experiments/check_reference_projection.py --exp-tag "$CHECK_TAG" \
    --cfg "$EXT_CFG" --matrix-dir "$EXT_MATRICES" --tag "$EXT_TAG" --whitening "$EXT_WHITENING" --phi-npy "$PHI_NPY" \
    --fits-json "$GLAM_FITS" --conc "$EXT_DATA_CONC" --pdrop "$EXT_REF_PDROP" --work "$GLAM_WORK" \
    --phi-mode "$PHI_MODE" --n-sub "$N_SUB" --rtol "$SOLVER_RTOL" ${TRACER_DIR:+--tracer-dir "$TRACER_DIR"} \
    --ref-field "$EXT_REF_FIELD" ${GAUSS_FIELD:+--gauss-field "$GAUSS_FIELD"} "${CHECK_FLAGS[@]}"
