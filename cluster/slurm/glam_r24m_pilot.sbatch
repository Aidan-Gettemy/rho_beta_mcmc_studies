#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=24                   # EDIT: one GLAM chain per task.  24 chains = 2 chains on every one of the 12 subdomain blocks in each local stage.
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G                     # every rank holds its own float32 PHI (1.2 GB) + CuPy/solver (24 ranks)
#SBATCH --partition=h200              # EDIT: 24 ranks x about 1.5 GiB of GPU memory do not fit on one A30 (24 GiB); h200 has 141 GiB per GPU
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00               # 40000 iterations (r24 needed two jobs of up to 6 h for this); EDIT to the partition limit.  A job killed at the limit does not requeue itself: resubmit with GLAM_RESUME=1
#SBATCH -J glam-r24m-pilot
#SBATCH --requeue                     # a killed job goes back in the queue and resumes from its last snapshot
#SBATCH --open-mode=append            # keep the log of earlier attempts
#SBATCH -o logs/glam-r24m-pilot-%j.out
#
# GLAM on the PCHIP beta(rho) curve, external Darcy solver + GPU tracer.  SELF-CONTAINED RECORD OF ONE EXPERIMENT:
# every sampler flag is written out below (GLAM_FLAGS), so this file alone says how the run was made.  Copy it for a
# new experiment (e.g. cp glam_run_template.sbatch glam_r24m_pilot.sbatch), edit the lines marked EDIT, and submit from the repo root:
#       sbatch cluster/slurm/<your copy>.sbatch
# Resume after a time-out:  sbatch --export=ALL,GLAM_RESUME=1 cluster/slurm/<your copy>.sbatch      (a requeue resumes by itself)
# Post-process:             GLAM_TAG=r24m-pilot sbatch cluster/slurm/node_glam_post.sbatch
#
# What this experiment does (see GLAM_FLAGS for the numbers):  r24m PILOT (3000 iterations): same settings as glam_r24m.sbatch, short, to check the archive and restart behaviour.
#   - 24 chains; local stages visit the 12 blocks round-robin (two chains per block in every stage);
#   - open archive: no cap on the number of anchors (max_anchors -1) and no minimum distance between them; the archive keeps the
#     previous anchors and every chain's stage-best state, so it accumulates diverse states.  Its size is limited only by
#     quality_tolerance (a candidate must be within that Phi of the best anchor) -- watch anchor_count and the exchange wall time
#     with experiments/glam_exchange_summary.py;
#   - chains are shared between roles: 25 % optimisation (exploit), 25 % frontier, 50 % exploration; the restart radius grows
#     from exploit (0.01) through frontier (0.03) to exploration (0.08), and a restart is accepted if its Phi is within
#     restart_phi_tolerance (30) of the anchor, else the radius shrinks by 0.8 (up to 3 attempts);
#   - sigmas, step-size adaptation and windows as in r24 (sigma_conc 0.002, sigma_dp 0.004; window1 14000, window2 26000 in the full run, 2000 and 1000 in the pilot);
#   - every value below that differs from r24 is marked with its r24 value.  These numbers are a first guess: run the pilot
#     (glam_r24m_pilot.sbatch) first and read the exchange summary before the long run.
# NOTE exchanges never stop here (anchor_stop_iteration -1): the tail is not a clean posterior sample.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# ------------------------------------------------------------------ run identity and external inputs
GLAM_TAG=r24m-pilot                                                    # EDIT: output goes to results/e6_glam_<tag>
export EXT_CFG=external/DATA.cfg                                 # solver configuration (grid, tracer times, physics)
export EXT_MATRICES=external/matrices
export EXT_TAG=38x38x45_38x38x45_l2.0x2.0x2.0_m500_IND
export EXT_WHITENING=$EXT_MATRICES/whitening_operator_SP1.npz
export EXT_DATA_CONC=external/data/Syn_2mm_TC.dat                # concentration data (cells x 10 times)
export EXT_REF_PDROP=external/data/Syn_2mm_pDrop.dat             # pressure-drop data [psi]
export GLAM_FITS=runs/juno_external_2026-10/tables/e3_external/fits.json    # PCHIP nodes and Kexp of the offline curve
export EXT_PETSC_OPTIONS="-pc_type jacobi -ksp_max_it 100000"   # Jacobi-preconditioned CG (GAMG diverges at large rho)
PHI_MODE=gpu                                                     # projection on the GPU (ram = CPU, slower with many ranks)
N_SUB=12                                                         # subdomain blocks of 500 coefficients
SOLVER_RTOL=1e-8

# ------------------------------------------------------------------ sampler flags (every GlamConfig field)
GLAM_FLAGS=(
  # --- Length and windows  (window 1: step sizes adapt; window 2: steps frozen, local stages still run; the tail for post-processing starts at window1)
  --n-iter 3000                       # pilot: a few exchanges' worth, to read the archive and restart statistics
  --window1 2000
  --window2 1000

  # --- Stage schedule (exchanges happen at the end of every stage)
  --burnin-length 500
  --global-stage-length 200
  --local-stage-length 200
  --global-stage-decay 1.0
  --local-stage-decay 1.0
  --min-global-stage-length 50
  --min-local-stage-length 50
  --anchor-stop-iteration -1
  --num-cycles 1000000000

  # --- Step-size adaptation (pCN, two step sizes; Robbins-Monro on the logit scale, pooled over chains)
  --beta-g0 0.01
  --beta-l0 0.02
  --target 0.25
  --adapt-interval 50
  --adapt-gain 2.0
  --adapt-gain-exponent 0.5
  --adapt-gain-hold 40

  # --- Local moves: which block a chain moves in a local stage (fixed: rank % 12; round_robin: (rank + k) % 12 in the k-th stage)
  --local-block-schedule round_robin

  # --- Prior of the state and the PCHIP range: log rho = mu_logrho + s_logrho * w, rho in [rho_min, rho_max]
  --mu-logrho 0.0
  --s-logrho 0.75
  --rho-min 0.05
  --rho-max 5.0

  # --- Likelihood: Phi = sum_d (||ref_d - sim_d|| / ||ref_d||)^2 / (2 sigma_d^2)
  --sigma-conc 0.002                  # r12 used 0.001: doubled (the reference's own concentration error is 0.39%)
  --sigma-dp 0.004                    # r12 used 0.002: doubled

  # --- Exchange: which states are candidates, how many anchors, how close restarts are proposed
  --anchor-include-previous true      # r24: false.  Keep earlier anchors (the archive accumulates)
  --anchor-use-best-state true        # r24: false.  Each chain's best state of the stage is a candidate too
  --anchor-current-only false
  --max-anchors -1                    # r24: 1.  Open archive (negative = unlimited)
  --restart-scale 0.03                # r24: 0.01
  --exploit-restart-scale 0.01        # optimisation: stay very close to the anchor
  --frontier-restart-scale 0.03       # r24: 0.01
  --exploration-restart-scale 0.08    # r24: 0.01
  --restart-phi-tolerance 30.0        # r24: 5.0.  Larger so that exploration restarts survive
  --max-restart-attempts 3
  --restart-scale-decay 0.8
  --random-reach-attempts 10
  --random-reach-initial-scale -1.0
  --random-reach-decay 0.95
  --quality-tolerance 50.0            # r24: 20.  Anchors must be within this Phi of the best anchor (this is what bounds the archive)
  --minimum-anchor-distance 0.0       # r24: 5.0.  No minimum distance
  --rho-distance-weight 1.0

  # --- Output
  --save-every 200
  --log-every 100
  --seed0 2026

  # --- Anchor-manager settings (archive selection, roles, frontier, equilibrium allocation); the code defaults, written out so a later change of a default does not alter this run
  --archive-selection-mode diversity
  --covariance-regularization 1e-06
  --pressure-covariance-regularization 1e-06
  --permeability-logdet-weight 1.0
  --pressure-logdet-weight 0.0
  --normalize-logdet-by-dimension true
  --distance-score-weight 0.1         # r24: 0.02
  --quality-score-weight 0.5
  --minimum-logdet-gain -1.0
  --diversity-alpha 0.85              # r24: 0.7
  --logdet-gain-scale 1.0
  --minimum-diversity-score 0.0       # r24: 0.2.  No score cut-off
  --archive-selection-temperature 0.3 # r24: 0.1.  More stochastic selection
  --exploit-fraction 0.25             # share of chains optimising near the best anchor
  --frontier-fraction 0.25            # r24: 0.4
  --explore-fraction 0.5              # r24: 0.35
  --frontier-target-phi -1.0
  --frontier-minimum-distance -1.0
  --successful-region-distance -1.0
  --successful-region-merge-distance -1.0
  --frontier-max-rounds 3
  --frontier-minimum-improvement 0.0
  --frontier-cooldown-rounds 3
  --frontier-relative-quality-tolerance -1.0
  --frontier-reseed-interval 0
  --frontier-quality-weight 0.6
  --frontier-distance-weight 0.4
  --preserve-frontier-exact true
  --minimum-anchor-occupancy 1
  --anchor-temperature 1.0
  --uniform-allocation-weight 0.3
  --equilibrium-allocation-enabled true
  --equilibrium-positive-fraction-threshold 0.6
  --equilibrium-drift-tolerance 0.0
  --equilibrium-min-observations 8
  --equilibrium-dispersion-strength 0.75
  --equilibrium-distance-bias-max 0.85
  --preserve-exact-anchor true
  --max-exact-anchor-copies -1
  --duplicate-tolerance 1e-12
)

# the GPU tracer and CuPy live in the darcyMCMC environment (cupy 13.4.1, petsc4py 3.21.4, MPICH-built mpi4py)
export BRHO_CONDA_ENV=/home/arg210006/.conda/envs/darcyMCMC     # pinned here on purpose: an exported BRHO_CONDA_ENV in the submitting shell is ignored
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
echo "python: $(which python)   CONDA_PREFIX=${CONDA_PREFIX:-none}"
[ "${CONDA_PREFIX:-}" = "$BRHO_CONDA_ENV" ] || { echo "ERROR: active environment is '${CONDA_PREFIX:-none}', expected $BRHO_CONDA_ENV; job stopped."; exit 1; }; python -c "import numpy, scipy; print('numpy', numpy.__version__)"
# stop at once (instead of running 8-64 broken processes) when the environment is wrong
[ -z "${BRHO_CONDA_FAILED:-}" ] || { echo "ERROR: the conda environment $BRHO_CONDA_ENV did not activate (see the WARNING above); job stopped."; exit 1; }
python -c "import mpi4py; print('mpi4py', mpi4py.__version__)" || { echo "ERROR: mpi4py not importable in $(which python)"; exit 1; }
[ "${PHI_MODE:-gpu}" != gpu ] || python -c "import cupy; print('cupy', cupy.__version__)" || { echo "ERROR: cupy not importable in $(which python) (needed for --phi-mode gpu and the tracer)"; exit 1; }
[ -f "$GLAM_FITS" ] || { echo "ERROR: GLAM_FITS=$GLAM_FITS not found. Fix: git sparse-checkout add runs/juno_external_2026-10/tables/e3_external   (this clone is probably sparse), or edit the GLAM_FITS line at the top of this script to the fits.json of the offline curve study (the manifest of an earlier run lists the path it used: grep fits_json results/e6_glam_<tag>/manifest.json)"; exit 1; }
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

# after a requeue (preemption, node failure) continue from the newest complete snapshot instead of starting over
RESUME_FLAG=""
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] || [ -n "${GLAM_RESUME:-}" ]; then RESUME_FLAG="--resume"; echo "requeued (restart ${SLURM_RESTART_COUNT:-0}): resuming"; fi
$LAUNCH -n "$SLURM_NTASKS" python experiments/run_glam.py --cfg "$EXT_CFG" --matrix-dir "$EXT_MATRICES" \
    --tag "$EXT_TAG" --whitening "$EXT_WHITENING" --phi-npy "$PHI_NPY" --fits-json "$GLAM_FITS" \
    --conc "$EXT_DATA_CONC" --pdrop "$EXT_REF_PDROP" --work "$GLAM_WORK" --phi-mode "$PHI_MODE" --n-sub "$N_SUB" --rtol "$SOLVER_RTOL" ${TRACER_DIR:+--tracer-dir "$TRACER_DIR"} \
    --exp-tag "$GLAM_TAG" "${GLAM_FLAGS[@]}" ${GLAM_ARGS:-} $RESUME_FLAG
