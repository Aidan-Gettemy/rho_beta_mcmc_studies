#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=12                   # r24m ran 24 chains; 12 ranks are enough here (the strided samples are dealt round-robin) and fit on one A30.  Only the forward-model step uses them (MPI); about 1.5 GiB of GPU memory each
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --partition=dev-a30
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00               # EDIT: the forward-model step runs the solver on every strided sample (24 chains x 40000/100 = 9600 solves on 12 ranks); MPSRF is capped by MPSRF max-minutes
#SBATCH -J glam-post-s80k_glamN_mask
#SBATCH -o logs/glam-post-s80k_glamN_mask-%j.out
#
# POST-PROCESSING OF ONE GLAM RUN.  Self-contained record: every flag of every step is written out below, so this file
# alone says how the figures and tables were made.  Copy it per experiment (e.g. cp glam_post_template.sbatch glam_post_s80k_glamN_mask.sbatch),
# edit the lines marked EDIT, and submit from the repo root:   sbatch cluster/slurm/<your copy>.sbatch
#
# Steps (switch each on/off in the settings block); figures go to figures/e6_glam_<tag>/, tables to tables/e6_glam_<tag>/:
#   1 DIAGNOSTICS   plot_glam_diagnostics.py  RE traces (conc, pressure drop), RE histogram, acceptance traces and histograms
#                   (the RE trace needs a run made with the per-iteration RE; older runs use the strided samples of step 3)
#   2 TRACES        plot_glam_traces.py       Phi history, rho, and selected theta entries versus iteration, all chains
#   3 SAMPLES       glam_postprocess.py       strided samples, best state, and (with --fields) the forward model on the best state
#                   and on every strided sample: best fields, pointwise posterior mean/sd of pressure, concentration, log K,
#                   per-z-disc concentration, per-sample RE.  The only step that calls the solver and the tracer (MPI, GPU).
#   4 FIELD_FIGURES plot_glam_fields.py       log K slices, tracer maps, concentration versus time (needs step 3 with --fields)
#   4b EXPORT       export_glam_fields.py     best sample, reference and posterior mean/sd/z-score as .npy (nx,ny,nz) and one VTK file
#                   for ParaView: results/e6_glam_<tag>/post/export/ (needs step 3 with --fields)
#   5 MPSRF         glam_mpsrf.py             PSRF of every coordinate (cheap) and MPSRF curves of the --dims you choose
#                   (expensive: the cost estimate is printed first; a spec above --max-minutes is skipped).  Saves the raw
#                   iteration-vs-MPSRF table tables/e6_glam_<tag>/mpsrf_curves.csv (+ psrf_max.csv) that can be copied anywhere and
#                   replotted:  python experiments/replot_glam_mpsrf.py --curves mpsrf_curves.csv --psrf psrf_max.csv
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------------------------ which run, which steps
GLAM_TAG=s80k_glamN_mask                  # study s80k, arm: glamN  (results/e6_glam_<tag>)
RUN_DIAGNOSTICS=1                     # 1 = run the step, 0 = skip it
RUN_TRACES=1
RUN_SAMPLES=1
RUN_FIELD_FIGURES=1
RUN_EXPORT=1
RUN_MPSRF=1
MPSRF_MODE=run                        # run = estimate, then compute | estimate = cost estimate only | psrf-only = cheap per-coordinate PSRF only

# ------------------------------------------------------------------ external inputs (same as the run)
export EXT_CFG=external/DATA.cfg
export EXT_MATRICES=external/matrices
export EXT_TAG=38x38x45_38x38x45_l2.0x2.0x2.0_m500_IND
export EXT_WHITENING=$EXT_MATRICES/whitening_operator_SP1.npz
export EXT_DATA_CONC=external/data/Syn_2mm_TC.dat
export EXT_REF_PDROP=external/data/Syn_2mm_pDrop.dat
export EXT_REF_FIELD=external/data/Kref.dat                       # reference permeability K [m^2]
export GLAM_FITS=runs/juno_external_2026-10/tables/e3_external/fits.json   # only used by the reference-rho line of check_glam_forward; the run's own path is in its manifest
export EXT_PETSC_OPTIONS="-pc_type jacobi -ksp_max_it 100000"

# ------------------------------------------------------------------ flags of every step
# 1 plot_glam_diagnostics.py   (not set here: --chains = all chains, --tail-start = window1 from the manifest, --linear = linear RE axis)
DIAG_FLAGS=(
  --tail-start 40000                  # fixed-step study: no adaptation window, so the tail is set by hand (second half of the 80000 iterations)
  --ar-window 100                     # iterations per acceptance window (a window counts for a kind with >= 10 moves of it)
  --stride 0                          # 0 = every iteration of the RE traces; k = every k-th
)
# 2 plot_glam_traces.py        entries are block:index; blocks 0-11 are the subdomain thetas, block 12 is the rho coordinate
TRACE_FLAGS=(
  --entries 0:0 3:0 6:0 11:0 12:0
  --chains all
  --max-chains 24                     # 24 chains
  --stride 0                          # 0 = about 2000 points
)
# 3 glam_postprocess.py        (not set here: --tail-start = window1, --float64 = keep the samples in double precision)
POST_FLAGS=(
  --tail-start 40000                  # samples come from iterations 40000-80000 (window1 is 0 in this study)
  --stride 100                         # iterations between the saved samples; the solver runs once per sample
  --best-scope all                    # all = best state of the whole run | tail = best state after window1
  --fields                            # re-run the forward model on the best state and on every sample (remove to skip the solver)
  --forward external
)
# 4 plot_glam_fields.py        (not set here: --ref-is-log if the reference file holds log K)
FIELD_FLAGS=(
  --cfg "$EXT_CFG"
  --ref-field "$EXT_REF_FIELD"
  --ref-conc "$EXT_DATA_CONC"
  --slice-fractions 0.025 0.25 0.5 0.75 0.975
  --layers 1 5 10 15 20 25 30 35 40 45          # z-layers (1-based) of the multi-layer log K figures
  --tracer-layer 23                             # z-layer of the tracer maps
  --conc-layers 1 11 23 34 45                   # z-layers of the concentration-versus-time figure
  --nsd 2.0                                     # band = mean +- nsd standard deviations
)
# 4b export_glam_fields.py
EXPORT_FLAGS=(
  --cfg "$EXT_CFG"
  --ref-field "$EXT_REF_FIELD"
  --ref-conc "$EXT_DATA_CONC"
  --units mm                                    # length unit of the VTK spacing (mm | m)
)
# 5 glam_mpsrf.py
#   --dims (several allowed, one MPSRF curve each; this is where you choose the number of dimensions):
#     rho  |  first:K  |  blocks:K (first K entries of every block)  |  block:B[:K]  |  random:K[:SEED]  |  all
#     all or a large K needs  chains * (window - 1) > dimension for W to be invertible (points before that are NaN)
MPSRF_DIMS=(rho first:100 block:0)    # EDIT
MPSRF_FLAGS=(
  --stride 10                         # iterations between thinned samples
  --start-iteration 40000             # MPSRF window starts at iteration 40000 (same in every arm; no adaptation to skip)
  --skip 1                            # use every skip-th thinned sample inside a window
  --step 10                           # thinned samples between curve points
  --burn-samples 10                   # first thinned sample of every window
  --level 1.2                         # convergence level for the read-out
  --ignore-first 10                   # curve points that cannot count as convergence
  --max-minutes 30                    # a spec whose estimated cost is above this is skipped
)

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
python -c "import numpy, mpi4py, cupy; print('numpy', numpy.__version__, 'mpi4py', mpi4py.__version__, 'cupy', cupy.__version__)" \
  || { echo "ERROR: numpy/mpi4py/cupy not importable in $(which python)"; exit 1; }
RUN="$BRHO_ROOT/results/e6_glam_$GLAM_TAG"
[ -d "$RUN" ] || { echo "ERROR: $RUN not found (check GLAM_TAG and BRHO_ROOT=$BRHO_ROOT)"; exit 1; }
echo "run: $RUN   ranks for the forward step: $SLURM_NTASKS"

# MPI launcher from the library mpi4py was built against (MPICH in darcyMCMC)
MPILIB=$(python -c "from mpi4py import MPI; print(MPI.Get_library_version())" | head -3 | tr '\n' ' ')
if [ -n "${MPI_LAUNCH:-}" ]; then LAUNCH="$MPI_LAUNCH"
elif echo "$MPILIB" | grep -qi "mpich" && [ -x "${CONDA_PREFIX:-/none}/bin/mpiexec" ]; then LAUNCH="$CONDA_PREFIX/bin/mpiexec"
elif echo "$MPILIB" | grep -qi "mpich"; then LAUNCH="srun --mpi=pmi2"
else LAUNCH="mpirun"; fi
echo "launcher: $LAUNCH"

step() { echo; echo "=== $(date +%T)  $* ==="; }
# a failing step is reported but does not stop the later ones (the figures that can be made are still made)
FAILED=()
run_step() { local name="$1"; shift; if "$@"; then :; else echo "STEP FAILED: $name (exit $?)"; FAILED+=("$name"); fi; }

if [ "$RUN_DIAGNOSTICS" = 1 ]; then
  step "1 diagnostics: RE traces, acceptance histograms"
  run_step diagnostics python experiments/plot_glam_diagnostics.py --exp-tag "$GLAM_TAG" "${DIAG_FLAGS[@]}"
fi
if [ "$RUN_TRACES" = 1 ]; then
  step "2 traces: Phi history, rho, theta entries"
  run_step traces python experiments/plot_glam_traces.py --exp-tag "$GLAM_TAG" "${TRACE_FLAGS[@]}"
fi
if [ "$RUN_SAMPLES" = 1 ]; then
  step "3 samples and the forward model on them"
  run_step samples $LAUNCH -n "$SLURM_NTASKS" python experiments/glam_postprocess.py --exp-tag "$GLAM_TAG" "${POST_FLAGS[@]}"
fi
if [ "$RUN_FIELD_FIGURES" = 1 ]; then
  step "4 field figures"
  run_step field_figures python experiments/plot_glam_fields.py --exp-tag "$GLAM_TAG" "${FIELD_FLAGS[@]}"
fi
if [ "$RUN_EXPORT" = 1 ]; then
  step "4b export for ParaView"
  run_step export python experiments/export_glam_fields.py --exp-tag "$GLAM_TAG" "${EXPORT_FLAGS[@]}"
fi
if [ "$RUN_MPSRF" = 1 ]; then
  step "5 MPSRF (mode: $MPSRF_MODE)"
  case "$MPSRF_MODE" in
    estimate) run_step mpsrf python experiments/glam_mpsrf.py --exp-tag "$GLAM_TAG" --dims "${MPSRF_DIMS[@]}" "${MPSRF_FLAGS[@]}" --estimate ;;
    psrf-only) run_step mpsrf python experiments/glam_mpsrf.py --exp-tag "$GLAM_TAG" "${MPSRF_FLAGS[@]}" --psrf-only ;;
    run)
      python experiments/glam_mpsrf.py --exp-tag "$GLAM_TAG" --dims "${MPSRF_DIMS[@]}" "${MPSRF_FLAGS[@]}" --estimate || true
      run_step mpsrf python experiments/glam_mpsrf.py --exp-tag "$GLAM_TAG" --dims "${MPSRF_DIMS[@]}" "${MPSRF_FLAGS[@]}" ;;
    *) echo "ERROR: MPSRF_MODE must be run, estimate or psrf-only"; FAILED+=("mpsrf") ;;
  esac
fi

echo
echo "figures: $BRHO_ROOT/figures/e6_glam_$GLAM_TAG   tables: $BRHO_ROOT/tables/e6_glam_$GLAM_TAG   arrays: $RUN/post"
[ "${#FAILED[@]}" -eq 0 ] || { echo "FAILED STEPS: ${FAILED[*]}"; exit 1; }
echo "all steps done"
