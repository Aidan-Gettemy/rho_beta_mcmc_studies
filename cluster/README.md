# Running on a cluster (SLURM templates)

Every experiment splits into **setup → many independent tasks → collect**:

| Step | Command | What it does |
|---|---|---|
| setup | `run_grid_study.py --setup-only` / `run_curve_sampler.py --setup-only --n-chains N` | loads the study's KLE basis and reference field from `$BRHO_ROOT/store/` (building them once if missing, with BLAS threads = `--cpus-per-task`), freezes them into the run (`kle.npz`, `truth.npz`), writes `manifest.json` |
| task k | `--pair-index env` / `--chain-id env` | runs grid pair / chain k, reading only the manifest and the frozen files (`env` = `$SLURM_ARRAY_TASK_ID`) |
| collect | `run_grid_study.py --collect`, `reports/report_all.py` | merges results, draws every figure and table |

Tasks never take settings from their own command line, so array jobs cannot
disagree about the configuration.

## First time on a new cluster

1. `git clone` the repo, create an environment: `pip install -r requirements.txt`
   (numba is optional; without it a vectorized NumPy path is used).
2. Edit `cluster/env.sh`: module/conda lines, `SLURM_ACCOUNT_ARGS`, study settings.
3. `sbatch cluster/slurm/smoke.sbatch`: it runs the tests, the solver benchmark
   and the whole pipeline at tiny budgets. Check `logs/brho-smoke-*.out`:
   every test must say PASS, and with numba installed
   `test_backends_agree_when_numba_present` actually compares the compiled kernel.
4. `bash cluster/submit_pipeline.sh` submits E2, E3, E4 and the reports as one
   dependency chain.

## Everyday use

- **Settings** live in `cluster/env.sh` (`PROBLEM_ARGS`, `E3_ARGS`, `E4_CHAINS`,
  `E4_ARGS`) and are recorded in each run's `manifest.json`.
- **Separate studies**: set `BRHO_ROOT=$PWD/runs/<study>` before submitting.
- **Progress**: `squeue -u $USER`; `python experiments/run_curve_sampler.py --status`.
- **Time limits**: chains checkpoint every `chunk` iterations. If tasks hit the
  wall time, resubmit the same array (`sbatch --array=... cluster/slurm/e4_chain.sbatch`)
  and they resume. Finished grid pairs are skipped.
- **Likelihood-off check** before a production E4:
  `PRIOR_TEST=1 bash cluster/submit_pipeline.sh e4`.
- **Only E4 again** (reusing the curve): `bash cluster/submit_pipeline.sh e4`.
- **PBS/other schedulers**: `--chain-id env` also reads `$PBS_ARRAY_INDEX` /
  `$PBS_ARRAYID`; translate the `#SBATCH` headers and `submit_pipeline.sh`.

## Why the basis is frozen

The covariance on a square grid has exactly repeated eigenvalues
(λᵢλⱼ = λⱼλᵢ). Inside each repeated pair, the eigenvectors that `eigh` returns
depend on floating-point rounding, and so on the BLAS build, the CPU and the
thread count. On one machine, changing only the thread count rotated 20 of
the 56 modes, and the same θ then gave log K fields differing by up to 1.1.
Chains on different nodes would silently use different coordinates. The
setup step therefore computes the basis once and saves it in `kle.npz`; every
task and report loads it (`Problem(pc, frozen=dir)`).

## The build-once store

`$BRHO_ROOT/store/{kle,truth}/<hash>.npz` holds each KLE basis and reference
field, and the hash covers exactly the settings that determine it. The first
setup job that needs one builds it, while the others wait on a lock file and then
load it. E2 setup and E3 can therefore start together, and E4 setup reuses what
they built. A stale `.lock` is only left behind if a job was killed mid-build;
delete it and resubmit. The store is not committed (runs keep frozen copies).

## Seeds

Chain / pair k uses `numpy.random.SeedSequence([seed0, k])` (`--seed-scheme spawn`,
the default): independent streams for any number of chains, reproducible from
the index alone. `--seed-scheme legacy` gives the July seeds (seed0 + 17k for the
grid, + 13k for the sampler).

## Getting results back

Commit the study's `runs/<study>/` folder (figures, tables, npz; small) on a
branch and push. Everything the reports need is inside it, including the
frozen basis, so the figures can be redrawn anywhere without rerunning.
