# Running on a cluster

Every experiment splits into **setup → many independent tasks → collect**:

| Step | Command | What it does |
|---|---|---|
| setup | `run_grid_study.py --setup-only` / `run_curve_sampler.py --setup-only --n-chains N` | loads the study's KLE basis and reference field from `$BRHO_ROOT/store/` (building them once if missing), freezes them into the run (`kle.npz`, `truth.npz`), writes `manifest.json`, compiles numba once |
| tasks | `--pair-index env` / `--chain-id env` | each process finds its tasks from the launcher: an array index, or an `srun`/`mpirun` rank (rank r of s runs tasks r, r+s, …) |
| collect | `run_grid_study.py --collect`, `reports/*.py` | merges results, draws every figure and table |

Tasks never take settings from their own command line, so parallel processes
cannot disagree about the configuration. No MPI library is needed: ranks only
read their rank from the environment.

Two ways to run the same steps:

- **Single-node jobs** (`cluster/slurm/node_*.sbatch`): one job on one node runs
  setup, then all tasks at once through `srun` (or `mpirun`), then the reports.
  This matches a node with many cores, used whole, like your MCMC jobs.
  **Recommended for you.**
- **Array jobs** (`cluster/submit_pipeline.sh` with the other templates): many
  one-core jobs chained by dependencies. Better when the scheduler packs small
  jobs well or you want more tasks than one node holds.

## Getting the code onto the cluster (code only, no results)

The repo is private, so the cluster needs read access. A **deploy key** gives it
access to this one repository and nothing else:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/brho_deploy -N ""      # on the cluster
cat ~/.ssh/brho_deploy.pub   # paste into GitHub: repo Settings > Deploy keys
                             # (tick "Allow write access" only to push results)
cat >> ~/.ssh/config <<'CFG'
Host github-brho
    HostName github.com
    IdentityFile ~/.ssh/brho_deploy
CFG
```

Then clone **without the committed results**. A partial, sparse clone downloads
no files under `runs/`, now or in history:

```bash
git clone --filter=blob:none --no-checkout git@github-brho:Aidan-Gettemy/rho_beta_mcmc_studies.git
cd rho_beta_mcmc_studies
git sparse-checkout set --no-cone '/*' '!/runs/'
git checkout main
git pull                      # later updates, still without runs/
```

GitHub's **Download ZIP** also leaves `runs/` out (see `.gitattributes`), but
it gives no `git pull` for updates and no way to push results.

## First time on the cluster

1. Clone as above. Use your numba conda environment, which needs numpy, scipy,
   matplotlib and numba (`requirements.txt`).
2. Edit `cluster/env.sh`: the `module load` / `source activate` lines, and
   `BRHO_ROOT`. Put studies on scratch, e.g. `/scratch/$USER/brho/<study>`.
3. Edit the `#SBATCH` headers in `cluster/slurm/node_*.sbatch`: partition, time
   and memory.
4. Smoke test: `sbatch cluster/slurm/smoke.sbatch` (tests, solver benchmark,
   tiny pipeline). Every test must PASS. With numba installed,
   `test_backends_agree_when_numba_present` checks the compiled kernel against
   the NumPy one.

## Running the experiments (single-node jobs)

From the repo root, with `cluster/env.sh` set:

```bash
sbatch cluster/slurm/node_e2_grid.sbatch        # E2: 5x5 grid, 25 ranks
sbatch cluster/slurm/node_e3_curve.sbatch       # E3: beta curve, 1 process
jid=$(sbatch --parsable cluster/slurm/node_e3_curve.sbatch)       # or chained:
PRIOR_TEST=1 sbatch --dependency=afterok:$jid cluster/slurm/node_e4_sampler.sbatch
sbatch cluster/slurm/node_e4_sampler.sbatch     # E4: 64 chains (after E3)
RESUME=1 sbatch cluster/slurm/node_e4_sampler.sbatch   # continue after a time limit
```

Numba: each node job compiles once in its setup step into a node-local cache
(`NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID`). The ranks then load the
compiled kernel instead of compiling 64 times at once.

## Bringing results back

Keep the raw run on scratch and commit only what is shareable:

```bash
bash cluster/export_results.sh /scratch/$USER/brho/<study> <study>
git checkout -b results-<study>
git add --sparse runs/<study> && git commit -m "results: <study>" && git push -u origin results-<study>
```

`export_results.sh` copies figures, tables, `SUMMARY.md`, manifests, the frozen
basis and truth, and small collected files (`grid.npz`, curves). It leaves
chain checkpoints and per-pair files behind, and `.gitignore` refuses them too,
so the repo stays small.

## Array jobs (alternative)

1. Edit `SLURM_ACCOUNT_ARGS` in `cluster/env.sh`.
2. `bash cluster/submit_pipeline.sh` submits E2, E3, E4 and the reports as one
   dependency chain.

- **Settings** live in `cluster/env.sh` (`PROBLEM_ARGS`, `E3_ARGS`, `E4_CHAINS`,
  `E4_ARGS`) and are recorded in each run's `manifest.json`.
- **Progress**: `squeue -u $USER`; `python experiments/run_curve_sampler.py --status`.
- **Time limits**: chains checkpoint every `chunk` iterations. If tasks hit the
  wall time, resubmit the same array and they resume. Finished grid pairs are skipped.
- **Likelihood-off check** before a production E4: `PRIOR_TEST=1 bash cluster/submit_pipeline.sh e4`.
- **PBS/other schedulers**: `--chain-id env` also reads `$PBS_ARRAY_INDEX` /
  `$PBS_ARRAYID` and MPI ranks; translate the `#SBATCH` headers.

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
