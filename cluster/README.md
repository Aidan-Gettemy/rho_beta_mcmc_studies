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
3. Check the `#SBATCH` headers in `cluster/slurm/*.sbatch`: every script sets
   `--partition=g1mig`; adjust time and memory as needed. `cluster/env.sh` runs
   inside the job and cannot change scheduling; use `sbatch --partition=...` or
   `export SBATCH_PARTITION=...` in your login shell to override without editing.
4. Smoke test: `sbatch cluster/slurm/smoke.sbatch` (tests, solver benchmark,
   tiny pipeline). Every test must PASS. With numba installed,
   `test_backends_agree_when_numba_present` checks the compiled kernel against
   the NumPy one.

## E1 with mpi4py

`cluster/slurm/mpi_e1_naive.sbatch` runs the naive-model study as one
mpi4py job (`MOVE=blocked` default, `global`, or `both` with a comparison), in the style of `MCMCpar.py`: `mpirun -np 64 python3
experiments/run_e1_naive.py`, one chain per rank. Rank 0 sets up the run (basis
and truth from the store, frozen copies, numba compiled) and broadcasts success;
summaries are gathered on rank 0. The report runs afterwards in the same job.
To continue after a time limit, use `RESUME=1 sbatch ...`; to start over, `FRESH=1`.
The budgets are `E1_ARGS_BLOCKED` and `E1_ARGS_GLOBAL` in the script; by default each
move gets 400k PDE solves per chain (200k blocked iterations at 2 solves each, 400k
global ones at 1).

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

### E5 study: beta free, truth in the KLE (decided 2026-09-28, see LOG.md)

Its own study folder, and `--truth-mode kle` in EVERY step (E3 first: K*
comes from the truth, and E5 refuses a curve built under another truth):

```bash
export BRHO_ROOT=$PWD/runs/cluster_kle
export PROBLEM_ARGS="--truth-mode kle --sigma-rel 2e-3"
sbatch cluster/slurm/node_e3_curve.sbatch                       # curve for this truth
PRIOR_TEST=1 sbatch cluster/slurm/node_e5_soft.sbatch           # likelihood off
sbatch cluster/slurm/node_e5_soft.sbatch                        # E5: 64 chains
MOVE=blocked sbatch cluster/slurm/mpi_e1_naive.sbatch           # E1 on the same truth
```
(sbatch passes BRHO_ROOT and PROBLEM_ARGS into the job; env.sh only fills
them in when they are unset.)

Numba: each node job compiles once in its setup step into a node-local cache
(`NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID`). The ranks then load the
compiled kernel instead of compiling 64 times at once.

### E8: block (subdomain) pCN for theta (2026-10-01)

E5 with move A replaced by Metropolis-within-Gibbs over local subspaces of θ
(`brho/blockpcn.py`): a coarse move, one move per 2×2 primal block and one per
3×3 dual block (interfaces at 0.25 / 0.75), each with its own adapted step.
Same study folder as E5 (`runs/cluster_kle`), so the curve, KLE and truth are
shared. In 2D one iteration costs 16 solves (coarse + 4 + 9, plus B and D), so
2.0M solves per chain (equal to E5 and E1) is 125k iterations.

```bash
export BRHO_ROOT=$PWD/runs/cluster_kle
export PROBLEM_ARGS="--truth-mode kle --sigma-rel 2e-3"
python experiments/run_block_prior_test.py                         # 1. geometry + likelihood-off check (seconds)
PRIOR_TEST=1 INNOV=block sbatch cluster/slurm/node_e5_soft.sbatch    # 2. full-sampler prior test
INNOV=block E5_ARGS="--n-total 125000 --n-burn 10000 --thin 25" \
    sbatch cluster/slurm/node_e5_soft.sbatch                         # 3. E8 at 2.0M solves
INNOV=block_random E5_ARGS="--n-total 500000 --n-burn 30000 --thin 100" \
    sbatch cluster/slurm/node_e5_soft.sbatch                         # 4. optional: one random block per iteration (4 solves / it)
```
`report_mpsrf.py` reads the solves per iteration from each run's manifest, so
E5, E7 and E8 share the PDE-solve axis. Block settings: `--block-n` (2),
`--block-energy` (0.9; 3D needs about 0.8, at 0.9 most 3D dual blocks are
empty), `--block-dual` (1), `--block-halo` (0), `--block-s0` / `--block-target`
(0 = use move A's).

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


## 3D study (2026-09-29)

Same pipeline as the 2D main study, in its own study folder, with the problem
switched to the unit cube (`--dim 3`). Settings (confirmed): 16³ cells,
ℓ = 0.2 (3.2 cells per ℓ, as in 2D), separable Nyström KLE with nq = 40
(r = 78 at 95% energy), the same boundary conditions (unit influx at x = 0,
p = 0 at x = 1, no flow on the side faces), a 3D chessboard of 2,048 sensors,
truth `kle` with β = 2.3, ρ = 1.5, 1% pressure noise, σ_e = 0.05,
σ_rel = 2e-3. One forward solve is about 10 ms, so 2.0M solves per chain is
about 5.5 h; the launchers checkpoint, so resubmit with `RESUME=1` after a
time limit.

```bash
export BRHO_ROOT=$PWD/runs/cluster_3d
export PROBLEM_ARGS="--dim 3 --n 16 --ell 0.2 --kle-method nystrom_sep --kle-nq 40 --truth-mode kle --sigma-rel 2e-3"
python benchmarks/bench_solver.py --dims 3 --sizes 16          # 0. timing on this machine
sbatch cluster/slurm/node_e3_curve.sbatch                        # 1. E3 curve (also the setup/truth)
PRIOR_TEST=1 sbatch cluster/slurm/node_e5_soft.sbatch            # 2. E5 prior test
E1_ARGS_BLOCKED="--n-iter 1000000 --n-burn 20000 --thin 200 --checkpoint-every 10000 --log-every 20000" \
    sbatch cluster/slurm/mpi_e1_naive.sbatch                     # 3. E1 at 2.0M solves
E5_ARGS="--n-total 667000 --n-burn 30000 --thin 100" \
    sbatch cluster/slurm/node_e5_soft.sbatch                     # 3. E5 at 2.0M solves
python experiments/run_ms_prior_test.py                          # 4. multiscale prior test (2x2x2)
PRIOR_TEST=1 INNOV=ms_raw   sbatch cluster/slurm/node_e5_soft.sbatch
PRIOR_TEST=1 INNOV=ms_white sbatch cluster/slurm/node_e5_soft.sbatch
INNOV=ms_white E5_ARGS="--n-total 667000 --n-burn 30000 --thin 100" \
    sbatch cluster/slurm/node_e5_soft.sbatch                     # 5. E7
bash cluster/export_results.sh $BRHO_ROOT e5_3d_2026-10           # export (study folder name)
```

Field figures show the horizontal mid-plane z = 0.53 of the cube (noted in
each panel title). With 2×2×2 subdomains that plane lies next to the
z = 0.5 interface, so the raw multiscale strategy is deficient across the
whole plotted plane, not only along a cross.
