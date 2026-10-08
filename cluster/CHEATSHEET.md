# Ganymede cheat sheet: rho_beta_mcmc_studies

Last updated: 2026-10-03

## 0. Every session

```bash
cd /scratch/ganymede2/arg210006/rho_beta_mcmc_studies      # same as ~/scratch/...
module purge; module load gnu9/9.4.0; module load conda/22.11.1
source activate $HOME/.conda/envs/gd24_numba
git switch main && git pull                                 # after merging PRs on GitHub
```

**Always set `BRHO_ROOT` to the study folder.** Without it, runs and reports write to the repo root.

| study | `BRHO_ROOT` | `PROBLEM_ARGS` |
|---|---|---|
| main 2D, r = 57 (32x32) | `$PWD/runs/cluster_kle` | `--truth-mode kle --sigma-rel 2e-3` |
| r = 208, seed 2026 | `$PWD/runs/cluster_l005` | `--n 64 --ell 0.05 --truth-mode kle --sigma-rel 2e-3` |
| r = 208, seed 2074 (representative) | `$PWD/runs/cluster_l005_s2074` | same as above + `--truth-seed 2074` |
| r = 208, stubs | `$PWD/runs/cluster_l005_s2063`, `_s2086` | + `--truth-seed 2063` / `2086` |

Note: `l005` is a lowercase L ("ell 0.05"), not 1005.

**Confused about where the 2D results are?** `runs/cluster_kle` is the only 2D (32x32) folder: live study on the cluster and results in the repo. `runs/e5_kle_2026-09` was retired on 2026-10-05; read its name as `cluster_kle`. Run status: `runs/cluster_kle/STUDY.md`.

## 1. Submit a sampler run (one node, 64 chains)

```bash
export BRHO_ROOT=$PWD/runs/<study>
export PROBLEM_ARGS="<from the table>"
TAG=<name> INNOV=global E5_ARGS="--n-total N --n-burn B --thin T [flags]" \
    sbatch --partition=cpu-preempt --time=12:00:00 cluster/slurm/node_e5_soft.sbatch
```

- **Output folder:** `results/e5_soft_<TAG>`. With no TAG it is `e5_soft`.
- **`INNOV`:** `global` is E5 and `block` is E8 (block pCN).
- **Prior test (likelihood off):** prefix with `PRIOR_TEST=1`.
- **Sizing:** iterations = 2.0M solves / solves per iteration.
  - E5 is 3 solves/it; E5 with `--theta-repeat k` is 2 + k; E8 2x2 is 16; E8 3x3 is 28.
  - Rule of thumb: burn-in about 8% of iterations, thin so you keep about 5,000 samples.

### Learned noise level (PR #76)

```bash
# r = 57
BRHO_ROOT=$PWD/runs/cluster_kle PROBLEM_ARGS="--truth-mode kle --sigma-rel 2e-3" TAG=a7_noise \
    E5_ARGS="--n-total 222300 --n-burn 18000 --thin 45 --theta-repeat 7 --learn-noise 1" \
    sbatch --partition=cpu-preempt --time=12:00:00 cluster/slurm/node_e5_soft.sbatch
# r = 208, seed 2026
BRHO_ROOT=$PWD/runs/cluster_l005 PROBLEM_ARGS="--n 64 --ell 0.05 --truth-mode kle --sigma-rel 2e-3" TAG=a7_noise \
    E5_ARGS="--n-total 222200 --n-burn 10000 --thin 33 --theta-repeat 7 --learn-noise 1" \
    sbatch --partition=cpu-preempt --time=12:00:00 cluster/slurm/node_e5_soft.sbatch
```

### E10: scale blocks + log beta collapsed + theta-repeat 7 + learned noise, to convergence

TAG `noise_collapse2` = with the lb slice refresh (PR after #91). The first run (TAG `noise_collapse`) froze 13 of 64 chains; do not resume it.

```bash
# r = 57, 4.0M solves
BRHO_ROOT=$PWD/runs/cluster_kle PROBLEM_ARGS="--truth-mode kle --sigma-rel 2e-3" TAG=noise_collapse2 INNOV=scale \
    E5_ARGS="--n-total 444400 --n-burn 18000 --thin 90 --theta-repeat 7 --learn-noise 1 --scale-edges 20,60 --collapse-beta 1" \
    sbatch --partition=cpu-preempt --time=12:00:00 cluster/slurm/node_e5_soft.sbatch
# r = 208, 6.0M solves (resume with RESUME=1 TAG=noise_collapse2 INNOV=scale)
BRHO_ROOT=$PWD/runs/cluster_l005 PROBLEM_ARGS="--n 64 --ell 0.05 --truth-mode kle --sigma-rel 2e-3" TAG=noise_collapse2 INNOV=scale \
    E5_ARGS="--n-total 666600 --n-burn 10000 --thin 99 --theta-repeat 7 --learn-noise 1 --scale-edges 20,60 --collapse-beta 1" \
    sbatch --partition=cpu-preempt --time=12:00:00 cluster/slurm/node_e5_soft.sbatch
# report: python reports/report_soft.py --exp e5_soft_scale_noise_collapse2   (as a job, section 5)
```

### Scale-block pCN with learned noise (E9; compare with a7_noise)

```bash
# r = 57
BRHO_ROOT=$PWD/runs/cluster_kle PROBLEM_ARGS="--truth-mode kle --sigma-rel 2e-3" TAG=noise INNOV=scale \
    E5_ARGS="--n-total 222300 --n-burn 18000 --thin 45 --theta-repeat 7 --learn-noise 1 --scale-edges 20,60" \
    sbatch --partition=cpu-preempt --time=12:00:00 cluster/slurm/node_e5_soft.sbatch
# r = 208, seed 2026
BRHO_ROOT=$PWD/runs/cluster_l005 PROBLEM_ARGS="--n 64 --ell 0.05 --truth-mode kle --sigma-rel 2e-3" TAG=noise INNOV=scale \
    E5_ARGS="--n-total 222200 --n-burn 10000 --thin 33 --theta-repeat 7 --learn-noise 1 --scale-edges 20,60" \
    sbatch --partition=cpu-preempt --time=12:00:00 cluster/slurm/node_e5_soft.sbatch
# report: python reports/report_soft.py --exp e5_soft_scale_noise   (as a job, section 5)
```

### Theta-repeat control (r = 57)

```bash
TAG=a14 E5_ARGS="--n-total 125000 --n-burn 10000 --thin 25 --theta-repeat 14" sbatch cluster/slurm/node_e5_soft.sbatch
TAG=a7  E5_ARGS="--n-total 222300 --n-burn 18000 --thin 45 --theta-repeat 7"  sbatch cluster/slurm/node_e5_soft.sbatch
TAG=a4  E5_ARGS="--n-total 333400 --n-burn 27000 --thin 67 --theta-repeat 4"  sbatch cluster/slurm/node_e5_soft.sbatch
```

### E1 (naive) with theta-repeat, MPI job (r = 57)

```bash
BRHO_ROOT=$PWD/runs/cluster_kle PROBLEM_ARGS="--truth-mode kle --sigma-rel 2e-3" TAG=a7 MOVE=blocked \
  E1_ARGS_BLOCKED="--n-iter 250000 --n-burn 5000 --thin 50 --checkpoint-every 10000 --log-every 25000 --theta-repeat 7" \
  sbatch --partition=cpu-preempt cluster/slurm/mpi_e1_naive.sbatch
# resume: RESUME=1 TAG=a7 MOVE=blocked sbatch --partition=cpu-preempt cluster/slurm/mpi_e1_naive.sbatch
```

## 2. Whole new study folder in one go (E3 curve -> setup -> node jobs)

```bash
BRHO_ROOT=$PWD/runs/<new_study> PROBLEM_ARGS="..." \
  RUNS="e5 b2 b3" SOLVES=2000000 THETA_REPEAT=1 PART=cpu-preempt TIME=12:00:00 \
  bash cluster/overnight_scaling.sh
```

- `RUNS`: any of `e5`, `b2`, `b3`. Use `SOLVES=500000` for stubs.
- E5 extras: `E5_EXTRA="--learn-noise 1"` and `E5_TAG=a7_noise`.
- **Only for NEW folders.** It reruns E3 and the setup.

## 3. Resume after a time limit or preemption

Use the same command plus `RESUME=1`. It continues from the checkpoints and nothing is redone.

```bash
BRHO_ROOT=$PWD/runs/<study> RESUME=1 INNOV=global TAG=<tag> \
    sbatch --partition=cpu-preempt --time=12:00:00 cluster/slurm/node_e5_soft.sbatch
```

- Leave out `TAG` for the plain `e5_soft` run.
- For E8 use `INNOV=block`, with `TAG=3x3` for the 3x3 run.
- **Do not leave out `RESUME=1`:** a non-resume submit reruns setup and deletes the chains.

## 4. Check on jobs

```bash
squeue -u $USER                                    # PD = waiting, R = running
tail logs/brho-e5-<jobid>.out                      # job log (per-chain logs: logs/brho-e5-<jobid>-rank*.out)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # finished jobs: state, time, memory
scancel <jobid>
python experiments/run_soft_sampler.py --status --theta-innovation global [--exp-tag <tag>]   # chain progress
```

## 5. Reports: as a job, NOT on the login node (it runs out of memory)

```bash
export BRHO_ROOT=$PWD/runs/<study>
mkdir -p logs
sbatch --partition=cpu-preempt --ntasks=1 --cpus-per-task=4 --mem=48G --time=01:00:00 \
    -J brho-report -o logs/brho-report-%j.out --export=ALL \
    --wrap "source cluster/env.sh; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4; \
            python reports/report_soft.py --exp e5_soft_<tag> && python reports/report_mpsrf.py"
```

- Expect about 10-20 minutes at 64x64.
- `report_mpsrf.py` with no `--exp` redoes the MPSRF figure for every run in the study; `--exp e5_soft_<tag>` does just one.
- If memory still runs out, check `sacct ... MaxRSS` and raise `--mem` to 64G.
- Interactive alternative: `salloc --partition=cpu-preempt --cpus-per-task=4 --mem=48G --time=01:00:00`.
- Block-pCN geometry figures: `python experiments/run_block_prior_test.py --from-exp e5_soft_block --blocks 2`.

### 3-move (E4) run on the main 2D truth, for the β-upgrade validation (2.0M solves)

```bash
BRHO_ROOT=$PWD/runs/cluster_kle PROBLEM_ARGS="--truth-mode kle --sigma-rel 2e-3" \
    E4_ARGS="--n-total 667000 --n-burn 30000 --thin 200" \
    sbatch --partition=cpu-preempt --time=12:00:00 cluster/slurm/node_e4_sampler.sbatch
# continue after a time limit: BRHO_ROOT=$PWD/runs/cluster_kle RESUME=1 sbatch --partition=cpu-preempt --time=12:00:00 cluster/slurm/node_e4_sampler.sbatch
# reports (as a job, section 5): python reports/report_sampler.py && python reports/report_beta_upgrade.py --solve-check 200
```

Same truth and σ_rel as the E5 reference in `runs/cluster_kle/tables/e5_soft` (β 2.14 [1.43, 3.21]), so the upgraded β can be checked against it.

### Ablation: the 3-move sampler without move C (A+B only), 2.0M solves

Same study, truth, data and σ_rel as the 2e-3 E4 run, in its own folder `results/e4_sampler_ab` (the E4 chains are untouched). Two solves per iteration, so 1M iterations = 2.0M solves; burn-in 45k iterations = 90k solves and thinning 150 put the convergence window at the same solve count (92.7k) as E4 and the burn-matched E1. Same seeds as E4, so chain k starts from the same prior draw.

```bash
BRHO_ROOT=$PWD/runs/cluster_kle_sig2e3 PROBLEM_ARGS="--truth-mode kle --sigma-rel 2e-3" \
    E4_TAG=ab E4_ARGS="--moves AB --n-total 1000000 --n-burn 45000 --thin 150" \
    sbatch --partition=g1mig --time=12:00:00 cluster/slurm/node_e4_sampler.sbatch
# continue after a time limit (keep E4_TAG):
# BRHO_ROOT=$PWD/runs/cluster_kle_sig2e3 E4_TAG=ab RESUME=1 sbatch --partition=g1mig --time=12:00:00 cluster/slurm/node_e4_sampler.sbatch
```

- The job ends with `report_sampler.py`, `report_mpsrf.py` and `report_beta_upgrade.py` for `e4_sampler_ab`.
- Then `python reports/report_e1_vs_e4.py` overlays it on the MPSRF comparison, adds `fig_e4ab_psrf.png` and an "E4 A+B only" column to the table.
- Status: `BRHO_ROOT=$PWD/runs/cluster_kle_sig2e3 python experiments/run_curve_sampler.py --status --exp-tag ab`.

### β upgrade of a 3-move (E4) run: honest β interval, no PDE solves

```bash
export BRHO_ROOT=$PWD/runs/<study>      # the study holding results/e4_sampler/chain*.npz
sbatch --partition=cpu-preempt --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:30:00 \
    -J brho-upgrade -o logs/brho-upgrade-%j.out --export=ALL \
    --wrap "source cluster/env.sh; python reports/report_beta_upgrade.py --exp e4_sampler --solve-check 200"
```

- Output: `tables/e4_sampler/beta_upgrade.{md,json}`, `figures/e4_sampler/fig_beta_upgrade.png`.
- `--solve-check 200` re-solves 200 upgraded samples (seconds) to measure the misfit change from the part of the constant field outside the KLE span; leave it out for zero solves.
- Check the weight ESS line: a small ESS (a few %) means the weights are dominated by a few samples; report it.

### Baseline: freeze (ρ*, θ*) at the lowest-Φ sample, pCN on β with PDE solves

The "obvious" fix, run so it can be shown next to the level swap. It uses about 80k PDE solves, roughly 3 minutes at 32×32.

```bash
export BRHO_ROOT=$PWD/runs/<study>
sbatch --partition=g1mig --ntasks=1 --cpus-per-task=2 --mem=32G --time=01:00:00 \
    -J brho-frozen -o logs/brho-frozen-%j.out --export=ALL \
    --wrap "source cluster/env.sh; python reports/frozen_beta_pcn.py --exp e4_sampler"
```

- Output: `tables/e4_sampler/frozen_beta_pcn.{md,json}`, `figures/e4_sampler/fig_frozen_beta_pcn.png`.
- The table also gives the exact 1D posterior on a grid (no MCMC). Its sd(log β) should be about σ_rel.
- Then `python reports/plot_beta_methods.py --exp e4_sampler` (seconds, no solves): `fig_beta_methods.png`, the five β estimates side by side, read from `frozen_beta_pcn.json`.

### Naive (E1) vs 3-move (E4) at equal cost

E1 is `runs/cluster_kle/results/e1_blocked` and E4 is `runs/cluster_kle_sig2e3/results/e4_sampler`. Both use the same truth, data and σ_rel 2e-3, with 2.0M solves per chain. Run it from the repo root; it needs no BRHO_ROOT and takes a few minutes, mostly reading the E1 traces.

```bash
sbatch --partition=g1mig --ntasks=1 --cpus-per-task=2 --mem=32G --time=01:00:00 \
    -J brho-e1e4 -o logs/brho-e1e4-%j.out --export=ALL \
    --wrap "source cluster/env.sh; python reports/report_e1_vs_e4.py"
```

- `runs/cluster_kle/tables/e1_blocked/e1_intervals.{md,json}`: E1's pooled medians and 90% intervals (needs the E1 chain files).
- `runs/cluster_kle/figures/e1_blocked/fig_e1_field.png`: the E1 counterpart of `fig_e4_field.png` (needs the E1 chain files).
- `runs/cluster_kle_sig2e3/figures/e1_vs_e4/`: `fig_e1_psrf.png`, `fig_e1_mpsrf.png`, `fig_e1_vs_e4_mpsrf.png` (drawn from the committed `mpsrf_*.csv`).
- `runs/cluster_kle_sig2e3/tables/e1_vs_e4/e1_vs_e4.{md,json}`: the side-by-side table.
- Burn-in: by default E1's convergence curves are recomputed so their window starts at the same forward-solve count as E4's (E4 keeps no θ before its 30k-iteration burn-in): `mpsrf_{theta,all}_burn_matched.csv` and `mpsrf_burn_matched.json` in `runs/cluster_kle/tables/e1_blocked/`. Also writes `fig_e4_psrf.png`.

## 6. Send results to GitHub

```bash
git fetch origin
git switch -c results-<what> origin/main           # new branch FIRST, never commit on main
git add --sparse runs/<study>
git commit -m "results: <what>"
git push -u origin results-<what>
```

- Then open the PR on GitHub, or ask Claude to.
- **Sparse clone:** after the first results commit of a study inside the repo, run `git sparse-checkout add runs/<study>`. Otherwise the next pull removes those files from your working tree. The same command brings them back if they've vanished.
- **Accidentally committed on main:** `git switch -c results-<what>` (takes the commit with you), then `git branch -f main origin/main`.
- **Study kept outside the repo (on scratch):** `bash cluster/export_results.sh /scratch/$USER/brho/<study> <study>` first, then the same git steps.

## 7. Quick fixes

| symptom | fix |
|---|---|
| `FileNotFoundError ... manifest.json` | wrong or missing `BRHO_ROOT` (or `1005` instead of `l005`) |
| report killed / out of memory | run it as the job in section 5 |
| `unrecognized arguments: --learn-noise` | merge PR #76 and `git pull` |
| run stopped part-way | section 3 (`RESUME=1`) |
| committed files vanished after a pull | `git sparse-checkout add runs/<study>` |
| "branch already exists" | `git switch <branch>` instead of `-c`, or pick a new name |
