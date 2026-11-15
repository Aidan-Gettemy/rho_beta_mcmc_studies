# dd_end study: how to run it and bring the results back

Cells and frozen noise: `experiments/dd_end_cells.py` (eps 0.235 at ell 0.2 on 16 x 16, 0.2 at ell 0.1 on 32 x 32, 0.09 at ell 0.05 on 64 x 64;
interface-flux prior: measured faces sd eps, unmeasured sqrt(eps^2 + 0.03^2)).  Job writer: `cluster/gen_dd_end_sbatch.py`.  Collector:
`experiments/dd_end_collect.py`.  Plan and hypotheses: the "Experiment summary" and "Experiment to-do list" documents in the project.

## Stage loop
1. Generate the jobs (from the repo root): `python3 cluster/gen_dd_end_sbatch.py --stage N` (N = 2 global references, 3 ell 0.2 and 0.1 cells,
   4 ell 0.05 cells, 6 extra truths with `--truth 2027` / `2028`, 7 noise sweep, 8 overlap).  It writes `cluster/slurm/dd_end_s<N>_*.sbatch`,
   `cluster/submit_dd_end_s<N>_t<truth>.sh` and updates `runs/msm_local_kle_2026-11/dd_end/manifest.csv`.
2. Submit: `bash cluster/submit_dd_end_s<N>_t<truth>.sh`.  Each job uses 64 CPUs: 64 chains per sampler (per block for the DD).
3. Iteration counts and time limits in the generator (`ITERS`) are provisional.  Set them from the timing of the stage-2 jobs; extend any run
   with sup-MPSRF above 1.2 (`experiments/extend_run.py`) before reading it; slow convergence is acceptable but must be reported.
4. Return only the small files, on a results branch (no raw chains):
   `git checkout -b results-dd-end-s<N>`;
   `git add --sparse -f runs/msm_local_kle_2026-11/dd_end/manifest.csv runs/msm_local_kle_2026-11/dd_end/l*/*/*/summary.json runs/msm_local_kle_2026-11/dd_end/l*/*/*/post runs/msm_local_kle_2026-11/dd_end/l*/*/*/post_pooled`;
   check `git status --short | grep chains.npz` prints nothing; commit and push; merge the pull request into main.
5. Collect: `python experiments/dd_end_collect.py` writes `dd_end/master_table.csv` (sd, error, error/sd, 90% coverage, sup-MPSRF, ratios to the
   global reference, timing).

Check the pilot and restart logs of the first job before submitting the rest of a stage.
