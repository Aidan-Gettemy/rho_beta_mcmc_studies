# 2D (beta, rho) experiments.  Set ROOT to keep runs apart, e.g.
#   make e2 ROOT=runs/july_raw ARGS="--truth-mode raw --seed-scheme legacy"
ROOT ?= .
PY = BRHO_ROOT=$(ROOT) python3
ARGS ?=

.PHONY: test bench e1 e1-global e2 e3 prior e4 reports all quick clean-quick

test:
	python3 tests/run_tests.py

bench:         ## forward-solve timings (numba vs NumPy, splu vs CG)
	python3 benchmarks/bench_solver.py

e1:            ## naive (theta,beta,rho) model, blocked pCN, 4 local chains (mpi: cluster/slurm/mpi_e1_naive.sbatch)
	$(PY) experiments/run_e1_naive.py --local-chains 4 $(ARGS)
	$(PY) reports/report_e1.py
e1-global:     ## same with one global pCN move (for the comparison)
	$(PY) experiments/run_e1_naive.py --move global --local-chains 4 $(ARGS)
	$(PY) reports/report_e1.py
e2:            ## 5x5 fixed-(beta,rho) pCN grid        (~1 min, 2 cores)
	$(PY) experiments/run_grid_study.py $(ARGS)
e3:            ## offline beta_bar(rho) curve           (~10 s)
	$(PY) experiments/run_beta_curve.py $(ARGS)
prior:         ## E4 sampler with likelihood off         (~10 s, no solves)
	$(PY) experiments/run_curve_sampler.py --fresh --prior-test 1 --n-total 40000 --n-burn 2000 --thin 10 --chunk 20000 $(ARGS)
e4:            ## 3-step sampler, 4 chains x 51k it     (~10-20 min)
	$(PY) experiments/run_curve_sampler.py $(ARGS)
reports:
	$(PY) reports/report_all.py

all: test e2 e3 prior e4 reports

quick:         ## smoke test of the whole pipeline into runs/quick
	BRHO_ROOT=runs/quick python3 experiments/run_e1_naive.py --quick --fresh --local-chains 2
	BRHO_ROOT=runs/quick python3 experiments/run_e1_naive.py --quick --fresh --local-chains 2 --move global
	BRHO_ROOT=runs/quick python3 experiments/run_grid_study.py --quick
	BRHO_ROOT=runs/quick python3 experiments/run_beta_curve.py --quick
	BRHO_ROOT=runs/quick python3 experiments/run_curve_sampler.py --quick --fresh
	BRHO_ROOT=runs/quick python3 experiments/run_curve_sampler.py --quick --fresh --prior-test 1
	BRHO_ROOT=runs/quick python3 reports/report_all.py

clean-quick:
	rm -rf runs/quick
