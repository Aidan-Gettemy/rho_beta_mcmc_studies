"""experiments/s80k_chain_analysis.py on a toy GLAM run with a keep-own mask (ThreadWorld ranks, no solver)."""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from test_glam import _beta  # noqa: E402
from brho import config as C  # noqa: E402
from brho.glam.comm import ThreadWorld  # noqa: E402
from brho.glam.post import load_chain  # noqa: E402
from brho.glam.sampler import run_glam  # noqa: E402
from brho.glam.toy import ToyModel  # noqa: E402
import s80k_chain_analysis as A  # noqa: E402

N_SUB, M, P = 4, 6, 4


def _masked_run(n_iter, out):
    model = ToyModel(_beta(), n_sub=N_SUB, m=M, n_cells=60, n_times=3, seed=5, sigma_conc=0.05, sigma_dp=0.05)
    cfg = C.GlamConfig(n_iter=n_iter, window1=60, window2=n_iter, burnin_length=20, global_stage_length=20, local_stage_length=15,
                       min_global_stage_length=10, min_local_stage_length=10, adapt_interval=10, beta_g0=0.1, beta_l0=0.2,
                       quality_tolerance=50.0, minimum_anchor_distance=0.5, restart_scale=0.0, random_reach_attempts=0, save_every=25, log_every=0)
    keep = [np.arange(M) >= 3 for _ in range(N_SUB)] + [np.zeros(1, bool)]          # first 3 of every block installed, the rest kept
    return ThreadWorld(P).run(lambda comm: run_glam(model, cfg, comm, out, keep_mask=keep, log=lambda *_: None))


def test_exchanges_are_found_in_the_chains_and_the_analysis_runs_end_to_end():
    out, root = tempfile.mkdtemp(), tempfile.mkdtemp()
    res = _masked_run(260, out)
    true_its = [e["iteration"] for e in res[0]["exchanges"]]
    assert len(true_its) >= 5
    # the exchange iterations read from the chains are exactly the recorded ones
    assert A.exchange_iterations(load_chain(out, 0, theta=False)).tolist() == true_its
    exch = {e["iteration"]: e["archive"] for e in res[0]["exchanges"]}
    rows = A.per_exchange_phi(out, list(range(P)), exch, n_expected=260)
    assert [r["iteration"] for r in rows] == true_its
    assert all(np.isfinite(r["pre_median"]) and np.isfinite(r["anchor_phi_best"]) for r in rows)
    leak = A.leakage_summary(rows)
    assert leak["all"]["n"] >= 4 and np.isfinite(leak["all"]["gap_to_best_mean"]) and isinstance(leak["single_anchor"], bool)
    # the restart states are anchor-installed in the first 3 coordinates of each block, so their Phi can differ from the anchor's
    # (the kept coordinates are the chain's own) but is finite for every chain that has a record at the exchange
    assert all(r["n_restart"] >= 1 for r in rows)
    # partition counts and the shape of the variance summary
    mask = os.path.join(root, "mask.npz")
    np.savez(mask, **{f"m{b}": (np.arange(M) >= 3) if b < N_SUB else np.zeros(1, bool) for b in range(N_SUB + 1)})
    mean, var, between = A.tail_moments(out, list(range(P)), 100, 5, N_SUB)
    s = A.variance_summary(var, mean, between, mask, N_SUB, M)
    assert s["overall"]["constrained"]["n"] == 12 and s["overall"]["unconstrained"]["n"] == 12 and len(s["per_block"]) == N_SUB
    # command line: tables are written under BRHO_ROOT
    os.environ["BRHO_ROOT"] = root
    try:
        sys.argv = ["x", "--exp-tag", "toy", "--out-dir", out, "--mask", mask, "--n-sub", str(N_SUB), "--chains", str(P), "--tail-start", "100",
                    "--stride", "5", "--n-iter", "260"]
        A.main()
    finally:
        os.environ.pop("BRHO_ROOT")
    tdir = os.path.join(root, "tables", "e6_glam_toy")
    for f in ("variance_by_index.csv", "variance_summary.json", "exchange_phi.csv", "leakage_summary.json"):
        assert os.path.exists(os.path.join(tdir, f)), f
    assert json.load(open(os.path.join(tdir, "leakage_summary.json")))["exchanges_in_chains"] == len(true_its)


def test_a_plain_pcn_run_has_no_exchanges():
    out = tempfile.mkdtemp()
    model = ToyModel(_beta(), n_sub=N_SUB, m=M, n_cells=60, n_times=3, seed=5, sigma_conc=0.05, sigma_dp=0.05)
    cfg = C.GlamConfig(n_iter=80, window1=0, window2=80, burnin_length=80, global_stage_length=20, local_stage_length=15, beta_g0=0.05, beta_l0=0.2,
                       save_every=25, log_every=0)
    ThreadWorld(2).run(lambda comm: run_glam(model, cfg, comm, out, fixed_steps=True, log=lambda *_: None))
    assert len(A.exchange_iterations(load_chain(out, 0, theta=False))) == 0
