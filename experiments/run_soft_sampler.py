"""
E5 - beta FREE with the soft curve prior (report.pdf, Remark 3) and the
4-move sampler (A: pCN theta, B: pCN w, C: ridge, D: beta-theta swap).
See brho/samplers.py (SoftCurveSampler) and brho/config.py (SoftSamplerConfig).

Decisions behind E5 (2026-09-28, recorded in LOG.md):
  * E4's hard constraint beta = beta_bar(rho) gave beta an interval of sd
    0.05 around a curve value 5% off the truth.  E5 samples log beta with
    prior N(log beta_bar(rho), s(rho)^2), s^2 = E3 scatter^2 + sigma_e^2.
  * Move D moves (beta, level-carrying theta) together, so beta mixes.
  * Main study: --truth-mode kle (truth drawn in the 57-mode KLE; stated in
    the paper).  sigma_rel unchanged (2e-3).  No posterior-informed proposal
    (it would have to be given to E1 too; kept to the group's pCN standard).

Same workflow as run_curve_sampler.py:
    python experiments/run_soft_sampler.py                 # setup + pool
    python experiments/run_soft_sampler.py --prior-test 1  # likelihood off
    python experiments/run_soft_sampler.py --setup-only --n-chains 64 [...]
    python experiments/run_soft_sampler.py --chain-id env  # each rank / task
    python experiments/run_soft_sampler.py --status
Requires results/e3_curve/curve_<field>.npz from the SAME study (same KLE,
same truth: K* comes from the truth).
Output: results/e5_soft/ (or results/e5_prior_test/)
"""
import _common  # noqa: F401  (first: pins threads before NumPy loads)
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from _common import parser, configs, configs_from_manifest, task_indices, C, kle_summary
from brho import io
from brho.problem import Problem
from brho.curve import BetaCurve
from brho.samplers import SoftCurveSampler, run_chunk
from run_curve_sampler import check_curve_matches

CFG = C.SoftSamplerConfig


def exp_name(prior_test, innovation="global"):
    return io.exp_soft(prior_test, innovation)


def chain_path(exp, cid):
    return os.path.join(io.results_dir(exp), f"chain{cid}.npz")


def run_chain(pc, scfg, exp, cid):
    prob = Problem(pc, frozen=io.results_dir(exp))
    curve = BetaCurve.load(os.path.join(io.results_dir(exp), "curve.npz"))
    sampler = SoftCurveSampler(prob, curve, scfg)
    t0 = time.time()
    it = 0
    while it < scfg.n_total:
        it = run_chunk(sampler, cid, chain_path(exp, cid))
        print(f"  chain {cid}: {it}/{scfg.n_total}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return cid, it


def _pool_job(args):
    return run_chain(*args)


def setup(pc, scfg, curve_field, curve_statistic, exp):
    out_dir = io.results_dir(exp)
    prob = Problem(pc, store=io.store_dir())
    check_curve_matches(prob)
    man3 = io.read_manifest(io.EXP_CURVE)
    if man3["configs"]["ProblemConfig"]["truth_mode"] != pc.truth_mode:
        raise SystemExit("E3 was run with truth_mode "
                         f"{man3['configs']['ProblemConfig']['truth_mode']}, E5 asks for "
                         f"{pc.truth_mode}: run E3 in this study first (same BRHO_ROOT)")
    src = os.path.join(io.results_dir(io.EXP_CURVE), f"curve_{curve_field}.npz")
    curve = BetaCurve.load(src, curve_statistic)
    for f in glob.glob(os.path.join(out_dir, "chain*.npz")):
        os.remove(f)
    curve.save(os.path.join(out_dir, "curve.npz"))
    prob.freeze(out_dir)
    prob.phys.warmup()
    smp = SoftCurveSampler(prob, curve, scfg)
    syn = prob.syn
    rt = syn.rho_true
    prior_at_truth = dict(
        rho=rt, m=smp.m(rt), s=smp.s(rt), beta_bar=float(np.exp(smp.m(rt))),
        z_truth=(np.log(syn.beta_true) - smp.m(rt)) / smp.s(rt))
    io.write_manifest(exp, C.to_dict(pc, scfg),
                      dict(r=prob.r, Kstar=syn.keff_obs,
                           kle_key=prob.kle_key(), kle=kle_summary(prob),
                           likelihood=prob.likelihood_summary(),
                           curve=dict(source=src, statistic=curve.statistic,
                                      rho_max=curve.rho_max),
                           soft_prior_at_truth=prior_at_truth,
                           constant_field_residual=smp.c_resid,
                           multiscale=(smp.ms.summary() if smp.ms is not None else None),
                           truth=syn.summary(), status="set up"))
    print(f"E5{' PRIOR TEST' if scfg.prior_test else ''} (beta free, soft curve "
          f"prior, moves A-D): {scfg.n_chains} chains x {scfg.n_total} it "
          f"(burn {scfg.n_burn}, thin {scfg.thin}); r={prob.r}; truth={pc.truth_mode}; "
          f"K*={syn.keff_obs:.3f}; curve={curve.statistic}; "
          f"theta innovation={scfg.theta_innovation} -> {exp}")
    if smp.ms is not None:
        ms = smp.ms.summary()
        print(f"  multiscale: {'x'.join([str(ms['blocks'])] * ms['dim'])} subdomains, local modes "
              f"{ms['r_k']} (p={ms['p']}), Sigma_ms eigenvalues "
              f"[{ms['sigma_ms_eig_min']:.3f}, {ms['sigma_ms_eig_max']:.3f}], "
              f"r_plus={ms['r_plus']}/{ms['r']}, whitening check {ms['operator_check']:.1e}")
    print(f"  soft prior at rho_true={rt}: beta_bar={prior_at_truth['beta_bar']:.3f}, "
          f"s={prior_at_truth['s']:.3f}; beta_true is {prior_at_truth['z_truth']:+.2f} s "
          f"from the curve; constant field outside the KLE span: "
          f"{100 * smp.c_resid:.1f}% (rms)")


def status(exp):
    _, scfg = configs_from_manifest(exp, CFG)
    done = 0
    for cid in range(scfg.n_chains):
        p = chain_path(exp, cid)
        it = int(np.load(p)["it"]) if os.path.exists(p) else 0
        done += it >= scfg.n_total
        print(f"chain {cid:3d}: {it:8d}/{scfg.n_total}")
    print(f"{done}/{scfg.n_chains} chains complete")
    return done == scfg.n_chains


def main():
    ap = parser(__doc__, CFG)
    ap.add_argument("--curve-field", default="kle", choices=["kle", "cholesky"])
    ap.add_argument("--curve-statistic", default=None)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--setup-only", action="store_true")
    mode.add_argument("--chain-id", help="int, or 'env' for the array index")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--warmup-only", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    prior_test = bool(args.prior_test) if args.prior_test is not None else False
    exp = exp_name(prior_test, args.theta_innovation or "global")

    if args.status:
        sys.exit(0 if status(exp) else 1)
    if args.warmup_only:
        pc, _ = configs_from_manifest(exp, CFG)
        Problem(pc, frozen=io.results_dir(exp)).phys.warmup()
        print("numba warm-up done")
        return
    if args.chain_id is not None:
        pc, scfg = configs_from_manifest(exp, CFG)
        for cid in task_indices(args.chain_id, scfg.n_chains):
            run_chain(pc, scfg, exp, cid)
        return

    pc, scfg = configs(args, CFG)
    have_run = os.path.exists(os.path.join(io.results_dir(exp), "manifest.json"))
    if args.setup_only or args.fresh or not have_run:
        setup(pc, scfg, args.curve_field, args.curve_statistic, exp)
    else:
        rec = configs_from_manifest(exp, CFG)
        if C.to_dict(*rec) != C.to_dict(pc, scfg):
            raise SystemExit(f"an {exp} run with different settings exists: rerun "
                             "with --fresh for a new run, or drop the flags to resume")
        print(f"resuming {exp} (use --fresh for a new run)")
    if args.setup_only:
        print(f"set up; now run --chain-id env over {scfg.n_chains} chains")
        return
    t0 = time.time()
    with Pool(min(scfg.n_workers, scfg.n_chains)) as pool:
        pool.map(_pool_job, [(pc, scfg, exp, c) for c in range(scfg.n_chains)])
    man = io.read_manifest(exp)
    man.update(status="complete", minutes=(time.time() - t0) / 60)
    io.save_json(os.path.join(io.results_dir(exp), "manifest.json"), man)
    print(f"done in {man['minutes']:.1f} min -> {io.results_dir(exp)}")


if __name__ == "__main__":
    main()
