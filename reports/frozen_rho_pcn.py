"""
Frozen-rho baseline: fix rho at the best-fit state of a curve-sampler run (the
kept sample with the lowest misfit, the same state as frozen_beta_pcn.py) and
sample (theta, log beta) with full PDE solves, alternating

    A  pCN on theta,                       prior N(0, I)
    B  pCN on z = (log beta - m*) / s*,    curve prior log beta ~ N(m*, s*^2),
                                           m* = log beta_bar(rho*), s*^2 = log_scatter^2 + sigma_e^2

each accepted with min(1, exp(Phi - Phi')), two solves per iteration.  Its target
is the posterior with the curve prior at rho*, whose log-beta marginal is the
level-split Gaussian N(mu, 1/P) of the paper (sd ~ 1/sqrt(P(rho*))); the point of
the run is how slowly separate theta and beta moves cross the level-swap orbit.

Chains start at theta* (the best-fit field) with log beta drawn from the curve
prior (overdispersed), so the PSRF of log beta shows how fast they agree.

    BRHO_ROOT=... python reports/frozen_rho_pcn.py --exp e4_sampler_ab --setup [--chains 64 --solves 1000000]
    BRHO_ROOT=... srun ... python reports/frozen_rho_pcn.py --exp e4_sampler_ab --chain-id env
    BRHO_ROOT=... python reports/frozen_rho_pcn.py --exp e4_sampler_ab --report

Files: results/<exp>/frozen_rho/{state.npz, settings.json, chainK.npz} (chains
checkpoint every --checkpoint iterations and resume), tables/<exp>/frozen_rho.{json,md},
figures/<exp>/fig_frozen_rho.png.  plot_beta_methods.py adds a row when
tables/<exp>/frozen_rho.json exists.
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import importlib.util
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from _common import have, load_problem, io, C
from brho.curve import BetaCurve
from brho import diagnostics as D
from brho.beta_upgrade import constant_coords, summarize

TARGETS = (0.25, 0.44)        # acceptance targets: theta block, 1D beta block


def _task_indices(value, n):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "exp_common", os.path.join(os.path.dirname(here), "experiments", "_common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.task_indices(value, n)


def run_dir(exp):
    d = os.path.join(io.results_dir(exp), "frozen_rho")
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ setup
def setup(exp, n_chains, solves, burn_frac, thin, checkpoint, seed):
    prob, man = load_problem(exp)
    d = io.results_dir(exp)
    scfg = C.from_dict(C.SamplerConfig, man["configs"]["SamplerConfig"])
    use_keff = bool(scfg.use_keff_term)
    curve = BetaCurve.load(os.path.join(d, "curve.npz"))
    syn, r = prob.syn, prob.r
    best = (np.inf, None, None)
    for p in io.chain_files(d):              # same order and rule as frozen_beta_pcn.py
        with np.load(p) as z:
            rho = z["kept_rho"].ravel()
            pobs = z["kept_pobs"].reshape(len(rho), -1)
            res = syn.d_p[None, :] - pobs
            phi = 0.5 * prob._w * np.einsum("ij,ij->i", res, res)
            if use_keff:
                phi += 0.5 * ((np.log(syn.keff_obs) - np.log(z["kept_keff"].ravel()))
                              / syn.sigma_e) ** 2
            phi[~np.isfinite(phi)] = np.inf
            j = int(np.argmin(phi))
            if phi[j] < best[0]:
                best = (float(phi[j]), float(rho[j]), z["kept_theta"].reshape(-1, r)[j].copy())
    phi_k, rho_s, theta_s = best
    m_s = float(np.log(curve.beta_bar(rho_s, syn.keff_obs)))
    s_s = float(np.sqrt(curve.log_scatter(rho_s) ** 2 + syn.sigma_e ** 2))
    phi_star = prob.potential(np.exp(m_s + rho_s * prob.kle.field(theta_s)), use_keff)[0]
    rd = run_dir(exp)
    np.savez(os.path.join(rd, "state.npz"), rho=rho_s, theta=theta_s, m=m_s, s=s_s,
             phi=phi_star, phi_kept=phi_k)
    n_it = solves // 2
    settings = dict(exp=exp, n_chains=n_chains, solves_per_chain=2 * n_it, iterations=n_it,
                    burn=int(burn_frac * n_it), thin=thin, checkpoint=checkpoint, seed=seed,
                    use_keff=use_keff, targets=TARGETS, rho_star=rho_s, m=m_s, s=s_s,
                    phi_star=phi_star)
    with open(os.path.join(rd, "settings.json"), "w") as f:
        json.dump(settings, f, indent=1)
    for p in io.chain_files(rd):
        os.remove(p)                            # a new setup is a new run
    prob.phys.warmup()
    print(f"{exp} frozen rho: rho* = {rho_s:.4f}, Phi* = {phi_star:.3f}; prior log beta "
          f"N({m_s:.4f}, {s_s:.4f}^2); {n_chains} chains x {n_it} it ({2 * n_it} solves), "
          f"burn {settings['burn']}, thin {thin}")


# ------------------------------------------------------------------ one chain
def run_chain(exp, cid):
    rd = run_dir(exp)
    st = json.load(open(os.path.join(rd, "settings.json")))
    z0 = np.load(os.path.join(rd, "state.npz"))
    prob, _ = load_problem(exp)
    rho, m, s = float(z0["rho"]), float(z0["m"]), float(z0["s"])
    kle, use_keff = prob.kle, st["use_keff"]
    n_it, burn, thin = st["iterations"], st["burn"], st["thin"]
    path = os.path.join(rd, f"chain{cid}.npz")

    def Phi(theta, lb):
        return prob.potential(np.exp(lb + rho * kle.field(theta)), use_keff)[0]

    if os.path.exists(path):                    # resume
        c = np.load(path, allow_pickle=False)
        it0 = int(c["it"])
        if it0 >= n_it:
            print(f"chain {cid}: complete"); return
        theta, z, phi = c["theta"].copy(), float(c["z"]), float(c["phi"])
        step = c["step"].copy(); acc = c["acc"].copy(); nacc = c["nacc"].copy()
        tr_lb, tr_phi = list(c["tr_lb"]), list(c["tr_phi"])
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(c["rng"]))
    else:
        it0 = 0
        rng = np.random.default_rng(np.random.SeedSequence([st["seed"], cid]))
        theta = z0["theta"].copy()
        z = float(rng.standard_normal())        # overdispersed start: curve prior
        phi = Phi(theta, m + s * z)
        step = np.array([0.012, 0.01])
        acc = np.zeros(2); nacc = np.zeros(2)  # window counts / post-burn totals
        tr_lb, tr_phi = [], []

    t0 = time.time()
    for it in range(it0, n_it):
        # A: theta
        th_p = np.sqrt(1 - step[0] ** 2) * theta + step[0] * rng.standard_normal(theta.shape)
        f = Phi(th_p, m + s * z)
        if np.log(rng.random()) < phi - f:
            theta, phi = th_p, f; acc[0] += 1
            if it >= burn: nacc[0] += 1
        # B: log beta (pCN on z)
        zp = np.sqrt(1 - step[1] ** 2) * z + step[1] * rng.standard_normal()
        f = Phi(theta, m + s * zp)
        if np.log(rng.random()) < phi - f:
            z, phi = zp, f; acc[1] += 1
            if it >= burn: nacc[1] += 1
        if it < burn and (it + 1) % 100 == 0:    # adapt, burn-in only
            for j in range(2):
                step[j] = np.clip(step[j] * (1.12 if acc[j] / 100 > TARGETS[j] else 1 / 1.12),
                                  1e-6, 0.99)
            acc[:] = 0
        if it % thin == 0:
            tr_lb.append(m + s * z); tr_phi.append(phi)
        if (it + 1) % st["checkpoint"] == 0 or it + 1 == n_it:
            tmp = path + ".tmp.npz"
            np.savez(tmp, it=it + 1, theta=theta, z=z, phi=phi, step=step, acc=acc, nacc=nacc,
                     tr_lb=np.array(tr_lb, np.float32), tr_phi=np.array(tr_phi, np.float32),
                     rng=json.dumps(rng.bit_generator.state))
            os.replace(tmp, path)
            print(f"chain {cid}: {it + 1}/{n_it} ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ------------------------------------------------------------------ report
def report(exp):
    rd = run_dir(exp)
    st = json.load(open(os.path.join(rd, "settings.json")))
    z0 = np.load(os.path.join(rd, "state.npz"))
    prob, _ = load_problem(exp)
    syn = prob.syn
    paths = io.chain_files(rd)
    cs = [np.load(p) for p in paths]
    L = min(len(c["tr_lb"]) for c in cs)
    lb = np.stack([c["tr_lb"][:L] for c in cs]).astype(float)       # (m, L)
    thin, burn = st["thin"], st["burn"]
    its = np.arange(L) * thin
    kb = int(np.searchsorted(its, burn))
    post = lb[:, kb:]
    rho, m, s = float(z0["rho"]), float(z0["m"]), float(z0["s"])
    c, _ = constant_coords(prob.kle.T)
    cc = float(c @ c)
    P = cc / rho ** 2 + 1 / s ** 2
    omega = m + rho * float(c @ z0["theta"]) / cc
    mu = (omega * cc / rho ** 2 + m / s ** 2) / P               # level-split target at theta*
    done = [int(c_["it"]) for c_ in cs]
    acc = np.mean([c_["nacc"] / max(1, int(c_["it"]) - burn) for c_ in cs], axis=0)
    # running diagnostics against forward solves (growing window from burn-in)
    pts = np.unique(np.linspace(kb + 20, L, 40).astype(int))
    run_psrf = np.array([D.psrf(lb[:, kb:k]) for k in pts])
    run_sd = np.array([lb[:, kb:k].std() for k in pts])
    within = np.array([lb[:, kb:k].std(1).mean() for k in pts])
    x_solves = its[pts - 1] * 2
    res = dict(
        exp=exp, n_chains=len(paths), iterations_done=min(done), settings=st,
        rho_star=rho, phi_star=float(z0["phi"]),
        beta=summarize(np.exp(post.ravel()), syn.beta_true),
        log_beta_sd=float(post.std()), within_chain_sd=float(post.std(1).mean()),
        spread_of_chain_means=float(post.mean(1).std()),
        psrf_log_beta=float(D.psrf(post)), ess_log_beta=float(D.ess_multi(post)),
        acceptance=dict(theta=float(acc[0]), beta=float(acc[1])),
        target_level_split=dict(mean_log_beta=float(mu), sd=float(1 / np.sqrt(P)),
                                note="N(mu, 1/P) at theta*'s offset; the frozen-rho target "
                                     "also averages over the data-allowed offsets"),
        pde_solves_total=int(sum(2 * d_ for d_ in done)),
        running=dict(solves=x_solves.tolist(), psrf=run_psrf.tolist(),
                     pooled_sd=run_sd.tolist(), within_sd=within.tolist()))
    td, fd = io.tables_dir(exp), io.figures_dir(exp)
    with open(os.path.join(td, "frozen_rho.json"), "w") as f:
        json.dump(res, f, indent=1)
    b = res["beta"]
    with open(os.path.join(td, "frozen_rho.md"), "w") as f:
        f.write(f"Frozen rho* = {rho:.4f} (lowest-Phi kept sample, Phi* = {res['phi_star']:.2f}); "
                f"{len(paths)} chains x {min(done)} it (2 solves each), burn {burn}\n\n"
                f"| quantity | value |\n|---|---|\n"
                f"| beta median [90%] | {b['median']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] |\n"
                f"| covers truth {syn.beta_true} | {b['covered']} |\n"
                f"| sd(log beta), pooled / within-chain | {res['log_beta_sd']:.4f} / {res['within_chain_sd']:.4f} |\n"
                f"| spread of chain means of log beta | {res['spread_of_chain_means']:.4f} |\n"
                f"| PSRF(log beta) | {res['psrf_log_beta']:.3f} |\n"
                f"| ESS(log beta), summed over chains | {res['ess_log_beta']:.0f} |\n"
                f"| acceptance theta / beta | {acc[0]:.2f} / {acc[1]:.2f} |\n"
                f"| level-split sd 1/sqrt(P(rho*)) | {1 / np.sqrt(P):.4f} |\n"
                f"| PDE solves, total | {res['pde_solves_total']:,} |\n")
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    a = ax[0]
    for i in range(min(8, len(paths))):
        a.plot(its * 2 / 1e6, lb[i], lw=0.6)
    a.axhline(np.log(syn.beta_true), color="k", ls="--", lw=1.2, label="truth")
    a.axhspan(mu - 1.645 / np.sqrt(P), mu + 1.645 / np.sqrt(P), color="#2a6f97", alpha=0.12,
              label="level-split 90% at θ*")
    a.axvline(burn * 2 / 1e6, color="0.4", ls=":", lw=0.8)
    a.set_xlabel("forward solves per chain (millions)"); a.set_ylabel("log β")
    a.set_title(f"frozen ρ*: log β traces, {min(8, len(paths))} of {len(paths)} chains",
                fontsize=10, loc="left")
    a.legend(fontsize=8)
    a = ax[1]
    a.plot(x_solves / 1e6, run_psrf, color="#b0563f", lw=2)
    a.axhline(1.2, color="k", ls="--", lw=1)
    a.set_yscale("log"); a.set_xlabel("forward solves per chain (millions)")
    a.set_title("PSRF of log β across chains", fontsize=10, loc="left")
    a = ax[2]
    a.plot(x_solves / 1e6, run_sd, color="#b0563f", lw=2, label="pooled sd")
    a.plot(x_solves / 1e6, within, color="#b0563f", lw=1, ls=":", label="mean within-chain sd")
    a.axhline(1 / np.sqrt(P), color="#2a6f97", lw=1.5, label="level-split sd 1/√P")
    a.axhline(prob.sigma_rel, color="0.5", lw=1, ls="--", label="σ_rel")
    a.set_yscale("log"); a.set_xlabel("forward solves per chain (millions)")
    a.set_title("sd of log β", fontsize=10, loc="left"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_frozen_rho.png"), dpi=150)
    print(open(os.path.join(td, "frozen_rho.md")).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="e4_sampler_ab")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--setup", action="store_true")
    mode.add_argument("--chain-id", help="int, or 'env' for the srun rank / array index")
    mode.add_argument("--report", action="store_true")
    ap.add_argument("--chains", type=int, default=64)
    ap.add_argument("--solves", type=int, default=1_000_000, help="per chain")
    ap.add_argument("--burn-frac", type=float, default=0.1)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--checkpoint", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if not have(a.exp):
        print(f"{a.exp}: no results"); return
    if a.setup:
        setup(a.exp, a.chains, a.solves, a.burn_frac, a.thin, a.checkpoint, a.seed)
    elif a.report:
        report(a.exp)
    else:
        n = json.load(open(os.path.join(run_dir(a.exp), "settings.json")))["n_chains"]
        for cid in _task_indices(a.chain_id, n):
            run_chain(a.exp, cid)


if __name__ == "__main__":
    main()
