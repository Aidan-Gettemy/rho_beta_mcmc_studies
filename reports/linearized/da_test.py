"""Delayed acceptance with the linearized surrogate (Christen & Fox 2005).
Stage 1: accept with the surrogate misfit ratio (no PDE solve).
Stage 2 (only if stage 1 accepts): one full solve, accept with
    min(1, exp[(Phi(x) - Phi(y)) - (Phi_lin(x) - Phi_lin(y))]).
The chain targets the exact posterior.  Here: global pCN on theta at the
truth's (beta, rho), step = E5's adapted theta step, chain started at the
truth.  Reports full solves per proposal and acceptance, against plain pCN."""
import os, sys, json, numpy as np
from lin import load, REPO

def run(root, lab, s, n=4000, seed=0):
    prob, _ = load(root)
    z = np.load(f"lin_{lab}.npz"); J, w, th0 = z["J"], float(z["w"]), z["th"]
    syn, kle = prob.syn, prob.kle
    b, r_ = syn.beta_true, syn.rho_true
    p0 = prob.potential(b * np.exp(r_ * kle.field(th0)))[1]
    d = syn.d_p
    phi_true = lambda t: prob.potential(b * np.exp(r_ * kle.field(t)))[0]
    phi_lin = lambda t: 0.5 * w * np.sum((d - p0 - J @ (t - th0)) ** 2)
    rng = np.random.default_rng(seed)
    x = th0.copy(); fx, lx = phi_true(x), phi_lin(x)
    solves = acc = st1 = 0; err = []
    for it in range(n):
        y = np.sqrt(1 - s * s) * x + s * rng.standard_normal(x.size)
        ly = phi_lin(y)
        if np.log(rng.random()) >= lx - ly:
            continue                                   # rejected by the surrogate: no solve
        st1 += 1
        fy = phi_true(y); solves += 1
        err.append((fy - fx) - (ly - lx))
        if np.log(rng.random()) < (fx - fy) - (lx - ly):
            x, fx, lx = y, fy, ly; acc += 1
    err = np.array(err)
    return dict(study=lab, step=s, proposals=n, stage1_pass=st1 / n, full_solves_per_proposal=solves / n,
                stage2_accept=acc / max(st1, 1), overall_accept=acc / n,
                surrogate_error_in_dPhi_sd=float(err.std()), surrogate_error_in_dPhi_mean=float(err.mean()))

if __name__ == "__main__":
    R = os.path.join(REPO, "runs")
    for root, lab, s in ((f"{R}/cluster_kle", "2d_r57", 0.0124), (f"{R}/cluster_l005", "2d_r208", 0.0245)):
        print(json.dumps(run(root, lab, s)), flush=True)
