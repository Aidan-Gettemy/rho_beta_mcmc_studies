"""Distribution of the informed energy over truths, each with its OWN
Jacobian (2r+1 solves per truth), for the study's truth recipe ("kle":
standardized full-rank draw projected on the KLE) and for exact prior draws
("kle_raw": the raw draw projected, theta ~ N(0, I))."""
import os, sys, json, numpy as np
from lin import load, REPO
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "runs", "cluster_l005")
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 40
prob, _ = load(root)
kle, syn = prob.kle, prob.syn
b, r_ = syn.beta_true, syn.rho_true
w = prob._w
def energy(th, h=1e-4):
    f = lambda t: prob.potential(b * np.exp(r_ * kle.field(t)))[1]
    J = np.stack([(f(th + h * e) - f(th - h * e)) / (2 * h) for e in np.eye(kle.r)], 1)
    lam, V = np.linalg.eigh(w * J.T @ J)
    wt = (lam / (1 + lam)) ** 2; wt /= wt.sum()
    return float(wt @ (V.T @ th) ** 2)
out = {"kle": [], "kle_raw": []}
for s in range(3000, 3000 + n_seeds):
    G = prob.full.sample(np.random.default_rng(s))
    out["kle"].append(energy(kle.project((G - G.mean()) / G.std())))
    out["kle_raw"].append(energy(kle.project(G)))
    print(s, round(out["kle"][-1], 3), round(out["kle_raw"][-1], 3), flush=True)
for k, v in out.items():
    v = np.array(v)
    print(k, f"mean {v.mean():.3f} sd {v.std():.3f} frac<1 {np.mean(v < 1):.2f}  "
             f"=> rho_hat = 1.5 sqrt(E): mean {np.mean(1.5 * np.sqrt(v)):.3f}")
