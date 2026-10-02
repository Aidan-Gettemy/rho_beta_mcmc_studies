"""Screen truth seeds by their energy in the data-informed directions
(linearized at the current truth; same frozen KLE basis, same truth recipe)."""
import os, sys, json, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from lin import load
for lab, root in (("2d_r208", os.path.join(REPO, "runs") + "/cluster_l005"),
                  ("2d_r57", os.path.join(REPO, "runs") + "/e5_kle_2026-09")):
    prob, man = load(root); pc, kle = prob.pc, prob.kle
    z = np.load(f"lin_{lab}.npz"); lam, V = z["lam"], z["V"]
    w = (lam / (1 + lam)) ** 2; w /= w.sum()
    sims = (np.random.default_rng(0).standard_normal((100000, len(lam))) ** 2) @ w
    rows = []
    for s in list(range(2000, 2100)) + [pc.truth_seed]:
        G = prob.full.sample(np.random.default_rng(s))
        th = kle.project((G - G.mean()) / G.std())
        e = float(w @ (V.T @ th) ** 2)
        rows.append((s, e, float(100 * np.mean(sims <= e)), float(th @ th / kle.r)))
    cur = rows[-1]; rows = rows[:-1]
    best = sorted(rows, key=lambda t: abs(t[2] - 50))[:5]
    print(lab, "current seed", cur[0], f"E={cur[1]:.3f} pct={cur[2]:.1f}")
    for s, e, pct, n2 in best:
        print(f"   seed {s}: E={e:.3f}  pct={pct:.1f}  |theta|^2/r={n2:.2f}  pred rho={1.5*np.sqrt(e):.3f}")
    lows = sorted(rows, key=lambda t: t[2])[:2]; highs = sorted(rows, key=lambda t: -t[2])[:2]
    print("   atypical low :", [(s, round(e, 3), round(p, 1)) for s, e, p, _ in lows])
    print("   atypical high:", [(s, round(e, 3), round(p, 1)) for s, e, p, _ in highs])
