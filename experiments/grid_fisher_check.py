"""
Fisher-information proxy at theta* for the nine configurations of the grid-refinement study
(docs/grid_arms_experiment.md, "Why sigma_abs").  For each (arm, grid):  J = d p_obs / d theta at theta*
(central differences, 20 columns) and F = w J'J with the arm's likelihood weight
    arms check, fixed:  w = 1 / (eps^2 ||p_obs||^2)      (relative-error likelihood)
    arm abs:            w = 1 / sigma_abs^2              (proper Gaussian, fixed per-observation noise)
Reported: trace F, logdet(I + F) (posterior-information proxy against the N(0, I) prior), number of
eigenvalues above 1 and above 100, and the largest eigenvalue (posterior sd in the stiffest direction ~ 1/sqrt).
    python experiments/grid_fisher_check.py     ->  tables/grid_arms/fisher_proxy.{md,json}
"""
import _common  # noqa: F401
import json
import os

import numpy as np

from brho import gridstudy as G

EPS = G.EPS
rows = []
sig_abs = G.sigma_abs_reference()
for n in G.GRIDS:
    S = G.GridSetup(n)
    th = G.theta_truth()
    T = S.kle.T
    for arm in G.ARMS:
        idx = G.obs_index(S, arm)
        f = lambda t: S.phys.solve(np.exp(T @ t))[idx]
        p0 = f(th)
        h = 1e-5
        J = np.empty((len(idx), S.r))
        for j in range(S.r):
            e = np.zeros(S.r); e[j] = h
            J[:, j] = (f(th + e) - f(th - e)) / (2 * h)
        w = 1.0 / sig_abs ** 2 if arm == "abs" else 1.0 / (EPS ** 2 * np.sum(p0 ** 2))
        ev = np.linalg.eigvalsh(w * J.T @ J)
        rows.append(dict(arm=arm, n=n, n_obs=int(len(idx)), trace=float(ev.sum()), logdet=float(np.log1p(ev).sum()),
                         n_gt1=int((ev > 1).sum()), n_gt100=int((ev > 100).sum()), max_eig=float(ev.max())))
os.makedirs("tables/grid_arms", exist_ok=True)
json.dump(dict(sigma_abs=sig_abs, eps=EPS, rows=rows), open("tables/grid_arms/fisher_proxy.json", "w"), indent=1)
base = {r["arm"]: r for r in rows if r["n"] == 16}
with open("tables/grid_arms/fisher_proxy.md", "w") as f:
    f.write(f"sigma_abs = {sig_abs:.6f}, eps = {EPS}\n\n| arm | grid | n obs | trace F | trace / trace(16) | logdet(I+F) | #eig>1 | #eig>100 | max eig |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['arm']} | {r['n']} | {r['n_obs']} | {r['trace']:.4g} | {r['trace']/base[r['arm']]['trace']:.2f} | "
                f"{r['logdet']:.2f} | {r['n_gt1']} | {r['n_gt100']} | {r['max_eig']:.3g} |\n")
print(open("tables/grid_arms/fisher_proxy.md").read())
