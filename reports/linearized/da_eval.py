"""How good can delayed acceptance with a linear surrogate be?  In the
posterior's typical set (after 3000 plain pCN steps from the truth), record
for 1500 proposals the true and surrogate misfit changes, for a surrogate
linearized at the truth and one re-linearized at the current chain state.
Efficiency = accepted moves per full solve (plain pCN: acceptance rate)."""
import os, json, numpy as np
from lin import load, REPO

R = os.path.join(REPO, "runs")
prob, _ = load(f"{R}/e5_kle_2026-09")
syn, kle = prob.syn, prob.kle
b, r_ = syn.beta_true, syn.rho_true
w = prob._w; d = syn.d_p
p_of = lambda t: prob.potential(b * np.exp(r_ * kle.field(t)))
z = np.load("lin_2d_r57.npz"); th0, J0 = z["th"], z["J"]
def jac(t, h=1e-4):
    return np.stack([(p_of(t + h * e)[1] - p_of(t - h * e)[1]) / (2 * h) for e in np.eye(kle.r)], 1)
rng = np.random.default_rng(1); s = 0.0124
x = th0.copy(); fx = p_of(x)[0]; acc = 0
for it in range(3000):                                    # plain pCN into the typical set
    y = np.sqrt(1 - s * s) * x + s * rng.standard_normal(x.size); fy = p_of(y)[0]
    if np.log(rng.random()) < fx - fy: x, fx = y, fy; acc += 1
print("burn-in acceptance", acc / 3000, "Phi", round(fx, 1))
px = p_of(x)[1]; Jx = jac(x)
lin = {"at truth": (p_of(th0)[1], J0, th0), "at chain state": (px, Jx, x.copy())}
c, *_ = np.linalg.lstsq(kle.T, np.ones(kle.T.shape[0]), rcond=None)   # T c ~ 1 (level direction)
cc = c @ c
x_lin = x.copy()
def phi_scaled(t):                       # level part exact by p(e^a K) = e^-a p(K), rest linear
    D = t - x_lin; dl = (c @ D) / cc
    return 0.5 * w * np.sum((d - np.exp(-r_ * dl) * (px + Jx @ (D - dl * c))) ** 2)
rows = []
for it in range(1500):
    y = np.sqrt(1 - s * s) * x + s * rng.standard_normal(x.size)
    fy = p_of(y)[0]; dtrue = fy - fx
    dl = {}
    for k, (p0, J, t0) in lin.items():
        ph = lambda t: 0.5 * w * np.sum((d - p0 - J @ (t - t0)) ** 2)
        dl[k] = ph(y) - ph(x)
    rows.append((dtrue, dl["at truth"], dl["at chain state"], phi_scaled(y) - phi_scaled(x)))
    if np.log(rng.random()) < -dtrue: x, fx = y, fy     # plain chain moves on
D = np.array(rows)
plain = np.mean(np.minimum(1, np.exp(-D[:, 0])))
out = dict(plain_accept_per_solve=round(float(plain), 3))
for j, k in ((1, "at truth"), (2, "at chain state"), (3, "at chain state, level exact")):
    out[k] = dict(corr=round(float(np.corrcoef(D[:, 0], D[:, j])[0, 1]), 3),
                  err_sd=round(float(np.std(D[:, 0] - D[:, j])), 2))
    for a in (0.5, 0.75, 1.0, 1.25):
        p1 = np.minimum(1, np.exp(-a * D[:, j])); p2 = np.minimum(1, np.exp(-(D[:, 0] - a * D[:, j])))
        out[k][f"scale {a}"] = dict(solves_per_proposal=round(float(p1.mean()), 3),
                                    accept_per_proposal=round(float((p1 * p2).mean()), 3),
                                    accept_per_solve=round(float((p1 * p2).mean() / p1.mean()), 3))
print(json.dumps(out, indent=1))
