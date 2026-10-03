"""Calibration of rho in the LINEARIZED hierarchical model (exact, no MCMC).
u = rho*theta, theta ~ N(0, I_r);  data z = J_u u + eps,  J_u = J / rho*_0
(J from lin.py at the study's truth, rho*_0 = 1.5);  log rho ~ N(0, 0.75^2).
Posterior of rho is 1-D: z ~ N(0, rho^2 J_u J_u^T + I/w)  (the likelihood
weight w of the study).  Noise in the simulated data: the study's actual
sigma_p ("as run": the relative likelihood assumes 1/sqrt(w) > sigma_p) or
exactly 1/sqrt(w) ("matched").  For each rho* on a grid: N truths, posterior
median, bias, 90% coverage.  Plus SBC with rho* drawn from its prior."""
import sys, json, numpy as np
lab = sys.argv[1] if len(sys.argv) > 1 else "2d_r208"
z = np.load(f"lin_{lab}.npz"); J, w = z["J"], float(z["w"])
kappa = {"2d_r57": 0.2018, "2d_r208": 0.04785}[lab]          # w * sigma_p^2 (lin.py)
U, s, _ = np.linalg.svd(J / 1.5, full_matrices=False)          # J_u singular values
r = len(s)
grid = np.exp(np.linspace(np.log(0.1), np.log(6.0), 1200))
logprior = -0.5 * (np.log(grid) / 0.75) ** 2                   # density in log rho
VAR = grid[:, None] ** 2 * s[None, :] ** 2 + 1.0 / w            # (grid, r)
rng = np.random.default_rng(0)

def post_summary(zc):                                         # zc: data in the U basis (r,)
    L = -0.5 * np.sum(zc ** 2 / VAR + np.log(VAR), 1) + logprior
    p = np.exp(L - L.max()); c = np.cumsum(p); c /= c[-1]
    q = lambda a: float(np.interp(a, c, grid))
    return q(0.5), q(0.05), q(0.95), c

def simulate(rho, noise_var):
    th = rng.standard_normal(r)
    return s * rho * th + np.sqrt(noise_var) * rng.standard_normal(r)

out = {}
for mode, nv in (("as run", kappa / w), ("matched", 1.0 / w)):
    rows = []
    for rho in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5):
        med, cov = [], []
        for _ in range(400):
            m, lo, hi, _ = post_summary(simulate(rho, nv))
            med.append(m); cov.append(lo <= rho <= hi)
        med = np.array(med)
        rows.append(dict(rho=rho, median_of_medians=round(float(np.median(med)), 3),
                         rel_bias=round(float(np.median(med) / rho - 1), 3),
                         coverage90=round(float(np.mean(cov)), 3)))
    ranks = []
    for _ in range(1000):                                        # SBC: rho from the prior
        rho = float(np.exp(0.75 * rng.standard_normal()))
        if rho > 5.5:
            continue
        _, _, _, c = post_summary(simulate(rho, nv))
        ranks.append(float(np.interp(rho, grid, c)))
    h = np.histogram(ranks, bins=10, range=(0, 1))[0]
    out[mode] = dict(by_rho=rows, sbc_rank_hist=h.tolist(),
                     sbc_cov90=round(float(np.mean((np.array(ranks) > 0.05) & (np.array(ranks) < 0.95))), 3))
print(lab, json.dumps(out, indent=1))
