"""Does visiting subdomains by importance (instead of uniformly) speed up
block pCN?  Gaussian surrogate at r = 208 (sim.py), IACT per theta-solve
(posterior principal components), 300k theta-solves after a 60k burn-in in
which steps (and, for 'adaptive', the visit probabilities) adapt; both are
frozen afterwards, so the kernel is a fixed valid random scan.

Rules (p_k = probability of moving block k; coarse included as a block):
  uniform    p_k = 1/M
  info       p_k ~ d_eff_k + floor, d_eff_k = tr((V'HV)(I + V'HV)^-1):
             the data-informed dimension in block k (prior-controlled blocks
             visited rarely) -- the "importance" rule
  dim        p_k ~ dim_k
  inv_s      p_k ~ 1/s_k            (s_k = adapted step)
  inv_s2     p_k ~ 1/s_k^2          (stiff blocks most often)
  adaptive   p_k ~ 1 / ESJD_k per unit dimension, learned in burn-in
             (blocks that move little per visit get more visits)
"""
import sys, json, time, numpy as np
sys.argv = ["x"]
import sim

def block_info(Vs):
    out = []
    for V in Vs:
        A = V.T @ sim.H @ V
        lam = np.linalg.eigvalsh(A)
        out.append(float(np.sum(lam / (1 + lam))))
    return np.array(out)

def run_rule(name, Vs, rule, seed):
    M = len(Vs); dims = np.array([V.shape[1] for V in Vs], float)
    # pilot: uniform, adapt steps (burn only) -> steps and per-block ESJD
    pilot = sim.run(Vs, probs=np.full(M, 1 / M), solves=20_000, burn=60_000, seed=seed)
    s = pilot["steps"]
    if rule == "uniform":  p = np.ones(M)
    elif rule == "info":   d = block_info(Vs); p = d + 0.05 * d.max()
    elif rule == "dim":    p = dims
    elif rule == "inv_s":  p = 1 / s
    elif rule == "inv_s2": p = 1 / s ** 2
    elif rule == "adaptive":
        # expected squared jump per visit per dimension ~ acc * s^2 (pCN, small s);
        # the visits a block needs ~ dim / ESJD -> p ~ dim / (acc s^2) ~ dim / s^2 at 25% acc
        p = dims / s ** 2
    p = p / p.sum()
    res = sim.run(Vs, probs=p, s_fixed=s, adapt=False, seed=seed + 100)
    return dict(rule=rule, layout=name, tau_med=round(res["tau_med"]), tau_95=round(res["tau_95"]),
                tau_max=round(res["tau_max"]), p_min=round(float(p.min()), 4), p_max=round(float(p.max()), 3))

if __name__ == "__main__":
    t0 = time.time()
    g = sim.run([np.eye(sim.r)], seed=7)
    print(json.dumps(dict(rule="global pCN", tau_med=round(g["tau_med"]), tau_95=round(g["tau_95"]),
                          tau_max=round(g["tau_max"]))), f"{time.time()-t0:.0f}s", flush=True)
    for name, n in (("2x2", 2), ("3x3", 3)):
        Vs = sim.layout(n)
        for rule in ("uniform", "info", "dim", "inv_s", "inv_s2", "adaptive"):
            t0 = time.time()
            print(json.dumps(run_rule(name, Vs, rule, seed=5)), f"{time.time()-t0:.0f}s", flush=True)
