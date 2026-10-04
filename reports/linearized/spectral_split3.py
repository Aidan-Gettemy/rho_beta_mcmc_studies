"""Three scale blocks (KLE modes 1-20, 21-60, 61-208) vs global pCN, r = 208 surrogate."""
import sys, json, numpy as np
sys.argv = ["x"]
import sim
I = np.eye(sim.r)
g_tau = dict(tau_max=22868, tau_med=13841)          # global, seed 21 (spectral_split.py)
for edges, probs in (((20, 60), (0.2, 0.3, 0.5)), ((20, 60), (1/3, 1/3, 1/3)), ((10, 40, 100), (0.15, 0.25, 0.3, 0.3))):
    cuts = (0,) + edges + (sim.r,)
    Vs = [I[:, a:b] for a, b in zip(cuts[:-1], cuts[1:])]
    res = sim.run(Vs, probs=list(probs), seed=21)
    print(f"blocks {cuts} p {np.round(probs, 2).tolist()}", json.dumps({k: round(v) for k, v in res.items() if k.startswith("tau_")}),
          "steps", np.round(res["steps"], 4).tolist(),
          f"speed-up max {g_tau['tau_max']/res['tau_max']:.2f}, median {g_tau['tau_med']/res['tau_med']:.2f}", flush=True)
