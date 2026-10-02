import os, sys, json, numpy as np
sys.argv = ["x"]
import sim
Vs = sim.layout(3); s = np.load("steps_3x3.npy")
p = 1 / s ** 2; p /= p.sum()                       # visit stiff blocks more often
res = sim.run(Vs, probs=p, s_fixed=s, adapt=False, seed=3)
print("3x3 weighted p~1/s^2", json.dumps({k: v for k, v in res.items() if k.startswith("tau_")}),
      "visit probs min/max", round(p.min(), 4), round(p.max(), 3), flush=True)
res = sim.run([np.eye(sim.r)], seed=11)
print("global seed 11", json.dumps({k: v for k, v in res.items() if k.startswith("tau_")}), flush=True)
