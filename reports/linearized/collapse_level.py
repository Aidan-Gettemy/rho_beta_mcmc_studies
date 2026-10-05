"""How much of theta's stiffness is the overall pressure magnitude?
With a prescribed inflow flux, p(beta K) = p(K) / beta exactly, so log beta
can be integrated out of a theta move in closed form (no PDE solve).  In the
linearized model (beta free, flat in the magnitude direction) that removes the
data direction g = p_obs(truth) from the misfit: H_c = w J'(I - g g'/|g|^2)J.
Compare the pCN acceptance budget A and the simulated IACT (r = 208)."""
import sys, json, numpy as np
sys.argv = ["x"]
import sim
J, w, g = sim.J, sim.w, sim.p_true
P = np.eye(len(g)) - np.outer(g, g) / (g @ g)
Jc = P @ J
H0, b0, S0 = sim.H, sim.b, sim.Sig
Hc = w * Jc.T @ Jc
bc = w * Jc.T @ (P @ (sim.eps + J @ sim.th_t))
def budget(H, b):
    S = np.linalg.inv(np.eye(H.shape[0]) + H); m = S @ b
    G = H @ S @ H; gg = H @ m - b
    return float(np.trace(G) + gg @ gg)
A0, Ac = budget(H0, b0), budget(Hc, bc)
l0, lc = np.linalg.eigvalsh(H0)[::-1], np.linalg.eigvalsh(Hc)[::-1]
print(f"budget A: lb fixed {A0:.3g}, lb collapsed {Ac:.3g} -> pCN step x{np.sqrt(A0/Ac):.2f}")
print("largest H eigenvalues, lb fixed:    ", np.round(l0[:5], 1))
print("largest H eigenvalues, lb collapsed:", np.round(lc[:5], 1))
print("share of trace(H) in the magnitude direction:", round(1 - np.trace(Hc) / np.trace(H0), 3))
if "--sim" in sys.argv[1:] or True:
    g0 = sim.run([np.eye(sim.r)], seed=31)
    print("global pCN, lb fixed    ", json.dumps({k: round(v) for k, v in g0.items() if k.startswith("tau_")}), "step", np.round(g0["steps"], 4), flush=True)
    sim.H, sim.b = Hc, bc
    sim.Sig = np.linalg.inv(np.eye(sim.r) + Hc)
    ev, sim.PC = np.linalg.eigh(sim.Sig)
    gc = sim.run([np.eye(sim.r)], seed=31)
    print("global pCN, lb collapsed", json.dumps({k: round(v) for k, v in gc.items() if k.startswith("tau_")}), "step", np.round(gc["steps"], 4),
          f"speed-up max {g0['tau_max']/gc['tau_max']:.2f}, median {g0['tau_med']/gc['tau_med']:.2f}", flush=True)
