"""Does a split of theta by KLE index (scale) separate stiff from soft
directions, as the likelihood-informed split does, without any Jacobian?
Acceptance budget A(V) = E|V' grad Phi|^2 (why_blocks.py); r = 208 surrogate."""
import sys, numpy as np
sys.argv = ["x"]
import sim
H, b, Sig = sim.H, sim.b, sim.Sig
m = Sig @ b; G = H @ Sig @ H; g = H @ m - b
A = lambda V: float(np.trace(V.T @ G @ V) + np.sum((V.T @ g) ** 2))
I = np.eye(sim.r); Ag = A(I)
lam = np.linalg.eigvalsh(H)[::-1]
print(f"r = {sim.r}, global A = {Ag:.3g}; H eigenvalues > 1: {np.sum(lam > 1)}")
dH = np.diag(H)
print("informativeness (diag H) by KLE index, means over bands:",
      [f"{lo+1}-{hi}: {dH[lo:hi].mean():.3g}" for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 150), (150, 208))])
for k in (20, 40, 60, 100):
    a1, a2 = A(I[:, :k]), A(I[:, k:])
    # soft block: its step can grow like sqrt(Ag / A_soft) relative to the global step
    print(f"KLE split at {k:3d}: A(modes 1-{k}) = {a1:.3g}, A(modes {k+1}-{sim.r}) = {a2:.3g} "
          f"({100*a2/Ag:.1f}% of the budget for {sim.r-k} modes; soft step x{np.sqrt(Ag/a2):.1f})")

# Simulated: IACT per theta-solve (posterior PCs), global pCN vs a two-block
# scale split (each block its own adapted step), random scan, frozen after burn-in.
if __name__ == "__main__":
    import json
    out = {}
    g = sim.run([I], seed=21)
    out["global"] = g
    print("global      ", json.dumps({k: round(v) for k, v in g.items() if k.startswith("tau_")}), "step", np.round(g["steps"], 4), flush=True)
    for k, p_soft in ((60, 0.5), (100, 0.5), (100, 0.75)):
        res = sim.run([I[:, :k], I[:, k:]], probs=[1 - p_soft, p_soft], seed=21)
        print(f"split {k:3d} p_soft {p_soft}", json.dumps({kk: round(v) for kk, v in res.items() if kk.startswith("tau_")}),
              "steps", np.round(res["steps"], 4), f"speed-up max {g['tau_max']/res['tau_max']:.2f}, median {g['tau_med']/res['tau_med']:.2f}", flush=True)
