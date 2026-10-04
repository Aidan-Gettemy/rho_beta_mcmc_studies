"""Why spatial blocks cannot beat global pCN (r = 208 surrogate).

For pCN on subspace V with step s, the log-acceptance ratio has variance
~ s^2 A(V), A(V) = E|V'grad Phi|^2 = tr(V' H Sig H V) + |V'(H m - b)|^2
(posterior expectation).  A fixed acceptance rate => s_V^2 ~ c / A(V).
A prior-dominated direction inside V moves like AR(1) with coefficient
sqrt(1 - s^2): it needs ~ 1/(acc s_V^2) visits, i.e. ~ A(V) / (c acc p_V)
solves when V is visited with probability p_V.  A is additive over
orthogonal directions, so for blocks that cover R^r the best allocation
(p_V ~ A(V)) gives  sum_V A(V)  >=  A(R^r): no gain over global pCN.
"""
import sys, json, numpy as np
sys.argv = ["x"]
import sim
H, b, Sig = sim.H, sim.b, sim.Sig
m = Sig @ b
G = H @ Sig @ H; g = H @ m - b
def A(V): return float(np.trace(V.T @ G @ V) + np.sum((V.T @ g) ** 2))
Ag = A(np.eye(sim.r))
print(f"global: A = {Ag:.3g}")
for name, n in (("2x2", 2), ("3x3", 3)):
    Vs = sim.layout(n)
    a = np.array([A(V) for V in Vs])
    S = sum(V @ V.T for V in Vs)
    print(f"{name}: {len(Vs)} moves, sum A = {a.sum():.3g} = {a.sum()/Ag:.2f} x global "
          f"(coverage eig min/max {np.linalg.eigvalsh(S).min():.2f}/{np.linalg.eigvalsh(S).max():.2f}); "
          f"block A range {a.min():.3g}-{a.max():.3g}")
# a split that DOES separate stiff from soft: H eigenvectors (likelihood-informed subspace)
lam, U = np.linalg.eigh(H); U = U[:, ::-1]; lam = lam[::-1]
for k in (10, 20, 40):
    a1, a2 = A(U[:, :k]), A(U[:, k:])
    print(f"LIS split k={k}: A(informed) {a1:.3g}, A(rest) {a2:.3g}  (H eig at k: {lam[k]:.2g})")

# per-direction prediction: progress(u) = sum_k p_k s_k^2 |V_k'u|^2, s_k^2 = c / A_k;
# global: c / A_g.  Ratio > 1 = faster than global.  u = posterior principal
# components (what sim.iact measures); slow directions = prior-dominated ones.
def block_info(Vs):
    return np.array([float(np.sum(l / (1 + l))) for l in
                     (np.linalg.eigvalsh(V.T @ H @ V) for V in Vs)])
print("\npredicted speed vs global pCN over posterior PCs (median / 5th pct / min):")
for name, n in (("2x2", 2), ("3x3", 3)):
    Vs = sim.layout(n); a = np.array([A(V) for V in Vs])
    dims = np.array([V.shape[1] for V in Vs], float)
    rules = dict(uniform=np.ones(len(Vs)), info=block_info(Vs) + 0.05 * block_info(Vs).max(),
                 dim=dims, optimal_A=a)
    for rule, p in rules.items():
        p = p / p.sum()
        prog = sum(pk / ak * ((V.T @ sim.PC) ** 2).sum(0) for pk, ak, V in zip(p, a, Vs)) * Ag
        print(f"  {name} {rule:9s} {np.median(prog):.2f} / {np.quantile(prog, .05):.2f} / {prog.min():.2f}")

# IACT of a direction = variance-weighted MEAN of the IACTs of its parts
# (ACF of a sum of independent parts), so slow parts dominate: decompose u in
# the eigenbasis of the rate operator R = A_g sum_k (p_k / A_k) V_k V_k'.
print("\npredicted speed vs global (IACT averaged over parts): median / 5th pct / min")
for name, n in (("2x2", 2), ("3x3", 3)):
    Vs = sim.layout(n); a = np.array([A(V) for V in Vs])
    dims = np.array([V.shape[1] for V in Vs], float)
    rules = dict(uniform=np.ones(len(Vs)), info=block_info(Vs) + 0.05 * block_info(Vs).max(),
                 dim=dims, optimal_A=a)
    for rule, p in rules.items():
        p = p / p.sum()
        R = Ag * sum(pk / ak * V @ V.T for pk, ak, V in zip(p, a, Vs))
        rr, E = np.linalg.eigh(R)
        tau_rel = ((E.T @ sim.PC) ** 2 / rr[:, None]).sum(0)     # IACT / global IACT
        sp = 1 / tau_rel
        print(f"  {name} {rule:9s} {np.median(sp):.2f} / {np.quantile(sp, .05):.2f} / {sp.min():.2f}"
              f"   rate eig min {rr.min():.2f}")
