"""First eigenfunctions on each grid and the reference field of theta* (docs/grid_arms_experiment.md)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import brho.gridstudy as G

out = sys.argv[1] if len(sys.argv) > 1 else "tables/grid_arms"
os.makedirs(out, exist_ok=True)
th = G.theta_truth()
S = {n: G.GridSetup(n) for n in G.GRIDS}
img = lambda v, n: v.reshape(n, n).T          # cell index ix * n + iy -> x horizontal, y vertical
ext = (0, 1, 0, 1)

fig, ax = plt.subplots(3, 5, figsize=(12.5, 7.8), constrained_layout=True)
for i, n in enumerate(G.GRIDS):
    for k in range(5):
        phi = S[n].kle.T[:, k] / np.sqrt(S[n].kle.lam[k])
        m = np.abs(S[16].kle.T[:, k] / np.sqrt(S[16].kle.lam[k])).max()
        a = ax[i, k]
        im = a.imshow(img(phi, n), origin="lower", extent=ext, cmap="RdBu_r", vmin=-m, vmax=m)
        a.set_xticks([]); a.set_yticks([])
        if i == 0: a.set_title(f"mode {k + 1}\n$\\lambda$ = {S[16].kle.lam[k]:.4f}", fontsize=10)
        if k == 0: a.set_ylabel(f"{n}x{n}", fontsize=11)
fig.suptitle("First five KLE eigenfunctions $\\phi_k$ (Nystrom extension of one 16x16-node Gauss-Legendre solve; same colour scale per column)")
fig.savefig(f"{out}/kle_eigenfunctions.png", dpi=150)

fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.6), constrained_layout=True)
lk = {n: S[n].kle.field(th) for n in G.GRIDS}
pr = {n: S[n].phys.solve(np.exp(lk[n])) for n in G.GRIDS}
for j, n in enumerate(G.GRIDS):
    im0 = ax[0, j].imshow(img(lk[n], n), origin="lower", extent=ext, cmap="viridis", vmin=-3, vmax=2)
    im1 = ax[1, j].imshow(img(pr[n], n), origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
    ax[0, j].set_title(f"{n}x{n}"); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
    ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
ax[0, 0].set_ylabel("log k = T $\\theta^*$"); ax[1, 0].set_ylabel("pressure p")
fig.colorbar(im0, ax=ax[0], shrink=0.85); fig.colorbar(im1, ax=ax[1], shrink=0.85)
fig.suptitle("Reference field of the same $\\theta^*$ (seed 2026, 20 modes) and its pressure on each grid")
fig.savefig(f"{out}/kle_reference_fields.png", dpi=150)

def block(v, n, m): return v.reshape(m, n // m, m, n // m).mean((1, 3)).ravel()
print("log k range", {n: (round(lk[n].min(), 2), round(lk[n].max(), 2)) for n in G.GRIDS})
for n in (32, 64):
    print(n, "vs 16: rel diff of block-mean log k %.4f, of block-mean p %.4f" % (
        np.linalg.norm(block(lk[n], n, 16) - lk[16]) / np.linalg.norm(lk[16]),
        np.linalg.norm(block(pr[n], n, 16) - pr[16]) / np.linalg.norm(pr[16])))
print("energy(20 modes) %.4f, lam[:5]" % S[16].kle.energy, S[16].kle.lam[:5])
