"""Frozen-direction check for the neighbour-conditioned (DDKLE-style) Gibbs sampler.
For each block k and conditioning set J, the conditional prior covariance of the block coefficients is the Schur
complement  S = G_kk - G_kJ G_JJ^+ G_Jk.  The fraction of the (marginal) prior variance that is left in each direction
is the eigenvalue v of G_kk^{-1/2} S G_kk^{-1/2} in [0,1]; a Gaussian Gibbs sweep then has autocorrelation ~ 1-v
in that direction (integrated autocorrelation time ~ 2/v)."""
import sys, numpy as np
sys.path.insert(0, "/home/claude/rho_beta_mcmc_studies")
from brho import msm_local as ML
from brho import kernels as KN

def sqrtm_inv(A, tol=1e-12):
    w, V = np.linalg.eigh(A); keep = w > tol * w.max()
    return (V[:, keep] / np.sqrt(w[keep])) @ V[:, keep].T

def neighbours(S, ms):
    P = np.asarray(S.points); h = 1.0 / S.n
    box = [(P[c].min(0), P[c].max(0)) for c in ms.block_cells]
    K = len(box); nb = []
    for a in range(K):
        nb.append([b for b in range(K) if b != a and
                   np.all(np.maximum(box[a][0] - box[b][1], box[b][0] - box[a][1]) < 1.5 * h)])
    return nb

def report(name, G, sl, nb, tag):
    """G: coefficient covariance (p x p); sl[k]: slice of block k; nb: adjacency."""
    vmins, vmed, nfrozen, ntot, iat = [], [], 0, 0, []
    allv = []
    for k in range(len(sl)):
        J = np.concatenate([np.arange(sl[b].start, sl[b].stop) for b in (nb[k] if tag == "nbr" else
                           [b for b in range(len(sl)) if b != k])])
        Gkk = G[sl[k], sl[k]]; GkJ = G[sl[k]][:, J]; GJJ = G[np.ix_(J, J)]
        Sc = Gkk - GkJ @ np.linalg.pinv(GJJ, rcond=1e-10, hermitian=True) @ GkJ.T
        W = sqrtm_inv(Gkk); v = np.linalg.eigvalsh(W @ Sc @ W); v = np.clip(v, 0, None)
        allv.append(v)
    v = np.concatenate(allv)
    print(f"  {name:34s} {tag:5s} blocks {len(sl):2d} coeffs {len(v):4d} | min v {v.min():.1e}  median {np.median(v):.2f} | "
          f"#v<1e-2 {int((v<1e-2).sum()):3d}  #v<1e-4 {int((v<1e-4).sum()):3d}  #v<1e-8 {int((v<1e-8).sum()):3d}")
    return allv

for grid, cfgs in (("primal", [dict(primal_local_dim=6), dict(primal_local_dim=10), dict(primal_local_dim=15)]),
                   ("dual", [dict(dual_energy=0.95), dict(dual_energy=0.99)])):
    for cfg in cfgs:
        S = ML.MSMSetup(**cfg); ms = getattr(S, grid)
        rk = list(ms.r_k); off = np.r_[0, np.cumsum(rk)]; sl = [slice(off[i], off[i + 1]) for i in range(len(rk))]
        nb = neighbours(S, ms)
        print(f"\n{grid} {cfg}  r_k={sorted(set(rk))} p={ms.p} global modes r={S.r}  neighbours/block: {sorted(set(map(len, nb)))}")
        # (A) the study's actual prior: theta ~ N(0, I_64), c_all = Pi theta
        GA = ms.Pi @ ms.Pi.T
        report("A exact 64-mode global prior", GA, sl, nb, "nbr")
        report("A exact 64-mode global prior", GA, sl, nb, "all")
        # (B) DDKLE target: I on the diagonal, K_pq on neighbours, zero elsewhere (full-kernel coupling)
        kp = dict(ell=S.ell, sigma2=1.0); C = KN.get("sqexp")["cov"](np.asarray(S.points), np.asarray(S.points), kp)
        GB = np.zeros((ms.p, ms.p))
        Pm = [ (ms.block_psi[k] / np.sqrt(ms.block_mu[k])) for k in range(len(rk))]   # cells x r_k  (P^T)
        for a in range(len(rk)):
            GB[sl[a], sl[a]] = np.eye(rk[a])
            for b in nb[a]:
                GB[sl[a], sl[b]] = Pm[a].T @ C[np.ix_(ms.block_cells[a], ms.block_cells[b])] @ Pm[b]
        wmin = np.linalg.eigvalsh(GB).min()
        print(f"  (G_target min eigenvalue {wmin:.2e}; negative => the DDKLE target is not a valid covariance)")
        report("B DDKLE target (neighbour K)", GB, sl, nb, "nbr")
