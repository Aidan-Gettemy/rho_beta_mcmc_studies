"""
Multiscale-sampling (MSM) of Ali, Al-Mamun, Pereira, Rahunanthan (J. Comput. Phys. 497, 2024) against global pCN,
on the grid-study problem (docs/paper_msm_experiment.md): Dirichlet Darcy slab, 16 x 16 cells, sqexp kernel with
ell = 0.2, the data of arm 1 of the grid study (checkerboard, relative-error likelihood, theta* of seed 2026).

Local KLE.  The unit square is split into 2 x 2 subdomains S_i (8 x 8 cells).  In every subdomain the kernel
restricted to the subdomain is diagonalised and the first 5 modes are kept, Y_i = sum_j sqrt(mu_j) psi_j c_ij, so
the stochastic dimension is p = 4 x 5 = 20 (= the global one).  Modes 5 and 6 of a block are an exactly tied pair
(isotropic kernel on a square): a cut at 5 keeps ONE direction of that pair.  The direction is fixed by the
canonical-basis rule of brho.kle (block-relative coordinates, so all four blocks keep the same function), which
makes it independent of the eigensolver.  The raw field is Y_raw = E c (zero-extended local modes), piecewise
and discontinuous across the interfaces.

Three fields from the same local coefficients c:
  raw     eta = E c                         (no projection; discontinuous)
  proj    eta = T B c, B = T^+ E            (L2 projection onto the global 20-mode KLE; continuous)
  smooth  the smoothing of the paper (below)

Smoothing (paper, p. 6, only described in words).  H_bar = 0.1 (= min(Lx/2, Ly/2), p. 8).  The band B_i of subdomain
i is the set of its cells within H_bar of an interior interface; N(x) = cells y with |y - x| <= H_bar (any
subdomain, centre included); for x in B_i the value is replaced by the PLAIN average of the field over N(x)
(mean preserving; variance factor about 0.89 for correlated cells, not rescaled):
        eta(x) <- (1 / |N(x)|) sum_{y in N(x)} V(y).
 * all-at-once:  V = E c (every subdomain new), all bands replaced simultaneously; eta is a linear function
   of c (Gaussian, singular covariance).
 * one subdomain i updated (the paper's Algorithm 1): V = current eta with the cells of S_i replaced by the new
   raw field of subdomain i; only the cells of band B_i are replaced by averages of V.  Cells of the other
   subdomains keep their stored values, so eta depends on the history of the chain (the "lag"); the state of the
   chain is (c, eta) and the likelihood is evaluated on eta.

Sampler.  pCN on groups of coordinates, acceptance on the likelihood only, one forward solve per iteration:
  scan "all"        one group with all 20 coordinates
  scan "coord"      20 groups of one coordinate, cyclic order (subdomain-major for the local coefficients)
  scan "subdomain"  4 groups of the 5 coefficients of a subdomain, cyclic
One step size per group, adapted multiplicatively to 25% acceptance for the first M iterations, then frozen.
"""
import numpy as np

from .gridstudy import GridSetup, GridData
from .kle import canonical_basis
from .samplers import StepAdapter

ELL, R, BLOCKS, LOCAL_DIM = 0.2, 20, 2, 5
HBAR = 0.1

# experiment name -> (field, scan, n_iter, adapt, s0)
EXPS = {
    "e0_global_all":   dict(field="global", scan="all",       n_iter=100000, adapt=10000, s0=0.2),
    "e1_global_one":   dict(field="global", scan="coord",     n_iter=150000, adapt=30000, s0=0.5),
    "e2_raw_all":      dict(field="raw",    scan="all",       n_iter=100000, adapt=10000, s0=0.2),
    "e3_proj_all":     dict(field="proj",   scan="all",       n_iter=100000, adapt=10000, s0=0.2),
    "e4_proj_one":     dict(field="proj",   scan="coord",     n_iter=150000, adapt=30000, s0=0.5),
    "e5_smooth_all":   dict(field="smooth", scan="all",       n_iter=100000, adapt=10000, s0=0.2),
    "e6_smooth_one":   dict(field="smooth", scan="coord",     n_iter=150000, adapt=30000, s0=0.5),
    "e7_smooth_sub":   dict(field="smooth", scan="subdomain", n_iter=150000, adapt=30000, s0=0.5),
}


class PaperSetup:
    def __init__(self, n=16, ell=ELL, r=R, blocks=BLOCKS, local_dim=LOCAL_DIM, hbar=HBAR):
        self.G = GridSetup(n, ell, r)
        self.n, self.ell, self.r = n, ell, r
        self.phys = self.G.phys
        self.points = np.asarray(self.phys.points)
        self.T = self.G.kle.T                                       # (N, r) global synthesis matrix
        self.N = self.T.shape[0]
        self.Tpinv = np.linalg.pinv(self.T)                          # theta-coordinates of a field (L2 projection)
        # ---- subdomains
        side = 1.0 / blocks
        ib = np.minimum((self.points[:, 0] / side).astype(int), blocks - 1)
        jb = np.minimum((self.points[:, 1] / side).astype(int), blocks - 1)
        self.sub_of_cell = ib * blocks + jb
        self.n_sub = blocks * blocks
        self.sub_cells = [np.flatnonzero(self.sub_of_cell == i) for i in range(self.n_sub)]
        # ---- local KLE (canonical basis inside the tied pair, block-relative coordinates)
        self.p = self.n_sub * local_dim
        self.local_dim = local_dim
        E = np.zeros((self.N, self.p))
        self.coef_sub = np.repeat(np.arange(self.n_sub), local_dim)
        self.mu = None
        for i, cells in enumerate(self.sub_cells):
            X = self.points[cells] - self.points[cells].min(0)
            C = np.exp(-0.5 * self._d2(X) / ell ** 2)
            lam, U = np.linalg.eigh(C)
            lam, U = np.clip(lam[::-1], 0, None), U[:, ::-1]
            # whole tie groups around the cut, canonical basis inside them, then keep local_dim modes
            top = local_dim
            while top < len(lam) and abs(lam[top] - lam[top - 1]) <= 1e-9 * lam[top - 1]:
                top += 1
            Uc = canonical_basis(lam[:top], U[:, :top], np.ones(len(cells)), X)
            self.tie_cut = top > local_dim
            if self.mu is None:
                self.mu = lam[:local_dim].copy()
            E[cells, i * local_dim:(i + 1) * local_dim] = Uc[:, :local_dim] * np.sqrt(lam[:local_dim])
        self.E = E
        self.Bproj = self.Tpinv @ E                                   # (r, p)
        self.Tproj = self.T @ self.Bproj                              # (N, p): eta = Tproj c
        # ---- smoothing
        D = np.sqrt(self._d2(self.points))
        dint = np.minimum(np.abs(self.points[:, 0] - 0.5), np.abs(self.points[:, 1] - 0.5))
        self.band = dint <= hbar + 1e-12
        self.band_idx = np.flatnonzero(self.band)
        W = (D <= hbar + 1e-12).astype(float)
        W /= W.sum(1, keepdims=True)
        self.A = W                                                    # averaging rows (all cells)
        self.A_band = W[self.band_idx]
        self.band_sub = [np.flatnonzero(self.band & (self.sub_of_cell == i)) for i in range(self.n_sub)]
        self.A_sub = [W[b] for b in self.band_sub]
        self.hbar = hbar

    @staticmethod
    def _d2(X):
        return ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)

    # ------------------------------------------------------------ fields
    def smooth_all(self, Y):
        """All bands replaced simultaneously by plain averages of the field Y (N,) or (..., N)."""
        Y = np.asarray(Y)
        out = Y.copy()
        out[..., self.band_idx] = Y @ self.A_band.T
        return out

    def smooth_subdomain(self, eta, i, Yi_new):
        """History update: the cells of S_i get the new raw field, its band is averaged from V (new raw values
        of S_i, stored values of the other subdomains)."""
        V = eta.copy()
        cells = self.sub_cells[i]
        V[cells] = Yi_new[cells]
        out = V.copy()
        out[self.band_sub[i]] = self.A_sub[i] @ V
        return out

    def field(self, kind, c):
        """eta for the stateless kinds ('global', 'raw', 'proj', 'smooth' all-at-once); c (..., dim)."""
        c = np.asarray(c)
        if kind == "global":
            return c @ self.T.T
        if kind == "raw":
            return c @ self.E.T
        if kind == "proj":
            return c @ self.Tproj.T
        if kind == "smooth":
            return self.smooth_all(c @ self.E.T)
        raise ValueError(kind)

    def coords(self, eta):
        """Common coordinates: theta-coordinates (L2 projection onto the global KLE) of a field."""
        return np.asarray(eta) @ self.Tpinv.T


def make_groups(scan, setup, d):
    if scan == "all":
        return [np.arange(d)]
    if scan == "coord":
        return [np.array([j]) for j in range(d)]
    if scan == "subdomain":
        return [np.flatnonzero(setup.coef_sub == i) for i in range(setup.n_sub)]
    raise ValueError(scan)


class PaperData:
    """The data of arm 1 of the grid study (checkerboard, relative-error likelihood), evaluated on a log field."""

    def __init__(self, setup, eps=0.003, seed=2026):
        self.g = GridData(setup.G, "check", eps, None, seed)
        self.S = setup
        self.theta_true = self.g.theta_true
        self.relerr_truth, self.phi_truth = self.g.relerr_truth, self.g.phi_truth
        self.n_obs = self.g.n_obs

    def potential(self, eta):
        p = self.S.phys.solve(np.exp(eta))
        r2 = float(np.sum((self.g.d - p[self.g.idx]) ** 2))
        return 0.5 * self.g.w * r2, np.sqrt(r2) / self.g.dnorm

    def summary(self):
        return self.g.summary()


def _pcn(x, s, rng):
    return np.sqrt(1.0 - s * s) * x + s * rng.standard_normal(np.shape(x))


def run_chain(setup, data, exp, seed0, chain, n_iter=None, n_adapt=None, s0=None, save_every=50, field_every=500,
              target=0.25, s_min=1e-4, prior=False):
    """One chain of experiment `exp` (a key of EXPS).  prior=True: likelihood off (every proposal accepted).
    Returns relerr / acc / move / phi (n_iter,), c and z (n_save, 20), s_hist (n_save, n_groups) and eta_thin
    ((n_iter // field_every, N), the field before iteration k * field_every)."""
    spec = EXPS[exp]
    n_iter = spec["n_iter"] if n_iter is None else n_iter
    n_adapt = spec["adapt"] if n_adapt is None else n_adapt
    s0 = spec["s0"] if s0 is None else s0
    kind, scan = spec["field"], spec["scan"]
    history = kind == "smooth" and scan != "all"
    d = setup.r if kind == "global" else setup.p
    groups = make_groups(scan, setup, d)
    ng = len(groups)
    every = max(100, 50 * ng)
    rng = np.random.default_rng(np.random.SeedSequence([seed0, chain]))
    ad = StepAdapter([s0] * ng, targets=[target] * ng, every=every, s_min=s_min)
    c = rng.standard_normal(d)
    eta = setup.smooth_all(setup.E @ c) if kind == "smooth" else setup.field(kind, c)
    phi, rel = (0.0, 0.0) if prior else data.potential(eta)
    n_save, n_thin = n_iter // save_every, n_iter // field_every
    out = dict(relerr=np.empty(n_iter, np.float32), acc=np.zeros(n_iter, np.uint8), move=np.zeros(n_iter, np.uint8),
               phi=np.empty(n_iter, np.float32), c=np.empty((n_save, d), np.float32),
               z=np.empty((n_save, setup.r), np.float32), s_hist=np.empty((n_save, ng), np.float32),
               eta_thin=np.empty((n_thin, setup.N), np.float32))
    for it in range(n_iter):
        if it % save_every == 0:
            j = it // save_every
            out["c"][j], out["s_hist"][j] = c, ad.s
            out["z"][j] = c if kind == "global" else setup.coords(eta)
        if it % field_every == 0:
            out["eta_thin"][it // field_every] = eta
        g = it % ng
        idx = groups[g]
        c_new = c.copy()
        c_new[idx] = _pcn(c[idx], ad.s[g], rng)
        if history:
            i = int(setup.coef_sub[idx[0]])
            sel = np.flatnonzero(setup.coef_sub == i)
            Yi = np.zeros(setup.N)
            Yi[setup.sub_cells[i]] = setup.E[np.ix_(setup.sub_cells[i], sel)] @ c_new[sel]
            eta_new = setup.smooth_subdomain(eta, i, Yi)
        else:
            eta_new = setup.field(kind, c_new)
        if prior:
            phi_new, rel_new, accepted = 0.0, 0.0, True
        else:
            phi_new, rel_new = data.potential(eta_new)
            accepted = np.log(rng.random()) < phi - phi_new
        if accepted:
            c, eta, phi, rel = c_new, eta_new, phi_new, rel_new
        ad.record(g, accepted)
        ad.tick(it, n_adapt)
        out["relerr"][it], out["acc"][it], out["phi"][it] = rel, accepted, phi
    return out
