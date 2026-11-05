"""
Grid-refinement study of global pCN on the Dirichlet Darcy slab (docs/grid_arms_experiment.md).

Same problem as the MSM study (Dirichlet 1 / 0 left / right, no flow top and bottom, sqexp kernel) with
correlation length ell = 0.2, a 20-mode KLE, and grids n = 16, 32, 64 on the same unit square.

KLE.  Built with the repo's weighted Nystrom builder ("nystrom_sep"): the eigenpairs come from a fixed
40-node-per-axis Gauss-Legendre quadrature, independent of the simulation grid, and are extended to the
cell centres of each grid by the Nystrom formula.  So T_16, T_32, T_64 are samples of the SAME 20
eigenfunctions and one theta* is the same continuous field on every grid.  The first 20 modes are kept
(the 20th and 21st eigenvalues are not tied, so no degenerate pair is split; checked in GridSetup).

Truth and data.  theta* = default_rng(2026).standard_normal(20) (the first 20 entries of the MSM
study's truth).  The forward model on each grid is applied to exp(T theta*), the observed pressures are
those at the arm's observation cells, and the noise vector xi is the next len(obs) standard normals of
the same stream (so arms with the same observation set share xi).

Arms (which cells observe the pressure, and the noise model):
  check  "fixed normalised sigma, increasing data": the checkerboard cells ((i + j) even) of the grid
         (128 / 512 / 2048 observations).  sigma_p = eps ||p_obs|| / sqrt(n_obs) and the relative-error
         likelihood Phi = ||d - O p||^2 / (2 eps^2 ||d||^2), eps = 0.003 -- exactly the MSM study.
  fixed  "fixed number of observations": 128 observations on every grid, in the grid cell closest to the
         centre of each cell of the 16 x 16 checkerboard.  Same noise model and likelihood as `check`.
         On the 32 and 64 grids the centre of a 16-cell is a vertex shared by four equally close cells;
         the tie is broken by the LOWER-LEFT of the central four (centre_cell_indices).
  abs    "fixed per-observation noise, increasing data": the checkerboard cells of the grid, additive
         noise of FIXED standard deviation sigma_abs on every observation, and the proper Gaussian
         likelihood Phi = sum (d - O p)^2 / (2 sigma_abs^2).  sigma_abs = eps ||p_obs^{16x16}|| = 0.0167
         (3.4% of the rms pressure of the 16 x 16 data): at 16 x 16 this is the SAME likelihood function as
         the other two arms; each refinement then adds information in proportion to the data (docs).

The sampler is the repo's global pCN (the `global` kind of msm_sampler.run_chain: likelihood-only
acceptance, multiplicative StepAdapter to 25% for the first M iterations, then frozen; chain seeds
SeedSequence([seed0, chain]); first draw rng.standard_normal(r)), here on theta in R^20.
"""
import numpy as np

from .kle import build_kle
from .physics2d import Darcy2D
from .samplers import StepAdapter

GRIDS = (16, 32, 64)
ARMS = ("check", "fixed", "abs")
ELL, R, EPS, SEED = 0.2, 20, 0.003, 2026
KLE_ENERGY = 0.9999          # only sets how many modes the builder computes; r_max keeps the first R
NQ = 16                      # Gauss-Legendre points per axis for the eigenproblem (256 nodes)


class GridKle:
    """The r-mode 2D Nystrom KLE on one grid: G = T theta.  Built with brho.kle.build_kle (method "nystrom",
    canonical basis): the eigenproblem is solved on an nq x nq Gauss-Legendre rule independent of the grid and
    only extended to the cell centres, so every grid gets samples of the same r eigenfunctions."""

    def __init__(self, phys, ell, r, nq):
        kle = build_kle(phys.points, phys.axes, "sqexp", dict(ell=ell, sigma2=1.0), energy=KLE_ENERGY,
                        method="nystrom", nq=nq, basis="canonical", r_max=r)
        if kle.r != r:
            raise ValueError("a cut at %d modes would split a degenerate group (the KLE has %d)" % (r, kle.r))
        self.T, self.r, self.lam, self.nq = kle.T, r, np.asarray(kle.lam), nq
        self.energy = float(self.lam.sum() / kle.total)

    def field(self, theta):
        return np.asarray(theta) @ self.T.T


class GridSetup:
    """Dirichlet permeameter on an n x n grid and its Nystrom KLE (r modes, correlation length ell)."""

    def __init__(self, n, ell=ELL, r=R, nq=NQ):
        self.n, self.ell, self.r, self.nq = n, ell, r, nq
        self.phys = Darcy2D(n, bc="dirichlet")
        self.kle = GridKle(self.phys, ell, r, nq)


def centre_cell_indices(n, base=16):
    """Cells of the n x n grid (index ix * n + iy) closest to the centre of each cell of the base x base
    checkerboard ((I + J) even), in the order of the base grid's cell index.  For n = base the cell itself;
    for n = b * base the centre of a base cell is a vertex of the central 2 x 2 fine cells (four equally
    close cells): the lower-left of those four is taken (documented tie-break)."""
    if n % base:
        raise ValueError("n must be a multiple of base")
    b = n // base
    off = b // 2 - 1 if b > 1 else 0
    ii, jj = np.meshgrid(np.arange(base), np.arange(base), indexing="ij")
    keep = ((ii + jj) % 2 == 0)
    return (((ii * b + off) * n + (jj * b + off))[keep]).ravel()


def obs_index(setup, arm):
    if arm in ("check", "abs"):
        return np.where(setup.phys.obs_mask)[0]
    if arm == "fixed":
        return centre_cell_indices(setup.n)
    raise ValueError(arm)


def theta_truth(r=R, seed=SEED):
    return np.random.default_rng(seed).standard_normal(r)


def sigma_abs_reference(eps=EPS, seed=SEED):
    """eps * ||p_obs|| of the noiseless 16 x 16 checkerboard data of theta* (the arm-3 noise level)."""
    S = GridSetup(16)
    p = S.phys.solve(np.exp(S.kle.field(theta_truth(S.r, seed))))
    return float(eps * np.linalg.norm(p[obs_index(S, "check")]))


class GridData:
    def __init__(self, setup, arm, eps=EPS, sigma_abs=None, seed=SEED):
        if arm not in ARMS:
            raise ValueError(arm)
        self.S, self.arm, self.eps, self.seed = setup, arm, eps, seed
        rng = np.random.default_rng(seed)
        self.theta_true = rng.standard_normal(setup.r)
        self.idx = obs_index(setup, arm)
        self.n_obs = len(self.idx)
        xi = rng.standard_normal(self.n_obs)
        p_true = setup.phys.solve(np.exp(setup.kle.field(self.theta_true)))
        self.p_true = p_true
        self.obs_clean = p_true[self.idx]
        pn = float(np.linalg.norm(self.obs_clean))
        if arm == "abs":
            self.sigma_abs = float(sigma_abs if sigma_abs is not None else sigma_abs_reference(eps, seed))
            self.sigma_p = self.sigma_abs
            self.w = 1.0 / self.sigma_abs ** 2
            self.phi_expected = 0.5 * self.n_obs
        else:
            self.sigma_abs = None
            self.sigma_p = eps * pn / np.sqrt(self.n_obs)
            self.d = None
            self.w = None
            self.phi_expected = 0.5
        self.d = self.obs_clean + self.sigma_p * xi
        self.dnorm = float(np.linalg.norm(self.d))
        if arm != "abs":
            self.w = 1.0 / (eps ** 2 * self.dnorm ** 2)
        r2 = float(np.sum((self.d - self.obs_clean) ** 2))
        self.relerr_truth = np.sqrt(r2) / self.dnorm
        self.phi_truth = 0.5 * self.w * r2

    def potential(self, theta):
        """(Phi, relative error ||d - O p|| / ||d||) of theta."""
        p = self.S.phys.solve(np.exp(self.S.kle.field(theta)))
        r2 = float(np.sum((self.d - p[self.idx]) ** 2))
        return 0.5 * self.w * r2, np.sqrt(r2) / self.dnorm

    def summary(self):
        return dict(arm=self.arm, n=self.S.n, ell=self.S.ell, r=self.S.r, kle_energy=self.S.kle.energy,
                    n_obs=self.n_obs, eps=self.eps, seed=self.seed, sigma_p=float(self.sigma_p),
                    sigma_abs=self.sigma_abs, likelihood_weight=float(self.w), data_norm=self.dnorm,
                    relerr_at_truth=float(self.relerr_truth), phi_at_truth=float(self.phi_truth),
                    phi_expected=float(self.phi_expected))


def _pcn(x, s, rng):
    return np.sqrt(1.0 - s * s) * x + s * rng.standard_normal(np.shape(x))


def run_chain(setup, data, seed0, chain, n_iter, n_adapt, s0=0.2, save_every=50, target=0.25, s_min=1e-4):
    """One global-pCN chain on theta in R^r.  Returns relerr / acc / phi (n_iter,), move (zeros),
    theta and s_hist at iterations 0, save_every, ... (the state before the step)."""
    r = setup.r
    rng = np.random.default_rng(np.random.SeedSequence([seed0, chain]))
    ad = StepAdapter([s0], targets=[target], every=100, s_min=s_min)
    theta = rng.standard_normal(r)
    phi, rel = data.potential(theta)
    n_save = n_iter // save_every
    out = dict(relerr=np.empty(n_iter, np.float32), acc=np.zeros(n_iter, np.uint8),
               move=np.zeros(n_iter, np.uint8), phi=np.empty(n_iter, np.float32),
               theta=np.empty((n_save, r), np.float32), s_hist=np.empty((n_save, 1), np.float32))
    for it in range(n_iter):
        if it % save_every == 0:
            j = it // save_every
            out["theta"][j], out["s_hist"][j] = theta, ad.s
        th_new = _pcn(theta, ad.s[0], rng)
        phi_new, rel_new = data.potential(th_new)
        accepted = np.log(rng.random()) < phi - phi_new
        if accepted:
            theta, phi, rel = th_new, phi_new, rel_new
        ad.record(0, accepted)
        ad.tick(it, n_adapt)
        out["relerr"][it], out["acc"][it], out["phi"][it] = rel, accepted, phi
    return out
