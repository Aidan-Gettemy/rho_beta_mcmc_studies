"""
One object bundling physics + KLE + synthetic data + likelihood, built from a
ProblemConfig.

Where the KLE basis and the reference field come from
-----------------------------------------------------
    Problem(pc, frozen=dir)   load kle.npz + truth.npz saved with a run (all
                              chain tasks and reports of that run)
    Problem(pc, store=root)   load them from the study's build-once store, or
                              build and save them there (setup steps)
    Problem(pc)               build in memory (tests, interactive use)
`Problem.freeze(dir)` writes the pair into a run folder.  The eigenvectors of
C1 inside degenerate eigenspaces depend on floating-point rounding (BLAS
build, CPU, thread count), so a basis is computed once and then only loaded.

Likelihood (pressure data; optional log-Keff term), ProblemConfig.likelihood:
    "relative" (default)  Phi(K) = ||d_p - O p(K)||^2 / (2 sigma_rel^2 ||d_p||^2)
                                 = relerr^2 / (2 sigma_rel^2)
    "absolute"            Phi(K) = ||d_p - O p(K)||^2 / (2 sigma_p^2)
                          [+ (log K* - log Keff(K))^2 / (2 sigma_e^2) in both]
The relative form uses the global normalized L2 norm (never per-sensor ratios:
the pressure vanishes at the outflow boundary).  It is the absolute form with
per-sensor standard deviation sigma_rel ||d_p||, which grows like sqrt(n_obs):
adding sensors or refining the grid does not rescale the information, so the
model error (e.g. KLE truncation, correlated between neighbouring sensors) is
not counted as independent evidence n_obs times.  sigma_rel = eps / sqrt(n_eff)
(config.sigma_rel_from): Phi at the truth ~ (n_eff / 2) (noise_rel / eps)^2.
"""
import os
import numpy as np
from .physics2d import Darcy2D
from .fv import Permeameter
from .kle import KLE, FullRankGaussian, build_kle, resolve_method
from .truth import make_synthetic, Synthetic
from . import store as ST

KEY_VERSION = 1


def kernel_params(pc):
    return dict(ell=pc.ell, sigma2=1.0)


def _grid(pc):
    return [pc.n] * getattr(pc, "dim", 2)      # [n, n] in 2D: keys of existing stores unchanged


def kle_key(pc, N):
    method = resolve_method(pc.kle_method, N, pc.kernel)
    key = dict(v=KEY_VERSION, grid=_grid(pc), kernel=pc.kernel, ell=pc.ell,
               energy=pc.energy, method=method)
    if method.startswith("nystrom"):
        key.update(nq=pc.kle_nq, quadrature=pc.kle_quadrature)
    return key


def truth_key(pc, N=None):
    key = dict(v=KEY_VERSION, grid=_grid(pc), kernel=pc.kernel, ell=pc.ell,
               q_in=pc.q_in, beta_true=pc.beta_true, rho_true=pc.rho_true,
               truth_mode=pc.truth_mode, truth_seed=pc.truth_seed,
               noise_rel=pc.noise_rel, sigma_e=pc.sigma_e)
    if pc.truth_mode in ("kle", "kle_raw"):   # the truth depends on the KLE basis
        key["kle"] = kle_key(pc, N if N is not None else pc.n * pc.n)
    return key


class Problem:
    MIN_CELLS_PER_ELL = 3

    def __init__(self, pc, frozen=None, store=None, log=print):
        self.pc = pc
        if pc.n * pc.ell < self.MIN_CELLS_PER_ELL:
            log(f"WARNING: {'x'.join(map(str, _grid(pc)))} grid resolves ell={pc.ell} with only "
                f"{pc.n * pc.ell:.1f} cells per correlation length (< "
                f"{self.MIN_CELLS_PER_ELL}); use --n >= "
                f"{int(np.ceil(self.MIN_CELLS_PER_ELL / pc.ell))}")
        dim = getattr(pc, "dim", 2)
        if dim == 2:
            self.phys = Darcy2D(pc.n, pc.q_in, solver=pc.solver, backend=pc.backend)
        elif dim == 3:
            self.phys = Permeameter((pc.n,) * 3, q_in=pc.q_in, solver=pc.solver,
                                    backend=pc.backend)
            self.phys.n = pc.n
        else:
            raise ValueError(f"dim must be 2 or 3, not {dim}")
        self.full = FullRankGaussian(self.phys.points, self.phys.axes,
                                     pc.kernel, kernel_params(pc))
        if frozen:
            kp = os.path.join(frozen, "kle.npz")
            if not os.path.exists(kp):
                raise FileNotFoundError(f"{kp} missing: run the setup step")
            self.kle = KLE.load(kp)
            self.syn = Synthetic.load(os.path.join(frozen, "truth.npz"))
            if self.kle.N != self.phys.N:
                raise ValueError(f"frozen KLE in {frozen} does not match the grid")
            self.frozen = True
        else:
            build_k = lambda: build_kle(self.phys.points, self.phys.axes,
                                        pc.kernel, kernel_params(pc), pc.energy,
                                        pc.kle_method, pc.kle_nq, pc.kle_quadrature)
            build_t = lambda: make_synthetic(self.phys, self.full, pc, kle=self.kle)
            if store:
                self.kle, _ = ST.get_or_build(store, "kle", kle_key(pc, self.phys.N),
                                              build_k, lambda o, p: o.save(p),
                                              KLE.load, log=log)
                self.syn, _ = ST.get_or_build(store, "truth", truth_key(pc, self.phys.N), build_t,
                                              lambda o, p: o.save(p), Synthetic.load,
                                              log=log)
            else:
                self.kle = build_k()
                self.syn = build_t()
            self.frozen = False
        self.r = self.kle.r
        # weight w of the squared pressure residual: Phi = w ||d - G||^2 / 2
        if pc.likelihood == "relative":
            from .config import resolve_sigma_rel
            self.sigma_rel = resolve_sigma_rel(pc)
            self._w = 1.0 / (self.sigma_rel ** 2 * float(self.syn.d_p @ self.syn.d_p))
        elif pc.likelihood == "absolute":
            self.sigma_rel = None
            self._w = 1.0 / self.syn.sigma_p ** 2
        else:
            raise ValueError(f"likelihood must be relative or absolute, not {pc.likelihood}")

    def kle_key(self):
        return kle_key(self.pc, self.phys.N)

    def freeze(self, directory):
        self.kle.save(os.path.join(directory, "kle.npz"))
        self.syn.save(os.path.join(directory, "truth.npz"))

    def potential(self, K, use_keff=False):
        """Returns (Phi, p_obs, Keff) for a permeability field K (N,)."""
        p, pobs, keff = self.phys.forward(K)
        res = self.syn.d_p - pobs
        phi = 0.5 * self._w * (res @ res)
        if use_keff:
            phi += 0.5 * ((np.log(self.syn.keff_obs) - np.log(keff))
                          / self.syn.sigma_e) ** 2
        return float(phi), pobs, float(keff)

    def noise_floor(self, use_keff=False):
        """Expected Phi at the truth: E||eta||^2 = n_obs sigma_p^2, weighted.
        Absolute: n_obs / 2.  Relative: (n_obs sigma_p^2 / ||d||^2) / (2 sigma_rel^2)
        ~ (n_eff / 2) (noise_rel / eps)^2, nearly grid-independent."""
        if self.pc.likelihood == "absolute":
            return (self.syn.n_obs + int(use_keff)) / 2.0
        return 0.5 * self._w * self.syn.n_obs * self.syn.sigma_p ** 2 + 0.5 * int(use_keff)

    def likelihood_summary(self):
        s = dict(likelihood=self.pc.likelihood, noise_floor=self.noise_floor(),
                 data_norm=float(np.linalg.norm(self.syn.d_p)))
        if self.sigma_rel is not None:
            s.update(sigma_rel=self.sigma_rel, eps_at_n_eff=dict(
                n_eff=self.pc.rel_n_eff, eps=self.sigma_rel * self.pc.rel_n_eff ** 0.5),
                weight_vs_absolute=self._w * self.syn.sigma_p ** 2)
        return s
