"""
Synthetic reference experiment (shared by E2, E3, E4).

Random-number order is identical to mcmc_study.py / hier_sampler.py
(seed 2026: Cholesky draw of G_in, then n_obs pressure-noise draws, then one
Keff-noise draw), so truth_mode="raw" with n=16 reproduces the July data exactly.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Synthetic:
    mode: str
    beta_true: float
    rho_true: float
    G_raw: np.ndarray          # Cholesky draw
    G_true: np.ndarray         # field actually used (raw or standardized)
    logK_true: np.ndarray
    K_true: np.ndarray
    p_true: np.ndarray
    obs_mask: np.ndarray
    d_p: np.ndarray
    sigma_p: float
    sigma_e: float
    keff_true: float
    keff_obs: float            # K*, the "measured" effective permeability
    level_true: float          # spatial mean of log K_true
    Gbar: float                # spatial mean of G_true
    G_std: float               # spatial std of G_true
    theta_true: np.ndarray = None   # truth_mode "kle" only: KLE coordinates

    @property
    def n_obs(self):
        return int(self.obs_mask.sum())

    def save(self, path):
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in asdict(self).items()
                                     if v is not None})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in z.files}
        for k in ("beta_true", "rho_true", "sigma_p", "sigma_e", "keff_true",
                  "keff_obs", "level_true", "Gbar", "G_std"):
            kw[k] = float(kw[k])
        kw["mode"] = str(kw["mode"])
        return cls(**kw)

    def summary(self):
        extra = {}
        if self.theta_true is not None:
            extra = dict(theta_norm2_over_r=float(self.theta_true @ self.theta_true
                                                  / len(self.theta_true)))
        return dict(**extra, mode=self.mode, beta_true=self.beta_true,
                    rho_true=self.rho_true, n_obs=self.n_obs,
                    sigma_p=self.sigma_p, sigma_e=self.sigma_e,
                    keff_true=self.keff_true, keff_obs=self.keff_obs,
                    level_true=self.level_true, Gbar=self.Gbar,
                    G_std=self.G_std,
                    noise_rel=float(self.sigma_p * np.sqrt(self.n_obs) / np.linalg.norm(self.d_p)))


def make_synthetic(phys, full, pc, kle=None):
    """full: kle.FullRankGaussian on the cells (exact Cholesky draw).
    truth_mode "kle": the "standardized" truth (the E1-E4 field) projected
    onto the span of the truncated KLE `kle` (the samplers' own basis):
    theta_true = T^+ G_std, G = T theta_true.  It is the E1-E4 field minus its
    small-scale part (spatial mean ~0, sd ~0.97), so (beta, rho) are still
    what the data see, and it lies exactly in the sampler's space.
    truth_mode "kle_raw": the raw full-rank draw projected instead; since the
    KLE modes are eigenvectors of C this is exactly theta_true ~ N(0, I_r), a
    draw from the sampler's prior, with its own nonzero spatial mean.
    Projecting (rather than drawing theta directly) keeps both independent of
    how eigh rotates tied modes (BLAS threads, machine)."""
    rng = np.random.default_rng(pc.truth_seed)
    theta_true = None
    G_raw = full.sample(rng)
    if pc.truth_mode in ("kle", "kle_raw"):
        if kle is None:
            raise ValueError(f'truth_mode "{pc.truth_mode}" needs the KLE basis')
        src = ((G_raw - G_raw.mean()) / G_raw.std() if pc.truth_mode == "kle"
               else G_raw)
        theta_true = kle.project(src)
    if theta_true is not None:
        G = kle.field(theta_true)
    elif pc.truth_mode == "standardized":
        G = (G_raw - G_raw.mean()) / G_raw.std()
    elif pc.truth_mode == "raw":
        G = G_raw.copy()
    else:
        raise ValueError(pc.truth_mode)
    logK = np.log(pc.beta_true) + pc.rho_true * G
    K = np.exp(logK)
    p = phys.solve(K)
    sigma_p = pc.noise_rel * (p.max() - p.min())
    d_p = p[phys.obs_mask] + sigma_p * rng.standard_normal(phys.obs_mask.sum())
    keff_true = phys.keff(K, p)
    keff_obs = keff_true * np.exp(pc.sigma_e * rng.standard_normal())
    return Synthetic(mode=pc.truth_mode, beta_true=pc.beta_true,
                     rho_true=pc.rho_true, G_raw=G_raw, G_true=G,
                     logK_true=logK, K_true=K, p_true=p,
                     obs_mask=phys.obs_mask.copy(), d_p=d_p,
                     theta_true=theta_true,
                     sigma_p=float(sigma_p), sigma_e=pc.sigma_e,
                     keff_true=float(keff_true), keff_obs=float(keff_obs),
                     level_true=float(logK.mean()), Gbar=float(G.mean()),
                     G_std=float(G.std()))
