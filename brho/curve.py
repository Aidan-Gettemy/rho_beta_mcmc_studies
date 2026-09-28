"""
E3 - offline Monte Carlo construction of the beta(rho) curve.

Scaling lemma: Keff(c K) = c Keff(K)  =>  imposing Keff = K* on realization
G_i at amplitude rho gives beta_i(rho) = K* / Keff(exp(rho G_i)).  All solves
are at beta = 1, and K* only rescales the curve, so we store the beta = 1
effective permeabilities keff1[a, i] and apply K* at evaluation time:

    beta_bar(rho) = K* / f(rho),    f = statistic over i of keff1[:, i]

The same M realizations G_i are used at every rho (common random numbers),
drawn from the truncated global KLE so the curve targets the same measure as
the sampler (multiscale report, consistency requirement 1).
Interpolation: monotone cubic (PCHIP) in log f vs rho, with the exact anchor
f(0) = 1 (a homogeneous plug has Keff = beta).
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

STATISTICS = ("ratio_of_mean", "mean_of_ratio", "median")


def realizations(prob, cfg, rng):
    if cfg.field == "kle":
        return prob.kle.field(prob.kle.sample_theta(rng, cfg.n_real))
    if cfg.field == "cholesky":
        return prob.full.sample(rng, cfg.n_real)
    raise ValueError(cfg.field)


def sweep(prob, cfg, progress=None):
    """Run the Monte Carlo sweep. Returns (rhos, keff1 (n_rho, M), G (M, N))."""
    rng = np.random.default_rng(cfg.seed)
    G = realizations(prob, cfg, rng)
    rhos = np.linspace(cfg.rho_min, cfg.rho_max, cfg.n_rho)
    keff1 = np.empty((len(rhos), cfg.n_real))
    for a, rho in enumerate(rhos):
        for i in range(cfg.n_real):
            K = np.exp(rho * G[i])
            keff1[a, i] = prob.phys.keff(K, prob.phys.solve(K))
        if progress:
            progress(a, rho, keff1[a])
    return rhos, keff1, G


def f_stat(keff1, statistic):
    if statistic == "ratio_of_mean":
        return keff1.mean(-1)
    if statistic == "mean_of_ratio":
        return 1.0 / (1.0 / keff1).mean(-1)
    if statistic == "median":
        return np.median(keff1, -1)
    raise ValueError(statistic)


class BetaCurve:
    def __init__(self, rhos, keff1, statistic="ratio_of_mean"):
        self.rhos, self.keff1 = np.asarray(rhos), np.asarray(keff1)
        self.statistic = statistic
        self.rho_max = float(self.rhos.max())
        f = f_stat(self.keff1, statistic)
        x = np.concatenate([[0.0], self.rhos])
        y = np.concatenate([[0.0], np.log(f)])        # anchor f(0) = 1
        self._logf = PchipInterpolator(x, y, extrapolate=False)
        # realization scatter of log Keff (= of log beta_i), anchored at 0 for rho = 0
        sd = np.log(self.keff1).std(-1, ddof=1)
        self._logsd = PchipInterpolator(x, np.concatenate([[0.0], sd]),
                                        extrapolate=False)

    # ------------------------------------------------------------ evaluate
    def f(self, rho):
        return np.exp(self._logf(rho))

    def beta_bar(self, rho, Kstar):
        """NaN outside [0, rho_max] (the sampler treats that as rejection)."""
        return Kstar / self.f(rho)

    def log_scatter(self, rho):
        """sd_i of log beta_i(rho) = sd_i of log Keff_i(rho) (E3 realizations);
        NaN outside [0, rho_max].  The soft curve prior (E5) uses
        s(rho)^2 = log_scatter(rho)^2 + sigma_e^2 (report.pdf, Remark 3)."""
        return self._logsd(rho)

    def band(self, Kstar):
        """Per-grid-point statistics of the beta_i sample (for plots/tables)."""
        b = Kstar / self.keff1
        return dict(rho=self.rhos, mean=b.mean(1), std=b.std(1, ddof=1),
                    median=np.median(b, 1),
                    q05=np.percentile(b, 5, 1), q95=np.percentile(b, 95, 1),
                    curve=self.beta_bar(self.rhos, Kstar),
                    f_mean=self.keff1.mean(1),
                    f_se=self.keff1.std(1, ddof=1) / np.sqrt(self.keff1.shape[1]),
                    keff_relstd=self.keff1.std(1, ddof=1) / self.keff1.mean(1))

    # ---------------------------------------------------------------- io
    def save(self, path, **extra):
        np.savez_compressed(path, rhos=self.rhos, keff1=self.keff1,
                            statistic=self.statistic, **extra)

    @classmethod
    def load(cls, path, statistic=None):
        z = np.load(path)
        return cls(z["rhos"], z["keff1"], statistic or str(z["statistic"]))
