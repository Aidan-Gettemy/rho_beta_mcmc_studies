"""Linearized (Gaussian) posterior at the truth: J by central differences."""
import os, sys, json
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import numpy as np

def load(root, exp="e5_soft"):
    os.environ["BRHO_ROOT"] = root
    import importlib
    from brho import io, config as C
    importlib.reload(io)
    from brho.problem import Problem
    man = io.read_manifest(exp)
    pc = C.from_dict(C.ProblemConfig, man["configs"]["ProblemConfig"])
    prob = Problem(pc, frozen=io.results_dir(exp))
    return prob, man

def jac(prob, h=1e-4):
    syn, kle = prob.syn, prob.kle
    th = kle.project(syn.G_true) if not hasattr(syn, "theta_true") else np.asarray(syn.theta_true)
    b, r_ = syn.beta_true, syn.rho_true
    f = lambda t: prob.potential(b * np.exp(r_ * kle.field(t)))[1]
    p0 = f(th)
    J = np.empty((len(p0), kle.r))
    for i in range(kle.r):
        e = np.zeros(kle.r); e[i] = h
        J[:, i] = (f(th + e) - f(th - e)) / (2 * h)
    return th, p0, J

def analyse(root, label, rho_post=None, phi_post=None):
    prob, man = load(root)
    syn = prob.syn
    th, p0, J = jac(prob)
    w = prob._w
    eps = syn.d_p - p0                      # the noise realization (p0 = p at truth, obs cells)
    kappa = w * syn.sigma_p ** 2
    H = w * J.T @ J
    lam, V = np.linalg.eigh(H); lam, V = lam[::-1], V[:, ::-1]
    r = len(lam)
    shrink = lam / (1 + lam)
    floor = prob.noise_floor()
    # linear-Gaussian theta posterior at fixed (beta, rho) = truth
    A = np.eye(r) + H
    m = np.linalg.solve(A, w * J.T @ eps - th)          # posterior mean of (theta - theta*)
    phi_mean = 0.5 * w * np.sum((eps - J @ m) ** 2)
    phi_post_pred = phi_mean + 0.5 * shrink.sum()
    # rho evidence in u = rho*theta space, theta ~ N(0, I)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    z = U.T @ (eps + J @ th)                             # informative components of the data
    rho_t = syn.rho_true
    grid = np.linspace(0.6, 2.6, 801)
    def loglik(rho):
        var = (rho / rho_t) ** 2 * s ** 2 + 1.0 / w
        return -0.5 * np.sum(z ** 2 / var + np.log(var))
    L = np.array([loglik(g) for g in grid])
    lp = L - 0.5 * (np.log(grid) / 0.75) ** 2 - np.log(grid)   # lognormal(0, 0.75) prior density
    pdf = np.exp(lp - lp.max()); cdf = np.cumsum(pdf); cdf /= cdf[-1]
    q = lambda a: float(np.interp(a, cdf, grid))
    # energy of theta* along data-informed directions
    c = V.T @ th                                         # theta* in H-eigenbasis
    wts = shrink ** 2
    e_inf = float(np.sum(wts * c ** 2) / np.sum(wts))
    out = dict(label=label, r=r, n_obs=int(syn.n_obs), kappa=float(kappa), floor=float(floor),
               d_eff=float(shrink.sum()), n_lam_gt1=int((lam > 1).sum()),
               lam_top=[float(x) for x in lam[:5]],
               phi_post_linear_pred=float(phi_post_pred), phi_post_mcmc=phi_post,
               rho_linear_median=q(0.5), rho_linear_90=[q(0.05), q(0.95)], rho_mcmc=rho_post,
               theta_norm2_over_r=float(th @ th / r),
               informed_energy=e_inf, rho_pred_from_energy=float(rho_t * np.sqrt(e_inf)))
    np.savez(os.path.join(os.path.dirname(__file__), f"lin_{label}.npz"), lam=lam, V=V, th=th, J=J, w=w)
    return out

if __name__ == "__main__":
    R = os.path.join(REPO, "runs") + ""
    res = [analyse(f"{R}/e5_kle_2026-09", "2d_r57", [1.473, 1.23, 1.79], 69.54),
           analyse(f"{R}/cluster_l005", "2d_r208", [1.204, 1.05, 1.38], 108.3)]
    for o in res:
        print(json.dumps(o, indent=1))
