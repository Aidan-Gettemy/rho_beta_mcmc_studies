"""pCN / block-pCN on the Gaussian surrogate of the theta posterior (beta, rho
fixed at the truth): Phi(theta) = 1/2 theta^T H theta - b^T theta.  Costs are
counted in theta-solves (one per move).  Mixing: integrated autocorrelation
time (IACT) of every posterior principal component, in solves."""
import os, sys, json, time
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import numpy as np
from lin import load
from brho.blockpcn import from_problem

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "lin_2d_r208.npz"))
J, w, th_t = z["J"], float(z["w"]), z["th"]
prob, _ = load(os.path.join(REPO, "runs") + "/cluster_l005")
syn = prob.syn
p_true = prob.potential(syn.beta_true * np.exp(syn.rho_true * prob.kle.field(th_t)))[1]
eps = syn.d_p - p_true
H = w * J.T @ J
b = w * J.T @ (eps + J @ th_t)                 # Phi(th) = 1/2 th'H th - b'th + const
r = H.shape[0]
Sig = np.linalg.inv(np.eye(r) + H)
ev, PC = np.linalg.eigh(Sig)                    # posterior principal components

def phi(TH):                                    # (C, r) -> (C,)
    return 0.5 * np.sum((TH @ H) * TH, 1) - TH @ b

def layout(n):
    if n == 0:
        return [np.eye(r)]                      # global pCN
    bp = from_problem(prob, n, 0.9)
    return [mv["V"] for mv in bp.moves if mv["dim"]]

def iact(X):                                    # X (C, N, k): Sokal window, chain-averaged ACF
    X = X - X.mean(1, keepdims=True)
    C, N, k = X.shape
    f = np.fft.rfft(X, n=2 * N, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :N].mean(0)
    ac = ac / ac[0]
    tau = np.ones(k)
    for j in range(k):
        s = 1.0
        for m in range(1, N):
            s += 2 * ac[m, j]
            if m >= 5 * s:
                break
        tau[j] = s
    return tau

def run(Vs, probs=None, solves=300_000, C=32, burn=60_000, rec=50, seed=0, s0=0.15, adapt=True,
        s_fixed=None):
    rng = np.random.default_rng(seed)
    nm = len(Vs)
    s = np.full(nm, s0) if s_fixed is None else np.array(s_fixed, float)
    acc = np.zeros(nm); cnt = np.zeros(nm)
    th = np.linalg.solve(np.eye(r) + H, b)[:, None] + np.linalg.cholesky(Sig) @ rng.standard_normal((r, C))
    th = th.T.copy()                            # start at exact posterior draws
    ph = phi(th)
    kept = []; used = 0; seq = 0
    while used < burn + solves:
        j = (seq % nm) if probs is None else int(rng.choice(nm, p=probs))
        seq += 1
        V = Vs[j]
        a = th @ V
        prop = th + ((np.sqrt(1 - s[j] ** 2) - 1) * a + s[j] * rng.standard_normal(a.shape)) @ V.T
        pp = phi(prop)
        ok = np.log(rng.random(C)) < ph - pp
        th[ok] = prop[ok]; ph[ok] = pp[ok]
        used += 1
        if used <= burn and adapt:
            acc[j] += ok.mean(); cnt[j] += 1
            if cnt[j] >= 100:
                s[j] = np.clip(s[j] * (1.12 if acc[j] / cnt[j] > 0.25 else 1 / 1.12), 1e-3, 0.95)
                acc[j] = cnt[j] = 0
        if used > burn and (used - burn) % rec == 0:
            kept.append(th @ PC)
    X = np.stack(kept, 1)                       # (C, N, r)
    tau = iact(X) * rec                         # in theta-solves
    return dict(steps=s, tau=tau, tau_max=float(tau.max()), tau_med=float(np.median(tau)),
                tau_95=float(np.quantile(tau, 0.95)))

if __name__ == "__main__":
    which = sys.argv[1:] or ["global", "2x2", "3x3"]
    out = {}
    for name in which:
        t0 = time.time()
        Vs = layout({"global": 0, "2x2": 2, "3x3": 3}[name])
        res = run(Vs)
        np.save(os.path.join(HERE, f"steps_{name}.npy"), res["steps"])
        np.save(os.path.join(HERE, f"tau_{name}.npy"), res["tau"])
        out[name] = {k: v for k, v in res.items() if k.startswith("tau_")}
        out[name]["steps"] = np.round(res["steps"], 4).tolist()
        print(name, json.dumps(out[name]), f"{time.time()-t0:.0f}s", flush=True)
