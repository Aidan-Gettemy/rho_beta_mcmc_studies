"""Plot the global theta of every chain of one MSM experiment (needs <dir>/chains.npz, cluster only).

  fig_theta_traces.png   theta_k vs iteration for all chains (thin lines) for selected coordinates,
                         plus |theta|^2 and the relative error per chain; adaptation end marked
  fig_theta_chainmeans.png  heat map of chain means (post-adaptation) for all 64 coordinates x chains
                         (chains sorted by median relative error), and per-coordinate between-chain
                         sd vs within-chain sd (what the MPSRF compares)
    python experiments/plot_msm_theta_chains.py --dir runs/msm_local_kle_2026-11/exp4_8_mixture_d8
    options: --coords 0 1 2 5 10 30 55 63   --state theta|c_primal|c_dual   --out <dir>/post
"""
import _common  # noqa: F401
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--coords", type=int, nargs="+", default=[0, 1, 2, 5, 10, 30, 55, 63])
    ap.add_argument("--state", default="theta")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    Z = np.load(os.path.join(a.dir, "chains.npz"))
    X = Z[a.state].astype(float)                       # (m, n_save, dim)
    m, ns, dim = X.shape
    try:
        st = json.load(open(os.path.join(a.dir, "summary.json")))["settings"]
        se, M, N = st["save_every"], st["adapt"], st["n_iter"]
    except Exception:
        N = Z["relerr"].shape[1]; se = N // ns; M = N // 10
    it = np.arange(ns) * se
    out = a.out or os.path.join(a.dir, "post"); os.makedirs(out, exist_ok=True)
    rel = Z["relerr"].astype(float)
    post = it >= M
    rmed = np.median(rel[:, M:], axis=1); order = np.argsort(rmed)
    co = [c for c in a.coords if c < dim]

    nc = 3; nr = int(np.ceil((len(co) + 2) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(15, 3.1 * nr), constrained_layout=True)
    ax = ax.ravel()
    cmap = plt.get_cmap("viridis")
    for k, c in enumerate(co):
        for j in range(m):
            ax[k].plot(it, X[j, :, c], lw=.5, alpha=.55, color=cmap(np.argsort(order)[j] / max(m - 1, 1)))
        ax[k].set_title(f"{a.state}[{c}]  (colour = chain's median rel. err., low to high)", fontsize=9)
    n2 = (X ** 2).sum(2)
    for j in range(m):
        ax[len(co)].plot(it, n2[j], lw=.5, alpha=.55, color=cmap(np.argsort(order)[j] / max(m - 1, 1)))
    ax[len(co)].set_title(f"|{a.state}|^2 (prior mean {dim if a.state=='theta' else ''})", fontsize=9)
    ax[len(co)].set_yscale("log")
    for j in range(m):
        ax[len(co) + 1].plot(np.arange(rel.shape[1])[::20], rel[j, ::20], lw=.5, alpha=.55,
                             color=cmap(np.argsort(order)[j] / max(m - 1, 1)))
    ax[len(co) + 1].set_yscale("log"); ax[len(co) + 1].set_title("relative error per chain", fontsize=9)
    for a_ in ax[:len(co) + 2]:
        a_.axvline(M, color="k", ls=":", lw=.8); a_.set_xlabel("iteration")
    for a_ in ax[len(co) + 2:]: a_.axis("off")
    fig.suptitle(f"{os.path.basename(os.path.normpath(a.dir))}: {m} chains, dotted line = end of adaptation")
    fig.savefig(os.path.join(out, "fig_theta_traces.png"), dpi=130); plt.close(fig)

    Xp = X[:, post, :]
    mu = Xp.mean(1)                                    # (m, dim) chain means
    W = Xp.var(1, ddof=1).mean(0); Bn = mu.var(0, ddof=1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    im = ax[0].imshow(mu[order], aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax[0].set_xlabel("coordinate"); ax[0].set_ylabel("chain (sorted by median rel. err.)")
    ax[0].set_title("chain means after adaptation"); fig.colorbar(im, ax=ax[0])
    ax[1].plot(np.sqrt(Bn), label="between-chain sd of chain means"); ax[1].plot(np.sqrt(W), label="within-chain sd")
    ax[1].set_xlabel("coordinate"); ax[1].legend(fontsize=8); ax[1].set_title("what the PSRF compares")
    ax[2].plot(np.sort(rmed), ".-"); ax[2].set_title("median rel. err. per chain (sorted)"); ax[2].set_xlabel("chain rank")
    fig.suptitle(f"{os.path.basename(os.path.normpath(a.dir))}: {a.state}, window iterations {M}..{N}")
    fig.savefig(os.path.join(out, "fig_theta_chainmeans.png"), dpi=130); plt.close(fig)
    print("wrote", out, f"| chains {m}, median relerr {np.median(rmed):.4f}, "
          f"chains above 0.03: {(rmed > 0.03).sum()}/{m} | mean |{a.state}|^2 after adaptation {n2[:, post].mean():.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
