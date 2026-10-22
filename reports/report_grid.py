"""
E2 figures and tables (non-uniqueness report Figs. 2-5 + two extras).

    fig_grid_beta_rho.png   MAIN: disks at the 25 pairs, colored/labelled by
                            posterior-mean RMS/sigma_p; truth star
    fig_grid_beta_rho_relerr.png  same disks, colored by the posterior mean of
                            the samples' relative error ||d - G|| / ||d||
                            (relative likelihood only; + grid_relerr table)
    fig_grid_fit.png        posterior-mean predicted vs observed pressure, 25 chains
    fig_grid_slice.png      posterior-mean log K along y=0.53 for 3 pairs + truth
    fig_grid_traces.png     misfit traces of all 25 chains vs the noise floor E[Phi(truth)]
    fig_grid_theta_norm.png posterior mean ||theta||^2 / r per pair (the weak
                            prior penalty that the hierarchical model exploits)
    fig_grid_fields.png     posterior-mean log K maps for a 5x5 panel
    tables/e2_grid/grid_table.{md,csv}, grid_summary.json
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import numpy as np
import matplotlib.pyplot as plt

from _common import have, load_problem, io
from brho import plotting as P

EXP = io.EXP_GRID
PICKS = [(0.5, 0.5), (2.3, 1.5), (10.0, 3.0)]


def relerr_grid(prob, z, n_burn):
    """fig_grid_beta_rho_relerr.png: the 25 pairs colored by the posterior mean
    of the samples' pressure relative error ||d - G(theta)|| / ||d||.  Under the
    relative likelihood Phi = relerr^2 / (2 sigma_rel^2), so each stored misfit
    trace gives relerr = sigma_rel sqrt(2 Phi) exactly, sample by sample.
    Reference: the truth's own relative error ||d - G(truth)|| / ||d|| (the
    noise), marked on the colorbar.  Table: grid_relerr.{md,csv}."""
    syn = prob.syn
    beta, rho = z["beta"], z["rho"]
    re = prob.sigma_rel * np.sqrt(2.0 * z["traces"][:, n_burn:])        # (25, n_post)
    mean_re = re.mean(1)
    d = syn.d_p
    re_truth = float(np.linalg.norm(d - syn.p_true[syn.obs_mask]) / np.linalg.norm(d))

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    vmin, vmax = 0.98 * min(re_truth, mean_re.min()), mean_re.max()
    sc = ax.scatter(beta, rho, c=100 * mean_re, s=420, cmap="viridis",
                    vmin=100 * vmin, vmax=100 * vmax, edgecolors="k", zorder=3)
    for b_, r_, m_ in zip(beta, rho, mean_re):
        light = (m_ - vmin) / (vmax - vmin) > 0.6
        at_truth = np.isclose(b_, syn.beta_true) and np.isclose(r_, syn.rho_true)
        ax.annotate(f"{100 * m_:.2f}", (b_, r_), ha="center", va="center", fontsize=7.5,
                    color="k" if (light or at_truth) else "w", zorder=6,
                    xytext=(0, -19) if at_truth else (0, 0), textcoords="offset points")
    P.truth_star(ax, syn.beta_true, syn.rho_true)
    ax.set_xscale("log")
    ax.set_xlabel(rf"$\beta$ (truth {syn.beta_true})")
    ax.set_ylabel(rf"$\rho$ (truth {syn.rho_true})")
    ax.set_title("Posterior mean of the samples' relative error "
                 r"$\|d - G(\theta)\| / \|d\|$ (%)" "\nper fixed-$(\\beta,\\rho)$ pCN chain; "
                 f"truth: {syn.mode}")
    cb = plt.colorbar(sc, ax=ax, label="relative error of the samples (%)")
    cb.ax.axhline(100 * re_truth, color="r", lw=2)
    cb.ax.text(1.05, 100 * re_truth, f" truth {100 * re_truth:.2f}%", color="r",
               transform=cb.ax.get_yaxis_transform(), va="center", fontsize=8)
    P.save(fig, EXP, "fig_grid_beta_rho_relerr")

    rows = [dict(beta=float(b_), rho=float(r_), relerr_mean=float(m_),
                 relerr_median=float(np.median(x)), relerr_q05=float(np.percentile(x, 5)),
                 relerr_q95=float(np.percentile(x, 95)), relerr_truth=re_truth)
            for b_, r_, m_, x in zip(beta, rho, mean_re, re)]
    io.write_table(EXP, "grid_relerr", rows)
    print(f"E2 relative error: samples {100 * mean_re.min():.2f}-{100 * mean_re.max():.2f}%, "
          f"truth {100 * re_truth:.2f}%")


def main():
    if not have(EXP):
        print("E2: no results"); return
    prob, man = load_problem(EXP)
    z = np.load(io.results_dir(EXP) + "/grid.npz")
    syn, n = prob.syn, prob.pc.n
    beta, rho, rms = z["beta"], z["rho"], z["rms"]
    sig = syn.sigma_p
    gcfg = man["configs"]["GridStudyConfig"]
    n_burn = gcfg["n_burn"]
    floor = prob.noise_floor()

    # ---- main 5x5 picture
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(beta, rho, c=rms / sig, s=420, cmap="coolwarm", vmin=0,
                    vmax=3, edgecolors="k", zorder=3)
    for b_, r_, m_ in zip(beta, rho, rms):
        ax.annotate(f"{m_ / sig:.1f}", (b_, r_), ha="center", va="center",
                    fontsize=8, zorder=4)
    P.truth_star(ax, syn.beta_true, syn.rho_true)
    ax.set_xscale("log")
    ax.set_xlabel(rf"$\beta$ (truth {syn.beta_true})")
    ax.set_ylabel(rf"$\rho$ (truth {syn.rho_true})")
    ax.set_title("Posterior-mean misfit RMS/$\\sigma_p$ per fixed-$(\\beta,\\rho)$ pCN chain\n"
                 f"truth: {syn.mode}")
    plt.colorbar(sc, ax=ax, label=r"RMS / $\sigma_p$")
    P.save(fig, EXP, "fig_grid_beta_rho")

    # ---- same picture, colored by the samples' own relative error
    if prob.pc.likelihood == "relative":
        relerr_grid(prob, z, n_burn)

    # ---- data fit overlay
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for i in range(len(beta)):
        ax.plot(syn.d_p, z["mean_pobs"][i], ".", ms=3.5, alpha=0.4)
    lo, hi = syn.d_p.min(), syn.d_p.max()
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xlabel("observed pressure"); ax.set_ylabel("posterior-mean prediction")
    ax.set_title(f"All {len(beta)} chains overlaid")
    P.save(fig, EXP, "fig_grid_fit")

    # ---- log K slices
    jr = P.slice_row(n)
    xs = prob.phys.xc
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(xs, P.line(syn.logK_true, n, jr), "k-", lw=2.5, label="truth")
    for b_, r_ in PICKS:
        i = np.where(np.isclose(beta, b_) & np.isclose(rho, r_))[0]
        if len(i):
            lk = np.log(b_) + r_ * z["mean_G"][i[0]]
            ax.plot(xs, P.line(lk, n, jr), "--", lw=1.7,
                    label=rf"$\beta={b_},\ \rho={r_}$")
    ax.set_xlabel(f"x (slice y = {xs[jr]:.2f})"); ax.set_ylabel(r"posterior mean $\log K$")
    ax.set_title(r"Different $(\beta,\rho)$, same recovered $\log K$ level")
    ax.legend()
    P.save(fig, EXP, "fig_grid_slice")

    # ---- traces
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    th = max(1, z["traces"].shape[1] // 400)
    for tr in z["traces"]:
        ax.semilogy(np.arange(0, len(tr), th), tr[::th], lw=0.6, alpha=0.6)
    ax.axhline(floor, color="k", ls=":", label=(rf"noise floor $n_{{obs}}/2={floor:.0f}$" if prob.pc.likelihood == "absolute" else rf"noise floor $\Phi$(truth) $\approx {floor:.0f}$"))
    ax.axvline(n_burn, color="gray", ls="--", lw=1, label="end burn-in")
    ax.set_xlabel("pCN iteration"); ax.set_ylabel(r"$\Phi(\theta)$")
    ax.set_title(f"Misfit traces of the {len(beta)} chains"); ax.legend()
    P.save(fig, EXP, "fig_grid_traces")

    # ---- ||theta||^2 / r
    tn = z["theta_norm2_mean"] / prob.r
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(beta, rho, c=np.log10(tn), s=420, cmap="PuOr_r", vmin=-1,
                    vmax=1, edgecolors="k", zorder=3)
    for b_, r_, v in zip(beta, rho, tn):
        ax.annotate(f"{v:.2f}", (b_, r_), ha="center", va="center", fontsize=8, zorder=4)
    P.truth_star(ax, syn.beta_true, syn.rho_true)
    ax.set_xscale("log"); ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\rho$")
    ax.set_title(r"posterior mean $\|\theta\|^2/r$ (prior-typical value 1)")
    plt.colorbar(sc, ax=ax, label=r"$\log_{10}\,\|\theta\|^2/r$")
    P.save(fig, EXP, "fig_grid_theta_norm")

    # ---- 5x5 panel of posterior-mean log K
    betas, rhos = z["betas"], z["rhos"]
    fig, axs = plt.subplots(len(rhos), len(betas), figsize=(2.1 * len(betas),
                                                           2.0 * len(rhos)))
    axs = np.atleast_2d(axs)
    lks = [np.log(b) + r * g for b, r, g in zip(beta, rho, z["mean_G"])]
    v0, v1 = np.percentile(np.concatenate(lks + [syn.logK_true]), [1, 99])
    for i, (b_, r_) in enumerate(zip(beta, rho)):
        a = axs[len(rhos) - 1 - int(np.argmin(abs(rhos - r_))),
                int(np.argmin(abs(betas - b_)))]
        P.field(a, lks[i], n, rf"$\beta$={b_:g}, $\rho$={r_:g}", vmin=v0,
                vmax=v1, cbar=False)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("posterior-mean $\\log K$ for every fixed pair (shared color scale)")
    P.save(fig, EXP, "fig_grid_fields")

    # ---- tables
    rows = [dict(beta=float(b_), rho=float(r_), rms_over_sigma=float(m_ / sig),
                 mean_phi=float(mp), best_phi=float(bp), acc=float(a_),
                 theta_norm2_over_r=float(t_), step=float(s_))
            for b_, r_, m_, mp, bp, a_, t_, s_ in zip(
                beta, rho, rms, z["mean_phi"], z["best_phi"], z["acc"], tn,
                z["s_final"])]
    io.write_table(EXP, "grid_table", rows)
    summ = dict(truth_mode=syn.mode, r=prob.r, n_chains=len(beta),
                n_iter=gcfg["n_iter"], n_burn=n_burn, noise_floor=floor,
                rms_over_sigma=[float((rms / sig).min()), float((rms / sig).max())],
                best_phi=[float(z["best_phi"].min()), float(z["best_phi"].max())],
                mean_phi=[float(z["mean_phi"].min()), float(z["mean_phi"].max())],
                acc=[float(z["acc"].min()), float(z["acc"].max())],
                minutes=man.get("minutes"))
    io.save_json(io.tables_dir(EXP) + "/grid_summary.json", summ)
    print("E2:", summ)


if __name__ == "__main__":
    main()
