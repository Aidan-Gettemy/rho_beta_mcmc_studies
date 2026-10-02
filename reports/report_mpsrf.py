"""
MPSRF and PSRF convergence curves for all KLE coefficients theta, in the
group's MATLAB convention (brho/mpsrf.py has the formulas).

    python reports/report_mpsrf.py [--exp e1_blocked] [--chains 16]
                                   [--skip 2 --step 15 --iter-burn 10]
                                   [--at 4900] [--level 1.2]

Uses the thinned samples saved in every chain file (kept_theta, every `thin`
iterations from iteration 0).  As in computeMPSRFcurve, the curve point at
thinned count k uses samples iter_burn, iter_burn + skip, ..., k (a growing
window from sample iter_burn); the x axis is k * thinning.  Two parameter sets:
    theta   all r KLE coefficients
    all     (theta, log beta, log rho)  = the whole sampled state up to an
            affine map (MPSRF is invariant to those)
Runs: e1_blocked, e1_global (E1), e4_sampler (E4) and e5_soft (E5; beta free,
so its "all" set is (theta, log beta, log rho) like E1's).  For E4, beta =
beta_bar(rho) is not a free parameter, so its "all" set is (theta, log rho),
and its samples start after burn-in (the sampler keeps only those).
With two or more runs present, figures/mpsrf_compare/ adds
    fig_mpsrf_compare_theta.png   MPSRF of all theta, every run on one axis
    fig_psrf_compare_theta.png    worst-component PSRF of theta, same
with x = PDE solves per chain (E1 blocked 2 per iteration, global 1, E4 3,
E5 / E7 3: A, B, D; C is free; E8 from the manifest's solves_per_iteration,
e.g. 16 for the 2D block sweep), so equal x means equal work; tables/mpsrf_compare/mpsrf_compare.json.
Writes, for each set, into figures/<exp>/ and tables/<exp>/:
    fig_mpsrf_<set>.png   MPSRF vs iteration: y = 1, y = 1.2, crossing marked
    fig_psrf_<set>.png    max-over-components PSRF vs iteration, and the final
                          PSRF of every component
    mpsrf_<set>.csv       iteration, MPSRF, max PSRF, PSRF of each component
    mpsrf_summary.json    convergence iterations and values at --at
"""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from _common import io
from brho import mpsrf as M


def _chain_paths(exp, n_chains=None):
    paths = io.chain_files(io.results_dir(exp))
    if not paths:
        raise FileNotFoundError(f"no chain files in {io.results_dir(exp)}")
    return paths[:n_chains] if n_chains else paths


def load_kept(exp, n_chains=None):
    """Thinned samples of one run in a common form.

    E1 (e1_blocked / e1_global): kept from iteration 0 (burn-in included);
        sets theta and all = (theta, log beta, log rho); 2 / 1 solves per it.
    E4 (e4_sampler): kept after burn-in only (the sampler's own layout);
        beta = beta_bar(rho) is not a free parameter, so all = (theta, log rho);
        3 solves per iteration."""
    paths = _chain_paths(exp, n_chains)
    zs = [np.load(p) for p in paths]
    if "kept_it" in zs[0].files:                                   # E1
        L = min(len(z["kept_it"]) for z in zs)
        theta = np.stack([z["kept_theta"][:L] for z in zs])
        lb = np.stack([z["kept_logbeta"][:L] for z in zs])
        lr = np.log(np.stack([z["kept_rho"][:L] for z in zs]))
        its = np.asarray(zs[0]["kept_it"][:L])
        move = str(zs[0]["move"]) if "move" in zs[0].files else "global"
        full = np.concatenate([theta, lb[..., None], lr[..., None]], axis=2)
        return dict(sets=dict(theta=theta, all=full),
                    extra=dict(theta=[], all=["logβ", "logρ"]),
                    its=its, m=len(paths), solves=2 if move == "blocked" else 1,
                    label=f"E1 naive pCN ({move})")
    L = min(len(z["kept_theta"]) for z in zs)                      # E4 / E5
    theta = np.stack([z["kept_theta"][:L] for z in zs])
    lr = np.log(np.stack([z["kept_rho"][:L] for z in zs]))
    cfgs = io.read_manifest(exp)["configs"]
    if "SoftSamplerConfig" in cfgs:                                # E5: beta free
        sc = cfgs["SoftSamplerConfig"]
        lb = np.log(np.stack([z["kept_beta"][:L] for z in zs]))
        its = sc["n_burn"] + sc["thin"] * np.arange(L)
        full = np.concatenate([theta, lb[..., None], lr[..., None]], axis=2)
        innov = sc.get("theta_innovation", "global")
        man = io.read_manifest(exp)
        if innov.startswith("block"):
            lab = ("E8 soft-curve sampler, block pCN θ"
                   + (" (random block)" if innov == "block_random" else " (sweep)"))
            b = man.get("block") or {}
            if b.get("coarse") is False:
                lab += ", no coarse"
            if (b.get("dual_halo") or 0) > (b.get("halo") or 0):
                lab += f", dual halo {b['dual_halo']:g}"
        elif innov == "global":
            lab = "E5 soft-curve sampler (β free)"
        else:
            lab = f"E7 soft-curve sampler, {innov.replace('ms_', 'multiscale ')} θ"
        return dict(sets=dict(theta=theta, all=full),
                    extra=dict(theta=[], all=["logβ", "logρ"]),
                    its=its, m=len(paths), solves=man.get("solves_per_iteration", 3),
                    label=lab)
    sc = cfgs["SamplerConfig"]
    its = sc["n_burn"] + sc["thin"] * np.arange(L)
    return dict(sets=dict(theta=theta, all=np.concatenate([theta, lr[..., None]], axis=2)),
                extra=dict(theta=[], all=["logρ"]),
                its=its, m=len(paths), solves=3, label="E4 3-step curve sampler")


def style(ax, fs=20):
    ax.tick_params(labelsize=fs)
    ax.xaxis.label.set_size(fs)
    ax.yaxis.label.set_size(fs)
    ax.title.set_size(fs + 2)


def fig_mpsrf(x, curve, conv, level, title, path):
    f, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, np.abs(curve), lw=2, label="MPSRF")
    ax.axhline(1.0, color="r", ls="-", lw=3, label="y=1.0")
    ax.axhline(level, color="r", ls=":", lw=3, label=f"y={level:g}")
    if conv is not None:
        ax.plot([conv, conv], [0, level], "k--", lw=3,
                label=f"MPSRF = {int(conv)}")
        ax.scatter([conv], [level], color="b", zorder=5)
    else:
        ax.plot([], [], "k--", lw=3, label=f"no crossing of {level:g}")
    ax.set_ylim(0, 5)
    ax.set_xlim(0, x.max())
    if np.nanmin(np.abs(curve)) > 5:
        ax.text(0.5, 0.5, f"above 5 throughout (final {abs(curve[-1]):.3g})",
                transform=ax.transAxes, ha="center", fontsize=18)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\hat{R}$")
    ax.set_title(title)
    ax.legend(fontsize=18)
    style(ax)
    f.tight_layout()
    f.savefig(path, dpi=150)
    plt.close(f)


def fig_psrf(x, psrf_max, psrf_final, conv, level, names, title, path):
    f, ax = plt.subplots(1, 2, figsize=(18, 7),
                         gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax[0].plot(x, psrf_max, lw=2, label="max PSRF")
    ax[0].axhline(1.0, color="r", ls="-", lw=3, label="y=1.0")
    ax[0].axhline(level, color="r", ls=":", lw=3, label=f"y={level:g}")
    if conv is not None:
        ax[0].plot([conv, conv], [0, level], "k--", lw=3, label=f"PSRF = {int(conv)}")
        ax[0].scatter([conv], [level], color="b", zorder=5)
    ax[0].set_ylim(0, 5)
    ax[0].set_xlim(0, x.max())
    if np.nanmin(psrf_max) > 5:
        ax[0].text(0.5, 0.5, f"above 5 throughout (final {psrf_max[-1]:.3g})",
                   transform=ax[0].transAxes, ha="center", fontsize=16)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel(r"$\max_i \hat{R}_i$")
    ax[0].set_title("PSRF (worst component) vs iteration")
    ax[0].legend(fontsize=16)
    idx = np.arange(len(psrf_final))
    ax[1].bar(idx, psrf_final, color=np.where(psrf_final > level, "#c44", "#48a"))
    ax[1].axhline(1.0, color="r", lw=2)
    ax[1].axhline(level, color="r", ls=":", lw=2)
    extra = names or []                                      # names after theta
    n = len(idx) - len(extra)                                # theta components
    ticks = list(range(0, n, 5)) + list(range(n, n + len(extra)))
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels([f"θ{t + 1}" for t in range(0, n, 5)] + extra,
                          rotation=60, fontsize=12)
    top = max(5.0, 1.05 * float(np.max(psrf_final)))
    ax[1].set_ylim(0, top)
    ax[1].set_xlabel("component")
    ax[1].set_ylabel(r"final $\hat{R}_i$")
    ax[1].set_title(f"final PSRF per component ({(psrf_final > level).sum()} of "
                    f"{len(psrf_final)} above {level:g})")
    for a in ax:
        style(a, 16)
    f.suptitle(title, fontsize=20)
    f.tight_layout()
    f.savefig(path, dpi=150)
    plt.close(f)


def run(exp, n_chains=None, skip=2, step=15, iter_burn=10, level=1.2, at=4900,
        dispiter=10):
    run_ = load_kept(exp, n_chains)
    sets, its, m = run_["sets"], run_["its"], run_["m"]
    thin = int(its[1] - its[0]) if len(its) > 1 else 1
    curves = {}
    fdir, tdir = io.figures_dir(exp), io.tables_dir(exp)
    print(f"{exp}: {m} chains, {len(its)} thinned samples each (thinning {thin})")
    print(f"Thinning {thin}: skip {skip}, step {step}, iter burn {iter_burn}")
    summary = dict(exp=exp, label=run_["label"], n_chains=m, n_samples=len(its),
                   thinning=thin, first_kept_iteration=int(its[0]),
                   solves_per_iteration=run_["solves"],
                   skip=skip, step=step, iter_burn=iter_burn, level=level)
    for name, arr in sets.items():
        r = arr.shape[2]
        cur = M.mpsrf_curve(arr, skip=skip, step=step, iter_burn=iter_burn,
                            dispiter=dispiter)
        if len(cur["k"]) == 0 or np.all(np.isnan(cur["mpsrf"])):
            print(f"[{name}] too few samples for a curve "
                  f"({len(its)} per chain, step {step}, iter burn {iter_burn})")
            continue
        x = its[cur["k"] - 1].astype(float) + thin      # iterations run so far
        conv = M.convergence_iteration(x, cur["mpsrf"], level)
        conv_p = M.convergence_iteration(x, cur["psrf"], level)
        j = int(np.searchsorted(x, at)) if x.max() >= at else len(x) - 1
        extra = run_["extra"][name]
        labels = ([f"θ{i + 1}" for i in range(r - len(extra))] + extra) if extra else None
        what = (f"all {r} KLE coefficients θ" if name == "theta"
                else f"(θ, {', '.join(e.replace('log', 'log ') for e in extra)}), {r} dims")
        fig_mpsrf(x, cur["mpsrf"], conv, level, f"MPSRF vs Iteration: {what}, {m} chains",
                  os.path.join(fdir, f"fig_mpsrf_{name}.png"))
        curves[name] = dict(x=x, mpsrf=cur["mpsrf"], psrf=cur["psrf"], conv=conv,
                            conv_psrf=conv_p, dim=r)
        fig_psrf(x, cur["psrf"], cur["psrf_i"][-1], conv_p, level, extra,
                 f"PSRF: {what}, {m} chains", os.path.join(fdir, f"fig_psrf_{name}.png"))
        head = "iteration,mpsrf,psrf_max," + ",".join(
            labels or [f"theta{i + 1}" for i in range(r)])
        np.savetxt(os.path.join(tdir, f"mpsrf_{name}.csv"),
                   np.column_stack([x, cur["mpsrf"], cur["psrf"], cur["psrf_i"]]),
                   delimiter=",", header=head, comments="", fmt="%.6g")
        summary[name] = dict(
            dim=r, mpsrf_convergence_iteration=conv, psrf_convergence_iteration=conv_p,
            at_iteration=float(x[j]), mpsrf_at=float(cur["mpsrf"][j]),
            psrf_max_at=float(cur["psrf"][j]), mpsrf_final=float(cur["mpsrf"][-1]),
            psrf_max_final=float(cur["psrf"][-1]),
            n_components_above_level_final=int((cur["psrf_i"][-1] > level).sum()),
            worst_components_final=[
                (labels[i] if labels else f"theta{i + 1}", round(float(cur["psrf_i"][-1][i]), 3))
                for i in np.argsort(cur["psrf_i"][-1])[::-1][:5]])
        c = "never (below the last iteration)" if conv is None else f"{int(conv)}"
        print(f"[{name}] Convergence using {m} Chains is {c}\n"
              f"[{name}] While at iteration {int(x[j])} MPSRF = {cur['mpsrf'][j]:.4f}, "
              f"max PSRF = {cur['psrf'][j]:.4f}; final MPSRF = {cur['mpsrf'][-1]:.4f}")
    io.save_json(os.path.join(tdir, "mpsrf_summary.json"), summary)
    return dict(summary=summary, curves=curves, solves=run_["solves"],
                label=run_["label"], m=m)


COLORS = {"e1_blocked": "#c0392b", "e1_global": "#e67e22", "e4_sampler": "#1f5fa8",
          "e5_soft": "#2e8b57",
          "e5_soft_ms_white": "#7b3fa0", "e5_soft_ms_raw": "#999999",
          "e5_soft_block": "#138d75", "e5_soft_block_random": "#76d7c4",
          "e5_soft_block_nocoarse": "#e87ba4", "e5_soft_block_nocoarse_dualhalo": "#4a3aa7"}


def compare(results, level=1.2):
    """All runs on the same axes, MATLAB style, x = PDE solves per chain
    (E1 blocked: 2 per iteration, E1 global: 1, E4: 3, E5: 3; E5's ridge
    move costs no solve), so equal x = equal work."""
    exp = "mpsrf_compare"
    fdir, tdir = io.figures_dir(exp), io.tables_dir(exp)
    table = {}
    for name, key, ylab, title in (
            ("theta", "mpsrf", r"$\hat{R}$", "MPSRF of all θ"),
            ("theta", "psrf", r"$\max_i \hat{R}_i$", "worst-component PSRF of θ")):
        f, ax = plt.subplots(figsize=(11, 8))
        xmax = 0.0
        for e, res in results.items():
            c = res["curves"].get(name)
            if c is None:
                continue
            xs = c["x"] * res["solves"]
            conv = c["conv"] if key == "mpsrf" else c["conv_psrf"]
            conv_s = None if conv is None else conv * res["solves"]
            lab = (f"{res['label']}, {res['m']} chains: "
                   + ("no crossing" if conv_s is None else f"{int(conv_s):,} solves")
                   + f" (final {abs(c[key][-1]):.2f})")
            col = COLORS.get(e)
            ax.plot(xs, np.abs(c[key]), lw=2.5, color=col, label=lab)
            if conv_s is not None:
                ax.plot([conv_s, conv_s], [0, level], "--", lw=3, color=col)
                ax.scatter([conv_s], [level], color=col, zorder=5, s=60)
            xmax = max(xmax, float(np.nanmax(xs)))
            table.setdefault(e, {})[key] = dict(
                convergence_iteration=conv, convergence_solves=conv_s,
                final=float(np.abs(c[key])[-1]),
                final_solves=float(xs[-1]), dim=c["dim"])
        ax.axhline(1.0, color="r", ls="-", lw=3, label="y=1.0")
        ax.axhline(level, color="r", ls=":", lw=3, label=f"y={level:g}")
        ax.set_ylim(0, 5)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("PDE solves per chain")
        ax.set_ylabel(ylab)
        ax.set_title(f"{title}: all runs at equal work")
        ax.legend(fontsize=15, loc="upper right")
        style(ax)
        f.tight_layout()
        path = os.path.join(fdir, f"fig_{key}_compare_theta.png")
        f.savefig(path, dpi=150)
        plt.close(f)
        print(f"  wrote {os.path.relpath(path, io.root())}")
    io.save_json(os.path.join(tdir, "mpsrf_compare.json"), table)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=None,
                    help="run folder, e.g. e1_blocked, e1_global, e4_sampler "
                         "(default: every run present, plus the E1-vs-E4 comparison)")
    ap.add_argument("--chains", type=int, default=None, help="use the first N chains")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--iter-burn", type=int, default=10)
    ap.add_argument("--level", type=float, default=1.2)
    ap.add_argument("--at", type=float, default=4900,
                    help="also report the values at this iteration")
    a = ap.parse_args(argv)
    exps = [a.exp] if a.exp else [
        e for e in (io.exp_e1("blocked"), io.exp_e1("global"), io.EXP_SAMPLER, io.EXP_SOFT,
                    io.exp_soft(False, "ms_white"), io.exp_soft(False, "ms_raw"),
                    io.exp_soft(False, "block"), io.exp_soft(False, "block_random"))
        if io.chain_files(io.results_dir(e))]
    # tagged variants (e5_soft_block_<tag>, ...): any other e5_soft_* run with chains
    rdir = os.path.join(io.root(), "results")
    if not a.exp and os.path.isdir(rdir):
        exps += sorted(e for e in os.listdir(rdir) if e.startswith(io.EXP_SOFT + "_")
                       and e not in exps and io.chain_files(os.path.join(rdir, e)))
    results = {}
    for e in exps:
        results[e] = run(e, a.chains, a.skip, a.step, a.iter_burn, a.level, a.at)
    if not a.exp and len(results) > 1:
        compare(results, a.level)


if __name__ == "__main__":
    main()
