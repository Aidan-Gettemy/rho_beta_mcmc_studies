"""Figures and tables of the chain-level analysis (variance by coefficient index, anchor Phi over exchanges, leakage) from the returned tables.
    python chain_figs.py        (from this folder; writes fig_var_index.png, fig_exchange_phi.png, tables_chain_var.tex, tables_chain_leak.tex)"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TAB = os.path.join(ROOT, "runs", "cluster", "tables")
ARMS = [("1", "pCN", "s80k_pcn", "#2a78d6", "o"), ("2", "one anchor", "s80k_glam1", "#eb6834", "s"),
        ("2b", "one anchor + memory", "s80k_glam1_best", "#1baf7a", "^"), ("3", "many anchors", "s80k_glamN", "#eda100", "D"),
        ("4", "one anchor + mask", "s80k_glam1_mask", "#e87ba4", "v"), ("5", "many anchors + mask", "s80k_glamN_mask", "#008300", "P")]
INK, GRID = "#333333", "#dddddd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})
N_SUB, M = 12, 500


def rd(tag, name):
    return list(csv.DictReader(open(os.path.join(TAB, f"e6_glam_{tag}", name))))


# ---- variance by coefficient index (mean over the 12 blocks of the pooled variance at each index)
mask = np.load(os.path.join(TAB, "e6_glam_s80k_pcn", "keep_mask_12chains.npz"))
K = np.array([int((~mask[f"m{b}"]).sum()) for b in range(N_SUB)])               # coefficients installed from the anchor in block b
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for a, nm, tag, col, mk in ARMS:
    R = rd(tag, "variance_by_index.csv")
    v = np.array([float(r["variance"]) for r in R if int(r["block"]) < N_SUB]).reshape(N_SUB, M).mean(0)
    sm = np.convolve(np.pad(v, (7, 7), mode="edge"), np.ones(15) / 15, mode="valid")
    ax.plot(sm, color=col, lw=1.6, label=f"{a} {nm}")
ax.axhline(1.0, color=INK, ls=":", lw=1); ax.text(M - 2, 1.02, "prior", ha="right", va="bottom", color=INK, fontsize=8)
ax.axvline(K.mean(), color="#888888", ls="--", lw=1)
ax.text(K.mean() + 4, 0.14, f"mean K = {K.mean():.0f}  (per block {K.min()}--{K.max()}):\nleft of the line the mask installs the anchor", color="#555555", fontsize=8, va="bottom")
ax.set_xlabel("coefficient index within a block"); ax.set_ylabel("pooled tail variance, mean over the 12 blocks")
ax.set_ylim(0, 1.12); ax.set_xlim(0, M); ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
fig.tight_layout(); fig.savefig("fig_var_index.png", dpi=200); plt.close(fig)

# ---- Phi over exchanges
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1.15, 1]))
X = {}
for a, nm, tag, col, mk in ARMS[1:]:
    R = rd(tag, "exchange_phi.csv"); X[tag] = R
    it = np.array([int(r["iteration"]) for r in R]); rm = np.array([float(r["restart_median"]) for r in R])
    ax[0].plot(it, rm, color=col, lw=1.3, label=f"{a} {nm}")
    s = it >= 80000
    ax[1].plot(it[s], rm[s], color=col, lw=1.4, marker=mk, ms=3.5, label=f"{a}")
    ab = np.array([float(r["anchor_phi_best"]) for r in R])
    ax[1].plot(it[s], ab[s], color=col, lw=1.0, ls="--")
ax[0].set_yscale("log"); ax[0].set_ylim(600, 20000)
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("median over chains of $\\Phi$ of the restart state"); ax[0].grid(color=GRID, lw=0.6)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper right")
ax[1].set_xlabel("iteration (last 20 000)"); ax[1].set_ylabel("$\\Phi$"); ax[1].grid(color=GRID, lw=0.6)
ax[1].set_title("solid: restart state;  dashed: best anchor", fontsize=8, color=INK)
fig.tight_layout(); fig.savefig("fig_exchange_phi.png", dpi=200); plt.close(fig)

# ---- tables
V = {tag: json.load(open(os.path.join(TAB, f"e6_glam_{tag}", "variance_summary.json"))) for _, _, tag, _, _ in ARMS}
p = V["s80k_pcn"]["overall"]
with open("tables_chain_var.tex", "w") as fh:
    fh.write("\\begin{tabular}{llcccc}\\toprule\n arm & & \\multicolumn{2}{c}{constrained (2109 coordinates)} & \\multicolumn{2}{c}{unconstrained (3891)} \\\\\n"
             " & & variance & ratio to pCN & variance & ratio to pCN \\\\ \\midrule\n")
    for a, nm, tag, col, mk in ARMS:
        o = V[tag]["overall"]
        c, u = o["constrained"]["variance_mean"], o["unconstrained"]["variance_mean"]
        fh.write(f"{a} & {nm} & {c:.3f} & {c / p['constrained']['variance_mean']:.2f} & {u:.3f} & {u / p['unconstrained']['variance_mean']:.2f} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\n")

with open("tables_chain_leak.tex", "w") as fh:
    fh.write("\\begin{tabular}{llccccc}\\toprule\n arm & & best anchor $\\Phi$ & restart $\\Phi$ & restart $-$ anchor & (\\%) & restart $-$ before restart \\\\ \\midrule\n")
    for a, nm, tag, col, mk in ARMS[1:]:
        R = [r for r in X[tag] if int(r["iteration"]) >= 80000]
        ab = np.mean([float(r["anchor_phi_best"]) for r in R]); rs = np.mean([float(r["restart_median"]) for r in R]); pr = np.mean([float(r["pre_median"]) for r in R])
        fh.write(f"{a} & {nm} & {ab:.0f} & {rs:.0f} & {rs - ab:+.0f} & {100 * (rs - ab) / ab:.1f} & {rs - pr:+.0f} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\n")
print(open("tables_chain_var.tex").read()); print(open("tables_chain_leak.tex").read(), "n exchanges per arm from 80000:", len([r for r in X["s80k_glam1"] if int(r["iteration"]) >= 80000]))
