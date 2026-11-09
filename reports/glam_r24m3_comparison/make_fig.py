"""Archive/best-Phi figure for the four-run comparison (reads exchanges.csv of r24, r24m, r24m2, r24m3 from the repo tables)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
T = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "cluster", "tables")
runs = {"r24": "tab:blue", "r24m": "tab:orange", "r24m2": "tab:green", "r24m3": "tab:red"}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for t, c in runs.items():
    e = pd.read_csv(os.path.join(T, f"e6_glam_{t}", "exchanges.csv"))
    ax[0].plot(e.iteration, e.phi_best, c=c, label=t)
    ax[1].plot(e.iteration, e.anchors, c=c, label=t)
    if t != "r24":
        ax[2].plot(e.iteration, e.eff_rank, c=c, label=t)
ax[0].set_yscale("log"); ax[0].set_ylabel(r"best anchor $\Phi$"); ax[1].set_yscale("log"); ax[1].set_ylabel("anchors")
ax[2].set_ylabel("effective rank of the anchors"); ax[2].set_yscale("log")
for a in ax: a.set_xlabel("iteration"); a.grid(alpha=0.3, which="both")
ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "fig_archive.png"), dpi=150)
