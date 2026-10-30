import csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T="../../runs/cluster/tables/e6_glam_%s/exchanges.csv"
def load(t):
    R=list(csv.DictReader(open(T%t))); return {k:np.array([float(r[k]) for r in R]) for k in R[0] if k not in("phase","best_role")}
a,m=load("r24"),load("r24m")
fig,ax=plt.subplots(1,3,figsize=(11,3.2))
ax[0].plot(m["iteration"],m["phi_best"],label="r24m best anchor"); ax[0].plot(m["iteration"],m["phi_med"],"--",label="r24m median anchor",color="C0",alpha=.6)
ax[0].plot(a["iteration"],a["phi_best"],label="r24 anchor (single)",color="C3"); ax[0].set_yscale("log"); ax[0].set_xlabel("iteration"); ax[0].set_ylabel(r"$\Phi$ of anchors"); ax[0].legend(fontsize=7)
ax[0].set_ylim(650,3000)
ax[1].plot(m["iteration"],m["anchors"],color="C0"); ax[1].set_xlabel("iteration"); ax[1].set_ylabel("anchors in archive (r24m)")
ax[2].plot(m["iteration"],m["eff_rank"],color="C0"); ax[2].set_xlabel("iteration"); ax[2].set_ylabel("effective rank of anchors (r24m)")
fig.tight_layout(); fig.savefig("fig_exchange.png",dpi=160)
