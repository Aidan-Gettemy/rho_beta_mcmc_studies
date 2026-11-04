"""Spectrum of the round-trip map theta -> B_g Pi_g theta (r x r) for the primal and dual grids:
what experiment 4 (mixture through the global state) applies at every accepted move.
A projection would have eigenvalues 0/1 only; values strictly in between contract.
  python experiments/msm_roundtrip_spectrum.py   ->  tables/msm_round2/roundtrip_spectrum.json,
                                                      runs/msm_local_kle_2026-11/compare2/fig_roundtrip_spectrum.png"""
import _common  # noqa: F401
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brho import msm_local as ML
S = ML.MSMSetup(); out = {}
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for nm, G, c in (("primal", S.primal, "C3"), ("dual", S.dual, "C0")):
    A = G.B @ G.Pi
    ev = np.sort(np.linalg.eigvals(A).real)[::-1]; sv = np.linalg.svd(A, compute_uv=False)
    out[nm] = dict(p=int(G.p), eig_max=float(ev.max()), eig_min=float(ev.min()), eig_mean=float(ev.mean()),
                   frac_eig_below_0p95=float((ev < .95).mean()), sv_max=float(sv.max()), sv_min=float(sv.min()),
                   n_eig_below_0p01=int((ev < .01).sum()))
    ax.plot(ev, c, label=f"{nm} (mean {ev.mean():.2f})")
ax.axhline(1, color="k", ls=":"); ax.set_xlabel("index (sorted)"); ax.set_ylabel(r"eigenvalue of $B_g\Pi_g$")
ax.legend(); ax.set_title("round-trip map of experiment 4"); fig.tight_layout()
os.makedirs("tables/msm_round2", exist_ok=True)
fig.savefig("runs/msm_local_kle_2026-11/compare2/fig_roundtrip_spectrum.png", dpi=150)
json.dump(out, open("tables/msm_round2/roundtrip_spectrum.json", "w"), indent=1); print(json.dumps(out, indent=1))
