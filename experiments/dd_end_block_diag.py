#!/usr/bin/env python3
"""dd_end diagnostic: why do the chains of one DD block disagree?   (needs chains.npz, so run it on the cluster)

    python experiments/dd_end_block_diag.py --dir RUN_DIR --blocks 21,50,52 [--out DIR]

For each block: per-chain mean of the local coefficients over the second half of the run, the between-chain spread against the
within-chain sd, the drift of each chain (second-half mean minus first-half mean, in units of the between-chain spread), a 2-means
split of the chain means (two modes?) and its gap, and two figures (traces of the leading coefficient for every chain; scatter of
the chain means on their two leading principal axes).  Prints a short verdict: modes (chains sit apart and do not drift) or slow
mixing (chains drift towards each other).  Small output, safe to commit (diag_block<k>.png and diag.txt)."""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def two_means(x, it=50):
    c = x[[np.argmin(x[:, 0]), np.argmax(x[:, 0])]].copy()
    for _ in range(it):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        for j in (0, 1):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    return lab, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "diag")
    os.makedirs(out, exist_ok=True)
    Z = np.load(os.path.join(a.dir, "chains.npz"))
    lines = []
    for k in [int(s) for s in a.blocks.split(",")]:
        C = Z[f"c_{k:02d}"] if f"c_{k:02d}" in Z.files else Z[[f for f in Z.files if f.endswith(f"_{k:02d}") or f == f"c_{k}"][0]]
        m, n, r = C.shape
        h = n // 2
        A, B = C[:, :h], C[:, h:]
        mean2 = B.mean(1)                                    # (chains, r) second-half means
        mean1 = A.mean(1)
        within = B.std(1).mean(0)                            # within-chain sd per coefficient
        between = mean2.std(0)                               # spread of chain means per coefficient
        ratio = between / np.maximum(within, 1e-12)
        lab, cen = two_means(mean2)
        n1 = int(lab.sum())
        sep = cen[1] - cen[0]
        dist = float(np.linalg.norm(sep))
        u = sep / max(dist, 1e-12)
        wsd = float(np.mean([(B[i] @ u).std() for i in range(m)]))          # within-chain sd along the separating direction
        gap = dist / max(wsd, 1e-12)
        drift = np.abs((mean2 - mean1) @ u) / max(dist, 1e-12)                 # movement along it, as a fraction of the gap
        u_, s_, vt = np.linalg.svd(mean2 - mean2.mean(0), full_matrices=False)
        pc = (mean2 - mean2.mean(0)) @ vt[:2].T
        lines.append(f"block {k}: dim {r}, chains {m}, saved states {n}")
        lines.append(f"  between/within sd per coefficient: " + " ".join(f"{x:.2f}" for x in ratio))
        lines.append(f"  2-means split of chain means: {m - n1} + {n1} chains, gap {gap:.1f} within-chain sds along the separating direction")
        lines.append(f"  movement of chain means along it between the two halves: median {np.median(drift):.2f} of the gap, max {drift.max():.2f}")
        modes = gap > 3 and min(n1, m - n1) >= 2 and np.median(drift) < 0.1
        lines.append("  verdict: " + ("chains sit in separate modes and do not move between halves: a multimodal block posterior" if modes else
                                      "no clear split into modes: look at the traces (slow mixing or a few stuck chains)"))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        lead = int(np.argmax(ratio))
        for i in range(m):
            ax[0].plot(np.arange(n), C[i, :, lead], lw=0.6)
        ax[0].set_title(f"block {k}: coefficient {lead}, every chain"); ax[0].set_xlabel("saved state")
        ax[1].scatter(pc[:, 0], pc[:, 1], c=lab, cmap="jet", s=25)
        ax[1].set_title("chain means (2nd half), two leading axes"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
        fig.savefig(os.path.join(out, f"diag_block{k}.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    open(os.path.join(out, "diag.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
