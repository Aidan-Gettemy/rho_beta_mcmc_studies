"""
Scale-block pCN for theta: move A on blocks of KLE modes by index (scale).

Why (reports/linearized/spectral_split.py, why_blocks.py): how strongly the
pressure data constrain a KLE mode falls steeply with its index (r = 208:
mean diag H 879 for modes 1-20, 2.3 for modes 151-208).  Global pCN uses one
step for all of theta, set by the stiff large-scale modes, so the weakly
informed fine-scale modes crawl.  Blocks by index separate stiff from soft
directions (spatial blocks cannot: every subdomain holds both), so each block
takes its own adapted step, and the stiff block, whose directions need little
movement, can be visited less often.  Surrogate (r = 208, per theta-solve):
1.7-1.8x on the slowest direction, 2.4-3.1x on the median.

The move on block j (coordinates idx_j of theta):
    theta'[idx_j] = sqrt(1 - s_j^2) theta[idx_j] + s_j xi,   xi ~ N(0, I),
    the other coordinates unchanged;  accept with min(1, exp(Phi - Phi')).
Under theta ~ N(0, I_r) the coordinates are independent, so this is pCN on
the block with the rest fixed: it leaves the prior invariant EXACTLY, and the
likelihood ratio is the full MH ratio.  The block is drawn at random with
fixed probabilities (a mixture of invariant kernels: invariant).  Gradient
free; the blocks come from the prior alone.  One PDE solve per move.

    edges "20,60": blocks modes 1-20, 21-60, 61-r (1-based); edges >= r are
    dropped, so the same physical split works at r = 57 (1-20, 21-57).
"""
import numpy as np


def parse_list(text, typ=float):
    text = (text or "").strip()
    return [typ(x) for x in text.replace(";", ",").split(",") if x.strip()] if text else []


class ScaleBlocks:
    def __init__(self, r, edges="20,60", probs=""):
        self.r = int(r)
        cuts = sorted({int(e) for e in parse_list(edges, int) if 0 < int(e) < self.r})
        bounds = [0] + cuts + [self.r]
        self.blocks = [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
        self.ranges = [(a + 1, b) for a, b in zip(bounds[:-1], bounds[1:])]   # 1-based, inclusive
        p = parse_list(probs)
        if p:
            if len(p) != len(self.blocks):
                raise ValueError(f"scale_probs has {len(p)} entries for {len(self.blocks)} "
                                 f"blocks {self.names()}")
            p = np.asarray(p, float)
            if np.any(p <= 0):
                raise ValueError("scale_probs must be positive")
            self.p = p / p.sum()
        else:
            self.p = np.full(len(self.blocks), 1.0 / len(self.blocks))
        self.n = len(self.blocks)

    def names(self):
        return [f"modes {a}-{b}" for a, b in self.ranges]

    def choose(self, rng):
        return int(rng.choice(self.n, p=self.p)) if self.n > 1 else 0

    def propose(self, theta, j, s, rng):
        idx = self.blocks[j]
        th = theta.copy()
        th[idx] = np.sqrt(1.0 - s * s) * theta[idx] + s * rng.standard_normal(idx.size)
        return th

    def summary(self):
        return dict(r=self.r, blocks=self.ranges, dims=[int(b.size) for b in self.blocks],
                    probs=[float(x) for x in self.p])
