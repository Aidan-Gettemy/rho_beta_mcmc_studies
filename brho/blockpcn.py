"""
Block (subdomain) pCN for theta: Metropolis-within-Gibbs over local
subspaces of theta-space, with a primal and a shifted (dual) subdomain
decomposition used in alternation (E8; the advisor's subdomain-sampling idea).

Why theta-space and not cell space.  The prior is a rank-r Gaussian
(G = T theta, r = 57 in 2D, 78 in 3D), so the field inside a block is almost
determined by the field outside it: the cell-space conditional of a block
given its complement has ~zero variance.  Block moves therefore act on
directions of theta.

Local subspace of block k (cells Omega_k, optional halo h):
    generalized eigenproblem   (T_k^T T_k) v = mu (T^T T) v,
    mu(v) = |T v|^2 on Omega_k / |T v|^2 on the domain   (fraction of the
    field energy of direction v that lies inside the block), mu in [0, 1].
    Keep mu >= mu* (`energy`); V_k = orthonormal basis (QR) of their span.
    Every field change made by the block move is a combination of the global
    (smooth) KLE modes, so nothing can jump at an interface, and at most
    (1 - mu*) of its energy leaks outside the block.
Coarse subspace: V_0 = orthogonal complement of the span of the PRIMAL local
    subspaces (the directions that straddle the primal interfaces); with it,
    coarse + primal span all of R^r.
Dual decomposition: interfaces shifted by half a block (2 x 2 primal -> 3 x 3
    dual with interfaces at 0.25 and 0.75); its blocks sit on top of the
    primal interfaces, so cells there move under local moves too.

The move on subspace V (orthonormal columns, dimension k):
    alpha = V^T theta,  alpha' = sqrt(1 - s^2) alpha + s xi,  xi ~ N(0, I_k),
    theta' = theta + V (alpha' - alpha),
    accept with min(1, exp(Phi(theta) - Phi(theta'))).
Under theta ~ N(0, I_r), alpha ~ N(0, I_k) is independent of the rest of
theta, so this is pCN on alpha with the complement held fixed: it leaves the
prior invariant EXACTLY and the likelihood-only ratio is the full MH ratio
(no whitening needed).  Its law depends only on the span of V (not on the
basis QR returns).  A fixed sweep (coarse, primal 1..P, dual 1..D) or a random
choice of block composes / mixes invariant kernels, so it is invariant too.
Cost: one PDE solve per block move (the solve is still global).
"""
import numpy as np
from scipy import linalg as SL

SCHEDULES = ("sweep", "random")


def _edges(blocks, dual):
    """Subdomain interfaces on [0, 1] per axis: primal i/b, dual (i + 1/2)/b."""
    if dual:
        inner = (np.arange(blocks) + 0.5) / blocks
    else:
        inner = np.arange(1, blocks) / blocks
    return np.r_[0.0, inner, 1.0]


def boxes(dim, blocks, dual):
    """[(lo, hi)] boxes (each lo, hi of length dim) of the decomposition,
    in the same lexicographic order as multiscale.block_index."""
    e = _edges(blocks, dual)
    m = len(e) - 1
    out = []
    for flat in range(m ** dim):
        ij, f = [], flat
        for _ in range(dim):
            ij.append(f % m)
            f //= m
        ij = ij[::-1]
        out.append((np.array([e[i] for i in ij]), np.array([e[i + 1] for i in ij])))
    return out


def box_mask(points, lo, hi, halo=0.0):
    """Cells (centres, strictly inside the unit cube) in the box widened by halo."""
    P = np.asarray(points, float)
    return np.all((P >= lo - halo) & (P < hi + halo), axis=1)


def local_subspace(T, mask, energy, A=None):
    """Orthonormal basis (r x k) of the theta-directions whose field has at
    least `energy` of its energy on the cells in `mask`; also returns the
    kept generalized eigenvalues mu (descending)."""
    if A is None:
        A = T.T @ T
    Tk = T[mask]
    mu, V = SL.eigh(Tk.T @ Tk, A)                    # ascending, A-orthonormal
    mu = np.clip(mu[::-1], 0.0, 1.0)
    V = V[:, ::-1]
    keep = mu >= energy
    if not keep.any():
        return np.zeros((T.shape[1], 0)), mu[:0]
    Q, _ = np.linalg.qr(V[:, keep])
    return Q, mu[keep]


class BlockPCN:
    """Local (and coarse) subspaces of theta for a blocks^dim primal
    decomposition and, if `dual`, the (blocks+1)^dim shifted one.

    moves[j] = dict(name, kind in {coarse, primal, dual}, V (r x k), dim k,
    mu (kept energy fractions; None for coarse), mask (cells of the block));
    moves[0] is the coarse move; blocks with an empty local subspace get no
    move (summary()['primal_dims'] / ['dual_dims'] still list them, as 0)."""

    def __init__(self, kle, points, blocks=2, energy=0.9, dual=True, halo=0.0):
        T = np.asarray(kle.T, float)
        self.T, self.r = T, T.shape[1]
        P = np.asarray(points, float)
        self.dim = P.shape[1]
        self.blocks, self.energy, self.dual, self.halo = int(blocks), float(energy), \
            bool(dual), float(halo)
        A = T.T @ T
        self.moves = []
        prim = []
        for k, (lo, hi) in enumerate(boxes(self.dim, self.blocks, False)):
            m = box_mask(P, lo, hi, self.halo)
            V, mu = local_subspace(T, m, self.energy, A)
            prim.append(V)
            self.moves.append(dict(name=f"P{k + 1}", kind="primal", V=V, dim=V.shape[1],
                                   mu=mu, mask=m))
        if self.dual:
            for k, (lo, hi) in enumerate(boxes(self.dim, self.blocks, True)):
                m = box_mask(P, lo, hi, self.halo)
                V, mu = local_subspace(T, m, self.energy, A)
                self.moves.append(dict(name=f"Q{k + 1}", kind="dual", V=V, dim=V.shape[1],
                                       mu=mu, mask=m))
        # blocks with an empty local subspace (no direction keeps `energy` in
        # it, e.g. small 3D dual blocks at energy 0.9) are listed in the
        # summary but get no move (no wasted solve)
        self.all_dims = {k: [mv["dim"] for mv in self.moves if mv["kind"] == k]
                         for k in ("primal", "dual")}
        self.moves = [mv for mv in self.moves if mv["dim"] > 0]
        Vp = np.hstack(prim)
        V0 = SL.null_space(Vp.T) if Vp.shape[1] else np.eye(self.r)
        self.moves.insert(0, dict(name="coarse", kind="coarse", V=V0, dim=V0.shape[1],
                                  mu=None, mask=None))
        self.span_rank = int(np.linalg.matrix_rank(np.hstack([V0, Vp])))
        self.n_local = len(self.moves) - 1
        self.target = (T ** 2).sum(1)                 # exact pointwise prior variance

    # ------------------------------------------------------------------ move
    def propose(self, theta, j, s, rng):
        """pCN proposal on subspace j (0 = coarse); None if it is empty."""
        V = self.moves[j]["V"]
        if V.shape[1] == 0:
            return None
        a = V.T @ theta
        return theta + V @ ((np.sqrt(1.0 - s * s) - 1.0) * a
                            + s * rng.standard_normal(V.shape[1]))

    def order(self, schedule, rng):
        """Moves of one theta update: 'sweep' = coarse, primal 1..P, dual
        1..D; 'random' = coarse and ONE local block chosen uniformly."""
        if schedule == "sweep":
            return range(len(self.moves))
        if schedule == "random":
            return (0, 1 + int(rng.integers(self.n_local)))
        raise ValueError(f"unknown block schedule {schedule!r}; have {SCHEDULES}")

    def solves(self, schedule):
        return len(self.moves) if schedule == "sweep" else 2

    # ----------------------------------------------------------- diagnostics
    def reach(self, kinds):
        """Per cell: prior variance the moves of these kinds can change, over
        the cell's prior variance (summed over moves, so overlap counts twice)."""
        tot = np.zeros(self.T.shape[0])
        for mv in self.moves:
            if mv["kind"] in kinds and mv["dim"]:
                tot += ((self.T @ mv["V"]) ** 2).sum(1)
        return tot / self.target

    def leak(self):
        """Worst fraction of field energy outside its block, per local move."""
        return {mv["name"]: float(1.0 - mv["mu"].min()) if mv["dim"] else None
                for mv in self.moves[1:]}

    def summary(self):
        dims = self.all_dims
        return dict(blocks=self.blocks, dim=self.dim, energy=self.energy, dual=self.dual,
                    halo=self.halo, r=self.r, coarse_dim=self.moves[0]["dim"],
                    primal_dims=dims["primal"], dual_dims=dims["dual"],
                    primal_total=int(sum(dims["primal"])), dual_total=int(sum(dims["dual"])),
                    n_moves=len(self.moves),
                    empty_blocks=sum(d == 0 for v in dims.values() for d in v),
                    span_rank=self.span_rank,
                    spans=self.span_rank == self.r)


def from_problem(prob, blocks=2, energy=0.9, dual=True, halo=0.0):
    return BlockPCN(prob.kle, prob.phys.points, blocks, energy, dual, halo)
