"""Stage 9 (replication of Xu, Liao & Li 2022): steady diffusion on the strip D = (0,3) x (0,1).

    -div(a grad u) = f   in D,   u = 0 on x = 0 and x = 3,   a du/dn = 0 on y = 0 and y = 1,
    f(x) = 3 exp(-|x - xs|^2),  xs = (1.5, 0.5).

Cell-centred finite volumes on an (nx, ny) = (3m, m) grid of square cells h = 1/m (m = 32: 96 x 32), harmonic face
transmissibilities, half-cell Dirichlet faces on the left/right ends, source integrated over each cell by the midpoint rule.
Cells are ordered C-style, index = ix * ny + iy (the same convention as brho.fv).

Prior (ours, not the paper's): log a = sd * sum_k sqrt(lam_k) phi_k theta_k, theta ~ N(0, I), with the paper's separable exponential
covariance exp(-|x1 - y1|/L - |x2 - y2|/L) (unit variance) and sd = 0.25.  The paper uses a = 1 + KLE with uniform [-1, 1] coefficients
(effective sd about 0.14).  KLE modes are the discrete (cell-centre) tensor-product modes, truncated to the number the paper reports.

Data: 161 sensors on x1 = 0.125 i (i = 1..23), x2 = 0.125 j (j = 1..7).  With m a multiple of 8 every sensor sits on a cell corner
and reads the average of the four surrounding cell values.  Noise: i.i.d. Gaussian, sd = 1% of the mean observed value.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LX, LY = 3.0, 1.0
SOURCE_C = (1.5, 0.5)
PAPER_KLE = {2.0: (27, 11), 1.0: (87, 31), 0.5: (307, 109)}   # (global modes, local modes of a 1 x 1 strip); L = 1: local 31 (extraction read 3)


class StripForward:
    def __init__(self, m=32, source_amp=3.0):
        self.m, self.nx, self.ny, self.h = m, 3 * m, m, 1.0 / m
        nx, ny, N = self.nx, self.ny, self.nx * self.ny
        self.N = N
        idx = np.arange(N).reshape(nx, ny)
        self.ax, self.bx = idx[:-1].ravel(), idx[1:].ravel()           # x-neighbours
        self.ay, self.by = idx[:, :-1].ravel(), idx[:, 1:].ravel()     # y-neighbours
        rows = np.concatenate([self.ax, self.bx, self.ay, self.by, np.arange(N)])
        cols = np.concatenate([self.bx, self.ax, self.by, self.ay, np.arange(N)])
        coo = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsc()
        self._rows, self._cols = rows, cols
        self.xc = (np.arange(nx) + 0.5) * self.h
        self.yc = (np.arange(ny) + 0.5) * self.h
        X, Y = np.meshgrid(self.xc, self.yc, indexing="ij")
        self.X, self.Y = X.ravel(), Y.ravel()
        f = source_amp * np.exp(-((self.X - SOURCE_C[0]) ** 2 + (self.Y - SOURCE_C[1]) ** 2))
        self.b = f * self.h ** 2
        self.left = idx[0, :]; self.right = idx[-1, :]

    def matrix(self, a):
        a = np.asarray(a, float).ravel()
        Tx = 2 * a[self.ax] * a[self.bx] / (a[self.ax] + a[self.bx])
        Ty = 2 * a[self.ay] * a[self.by] / (a[self.ay] + a[self.by])
        diag = (np.bincount(self.ax, Tx, self.N) + np.bincount(self.bx, Tx, self.N) +
                np.bincount(self.ay, Ty, self.N) + np.bincount(self.by, Ty, self.N))
        diag[self.left] += 2 * a[self.left]; diag[self.right] += 2 * a[self.right]     # half-cell to u = 0 (T = (h/(h/2)) a)
        vals = np.concatenate([-Tx, -Tx, -Ty, -Ty, diag])
        return sp.csc_matrix((vals, (self._rows, self._cols)), shape=(self.N, self.N))

    def solve(self, a):
        return spla.splu(self.matrix(a), permc_spec="MMD_AT_PLUS_A").solve(self.b)

    def outflow(self, a, u):
        """Total flux leaving through x = 0 and x = 3 (equals the integral of f)."""
        a = np.asarray(a, float).ravel()
        return float(np.sum(2 * a[self.left] * u[self.left]) + np.sum(2 * a[self.right] * u[self.right]))


def sensor_operator(fw):
    """161 x N averaging operator: x1 = 0.125 i, x2 = 0.125 j (corner of 4 cells when m is a multiple of 8)."""
    assert fw.m % 8 == 0
    s = fw.m // 8
    rows, cols = [], []
    k = 0
    for i in range(1, 24):
        for j in range(1, 8):
            ix, iy = i * s, j * s                      # the corner is shared by cells (ix-1, ix) x (iy-1, iy)
            for dx in (-1, 0):
                for dy in (-1, 0):
                    rows.append(k); cols.append((ix + dx) * fw.ny + iy + dy)
            k += 1
    return sp.csr_matrix((np.full(len(rows), 0.25), (rows, cols)), shape=(k, fw.N))


def build_kle(fw, L, n_modes, sd=0.25):
    """Synthesis matrix T (N x r) with log a = T theta; unit-variance separable exponential kernel on the cell centres."""
    def eig(x):
        C = np.exp(-np.abs(x[:, None] - x[None, :]) / L)
        w, v = np.linalg.eigh(C)
        return w[::-1], v[:, ::-1]
    wx, vx = eig(fw.xc); wy, vy = eig(fw.yc)
    lam = np.outer(wx, wy).ravel()
    order = np.argsort(lam)[::-1][:n_modes]
    ix, iy = np.unravel_index(order, (fw.nx, fw.ny))
    phi = vx[:, ix][:, None, :] * vy[:, iy][None, :, :]           # (nx, ny, r)
    T = sd * phi.reshape(fw.N, n_modes) * np.sqrt(lam[order])
    energy = lam[order].sum() / lam.sum()
    return T, energy


class StripProblem:
    def __init__(self, L=1.0, m=32, n_modes=None, sd=0.25, noise_rel=0.01, truth_seed=2026):
        self.L = L
        self.fw = StripForward(m)
        self.H = sensor_operator(self.fw)
        n_modes = n_modes or PAPER_KLE[L][0]
        self.T, self.energy = build_kle(self.fw, L, n_modes, sd)
        self.r = n_modes
        rng = np.random.default_rng(truth_seed)
        self.theta_true = rng.standard_normal(self.r)
        self.logk_true = self.T @ self.theta_true
        self.u_true = self.fw.solve(np.exp(self.logk_true))
        clean = self.H @ self.u_true
        self.sigma = noise_rel * float(np.mean(np.abs(clean)))
        self.d = clean + self.sigma * rng.standard_normal(len(clean))

    def potential(self, theta):
        u = self.fw.solve(np.exp(self.T @ theta))
        return 0.5 * float(np.sum((self.d - self.H @ u) ** 2)) / self.sigma ** 2


def run_pcn(pb, seed, n_iter, n_adapt, s0=0.1, target=0.25, every=100, save_every=10):
    from .samplers import StepAdapter
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    ad = StepAdapter([s0], targets=[target], every=every)
    th = rng.standard_normal(pb.r)
    phi = pb.potential(th)
    out = np.empty((n_iter // save_every, pb.r), np.float32)
    acc = 0
    for it in range(n_iter):
        if it % save_every == 0:
            out[it // save_every] = th
        s = ad.s[0]
        new = np.sqrt(1 - s * s) * th + s * rng.standard_normal(pb.r)
        pn = pb.potential(new)
        ok = np.log(rng.random()) < phi - pn
        if ok:
            th, phi = new, pn
        ad.record(0, ok); ad.tick(it, n_adapt)
        if it >= n_adapt:
            acc += ok
    return out, acc / max(n_iter - n_adapt, 1), float(ad.s[0])
