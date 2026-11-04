"""
Export the global, primal and dual KLE bases of the MSM study as plain-text matrices for MATLAB.

    python experiments/export_msm_kle_txt.py --out <dir> [--primal-dims 4 6 8]
Writes (whitespace-delimited, %.10e) under <dir>/ : README.txt, load_msm_kle.m and
    cell_centers.txt            (N x 2)  x, y of every cell centre, N = 1024
    kle_global_T.txt            (N x 64) field = T * theta, T = Phi * sqrt(lambda)
    kle_global_Phi.txt          (N x 64) orthonormal eigenvectors;  kle_global_lambda.txt (64 x 1)
    <grid>_block_index.txt      (N x 1)  block label of every cell, 1-based
    <grid>_local_modes.txt      (N x p)  column (k,j): zero-extended sqrt(mu_kj) * psi_kj; piecewise field = local_modes * c
    <grid>_local_index.txt      (p x 3)  block, mode, mu
    <grid>_B.txt                (64 x p) c -> theta;  <grid>_Pi.txt (p x 64) theta -> c;  <grid>_Sigma.txt = B B'
    <grid>_assembled_modes.txt  (N x p)  T * B: the field of each local mode after projection onto the global KLE
primal4 also: primal4_null_V0.txt (64 x 6) and primal4_null_fields.txt (N x 6) = T * V0.
Grids: primal4 (4 modes/block, the study), and primalD for each extra --primal-dims, dual (5x5, 95% energy).
Cell index m = ix*n + iy (ix = x index, slow); in MATLAB  A = reshape(f, 32, 32)  has A(iy, ix).
"""
import _common  # noqa: F401
import argparse
import os

import numpy as np

from brho import msm_local as ML


def w(path, a):
    np.savetxt(path, np.atleast_2d(a) if np.ndim(a) else np.atleast_2d(a), fmt="%.10e")


def local_modes_full(ms, N):
    E = np.zeros((N, ms.p)); off = 0
    for idx, psi, mu in zip(ms.block_cells, ms.block_psi, ms.block_mu):
        n_k = len(mu); E[idx, off:off + n_k] = psi * np.sqrt(mu); off += n_k
    return E


def block_index(ms, N):
    b = np.zeros(N, int)
    for k, idx in enumerate(ms.block_cells): b[idx] = k + 1
    return b


def dump_grid(out, name, ms, N, T):
    E = local_modes_full(ms, N)
    li = np.array([(k + 1, j + 1, mu) for k, m in enumerate(ms.block_mu) for j, mu in enumerate(m)])
    w(f"{out}/{name}_block_index.txt", block_index(ms, N)[:, None].astype(float))
    w(f"{out}/{name}_local_modes.txt", E)
    w(f"{out}/{name}_local_index.txt", li)
    w(f"{out}/{name}_B.txt", ms.B); w(f"{out}/{name}_Pi.txt", ms.Pi); w(f"{out}/{name}_Sigma.txt", ms.B @ ms.B.T)
    w(f"{out}/{name}_assembled_modes.txt", T @ ms.B)
    assert np.abs(ms.local_field(np.eye(ms.p)[0]) - E[:, 0]).max() < 1e-12     # consistent with the sampler's map


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--primal-dims", type=int, nargs="*", default=[6, 8])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    S = ML.MSMSetup(); N = S.phys.N; T = S.kle.T
    w(f"{a.out}/cell_centers.txt", S.points)
    w(f"{a.out}/kle_global_T.txt", T)
    w(f"{a.out}/kle_global_Phi.txt", T / np.sqrt(S.kle.lam_all[:S.r])[None, :] if hasattr(S.kle, "lam_all") else T)
    w(f"{a.out}/kle_global_lambda.txt", np.asarray(S.kle.lam_all[:S.r])[:, None])
    dump_grid(a.out, "primal4", S.primal, N, T)
    dump_grid(a.out, "dual", S.dual, N, T)
    w(f"{a.out}/primal4_null_V0.txt", S.primal.V0); w(f"{a.out}/primal4_null_fields.txt", T @ S.primal.V0)
    for d in a.primal_dims:
        Sd = ML.MSMSetup(primal_local_dim=d)
        dump_grid(a.out, f"primal{d}", Sd.primal, N, T)
    open(f"{a.out}/README.txt", "w").write(__doc__)
    open(f"{a.out}/load_msm_kle.m", "w").write(MATLAB)
    print("written", len(os.listdir(a.out)), "files to", a.out)


MATLAB = r"""% Load and plot the MSM-study KLE bases (run from the export folder).
n = 32; xy = load('cell_centers.txt'); xc = unique(xy(:,1)); yc = unique(xy(:,2));
F = @(v) reshape(v, n, n);                     % A(iy, ix); contourf(xc, yc, A) has x horizontal
T = load('kle_global_T.txt'); lam = load('kle_global_lambda.txt');
figure; for k = 1:6
    subplot(2,3,k); contourf(xc, yc, F(T(:,k)), 20); colormap jet; axis square; title(sprintf('global mode %d (column of T)', k));
end
grids = {'primal4', 'dual'};
for g = 1:numel(grids)
    name = grids{g};
    E  = load([name '_local_modes.txt']);        % piecewise local modes, field = E * c
    A  = load([name '_assembled_modes.txt']);    % the same after projection onto the global KLE (T * B)
    bi = load([name '_block_index.txt']);
    li = load([name '_local_index.txt']);        % block, mode, mu
    figure; contourf(xc, yc, F(bi), max(bi)); colormap jet; axis square; colorbar; title([name ' blocks']);
    j = find(li(:,1) == 7, 4);                   % the first 4 modes of block 7
    figure;
    for q = 1:numel(j)
        subplot(2,4,q);   contourf(xc, yc, F(E(:,j(q))), 20); axis square; title(sprintf('%s: local mode %d, block 7', name, q));
        subplot(2,4,4+q); contourf(xc, yc, F(A(:,j(q))), 20); axis square; title('after projection onto global KLE');
    end
    colormap jet;
end
V = load('primal4_null_fields.txt');             % the 6 directions the primal grid cannot reach (fields T*V0)
figure; for k = 1:6
    subplot(2,3,k); contourf(xc, yc, F(V(:,k)), 20); colormap jet; axis square; title(sprintf('primal4 null direction %d', k));
end
"""
if __name__ == "__main__":
    main()
