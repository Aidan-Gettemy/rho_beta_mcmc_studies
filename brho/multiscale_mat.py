"""
Multiscale subdomain field source from MATLAB matrices (NumPy, no GPU).

Reproduces the production recipe of the group's multiscale sampler:

    1. theta_i ~ N(0, I_m)  for each subdomain i (m = 500 local modes)
    2. Y[elem_i] = T_i @ theta_i            (assemble the central cells of each subdomain)
    3. YP = PHI @ W @ PHI^T @ Y             (whitened projection onto the global KLE)
       with W = U diag(invsqrt) U^T  (U = eigvec, invsqrt from the whitening .npz)
    4. K = beta * exp(rho * YP)             (done by the caller; YP has ~unit-variance modes)

    from brho.multiscale_mat import load_multiscale
    f = load_multiscale("external/matrices", "38x38x45_38x38x45_l2.0x2.0x2.0_m500_IND",
                        "external/matrices/whitening_operator_SP1.npz")
    YP = f.sample(np.random.default_rng(0))

The whitening file must have been built for the same PHI (same number of global modes); a mismatch
raises ValueError.  Reading the .mat files needs h5py (v7.3 files) or scipy (older files).

Memory: PHI is (64980 x n_global), about 1.2 GB in float32 for 4699 modes.  Convert it once with
`experiments/run_external_curve.py --cache-phi` to a .npy file; workers then open it with mmap, so 64
processes on one node share one copy in the page cache.  The same recipe is the field source that
the sampler needs for the external-solver runs (the curve and the sampler must use the same operator).
"""
import os

import numpy as np


# ----------------------------------------------------------------------------- I/O
def _read_mat(path, names):
    """Read variables from a MATLAB .mat file, returned in MATLAB orientation.
    v7.3 files are HDF5 (h5py shows the dimensions reversed, so we transpose back);
    older files go through scipy."""
    try:
        import h5py
        with h5py.File(path, "r") as h:
            return {n: np.array(h[n]).T for n in names}
    except (ImportError, OSError):      # no h5py, or an older (non-HDF5) .mat file
        from scipy.io import loadmat
        d = loadmat(path)
        return {n: np.asarray(d[n]) for n in names}


def cache_phi(mat_path, npy_path, dtype=np.float32):
    """One-time conversion of phi_matrix_full_*.mat -> C-ordered float32 .npy (cells x modes)."""
    phi = _read_mat(mat_path, ["PHI"])["PHI"].astype(dtype, order="C")
    np.save(npy_path, phi)
    return phi.shape


class MultiscaleField:
    def __init__(self, T_list, elem_list, PHI, eigvec, invsqrt, dtype=np.float32):
        self.dtype = dtype
        self.T = [np.ascontiguousarray(T, dtype=dtype) for T in T_list]       # (cells_i, m_i)
        self.elem = [np.asarray(e, dtype=np.int64) for e in elem_list]        # 0-based cell ids
        self.PHI = PHI if PHI.dtype == dtype else PHI.astype(dtype)           # (N, M)  maybe memmap
        self.U = np.asarray(eigvec, dtype=dtype)                              # (M, M)
        self.invsqrt = np.asarray(invsqrt, dtype=dtype).ravel()               # (M,)
        self.N = self.PHI.shape[0]
        self.M = self.PHI.shape[1]
        self.projector = None          # optional replacement of whiten_project (e.g. GpuProjector)
        self._check()

    def _check(self):
        if self.U.shape != (self.M, self.M) or self.invsqrt.size != self.M:
            raise ValueError(
                f"whitening operator does not match the global basis: PHI has {self.M} modes, "
                f"eigvec is {self.U.shape}, invsqrt has {self.invsqrt.size}. "
                "Use the whitening file built for this PHI.")
        allcells = np.concatenate(self.elem)
        if allcells.size != self.N or not np.array_equal(np.sort(allcells), np.arange(self.N)):
            raise ValueError("central cells of the subdomains do not partition the domain exactly once")
        for T, e in zip(self.T, self.elem):
            if T.shape[0] != e.size:
                raise ValueError("T rows must equal the number of central cells of the subdomain")

    # ------------------------------------------------------------------ pieces
    def sample_theta(self, rng):
        return [rng.standard_normal(T.shape[1]).astype(self.dtype) for T in self.T]

    def assemble(self, theta_list):
        Y = np.zeros(self.N, dtype=self.dtype)
        for T, e, th in zip(self.T, self.elem, theta_list):
            Y[e] = T @ th
        return Y

    def whiten_project(self, Y):
        """YP = PHI @ U @ diag(invsqrt) @ U^T @ PHI^T @ Y  (Y: (N,) or (N, k))."""
        one = Y.ndim == 1
        Yk = Y.reshape(self.N, -1).astype(self.dtype, copy=False)
        if self.projector is not None:
            out = self.projector(Yk)
            return out[:, 0] if one else out
        z = self.PHI.T @ Yk
        z = self.U.T @ z
        z *= self.invsqrt[:, None]
        z = self.U @ z
        out = self.PHI @ z
        return out[:, 0] if one else out

    def field(self, theta_list):
        return self.whiten_project(self.assemble(theta_list))

    def sample(self, rng):
        return self.field(self.sample_theta(rng))


def load_multiscale(matrix_dir, tag, whitening_npz, n_sub=12, phi_npy=None, dtype=np.float32):
    """Load the 38x38x45 files named like KLE_matrix_<tag>_<i>.mat, SG_elem_<tag>_<i>.mat,
    phi_matrix_full_<tag>.mat (<tag> includes the trailing _IND), i = 1..n_sub."""
    T_list, elem_list = [], []
    for i in range(1, n_sub + 1):
        T = _read_mat(os.path.join(matrix_dir, f"KLE_matrix_{tag}_{i}.mat"), ["T"])["T"]
        e = _read_mat(os.path.join(matrix_dir, f"SG_elem_{tag}_{i}.mat"), ["elem"])["elem"].ravel()
        T_list.append(T)
        elem_list.append(e)
    allcells = np.concatenate(elem_list)
    if allcells.min() == 1:                 # MATLAB indices are 1-based
        elem_list = [e - 1 for e in elem_list]
    if phi_npy and os.path.exists(phi_npy):
        PHI = np.load(phi_npy, mmap_mode="r")
    else:
        PHI = _read_mat(os.path.join(matrix_dir, f"phi_matrix_full_{tag}.mat"), ["PHI"])["PHI"]
    w = np.load(whitening_npz)
    return MultiscaleField(T_list, elem_list, PHI, w["eigvec"], w["invsqrt"], dtype=dtype)



class GpuProjector:
    """YP = PHI U diag(invsqrt) U^T PHI^T Y with PHI, U on the device (xp = cupy; xp = numpy for tests).
    The CPU projection reads ~2.6 GB per call and saturates the node's memory bandwidth at ~70 GB/s (6x slower
    with 64 processes); on a GPU it is bandwidth-bound at TB/s.  Each rank keeps its own copy (1.2 GB float32);
    PHI is uploaded in row blocks so the host never holds a second copy."""

    def __init__(self, PHI, U, invsqrt, xp, block=8192):
        self.xp = xp
        N, M = PHI.shape
        self.PHI = xp.empty((N, M), dtype=xp.float32)
        for i in range(0, N, block):
            self.PHI[i:i + block] = xp.asarray(np.asarray(PHI[i:i + block], dtype=np.float32))
        self.U = xp.asarray(np.asarray(U, dtype=np.float32))
        self.invsqrt = xp.asarray(np.asarray(invsqrt, dtype=np.float32).ravel())

    def __call__(self, Yk):
        xp = self.xp
        y = xp.asarray(np.ascontiguousarray(Yk, dtype=np.float32))
        z = self.PHI.T @ y
        z = self.U.T @ z
        z *= self.invsqrt[:, None]
        out = self.PHI @ (self.U @ z)
        return out.get() if hasattr(out, "get") else np.asarray(out)


def attach_gpu_projector(field, xp):
    field.projector = GpuProjector(field.PHI, field.U, field.invsqrt, xp)
    return field
