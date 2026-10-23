"""Least-squares projection of a target field onto the model space of the sampler.

The sampler's standardized field is  YP = P A theta,  A: theta (12 blocks) -> Y (assemble: Y[elem_i] = T_i theta_i, the
subdomain cells partition the domain once),  P = PHI U diag(invsqrt) U^T PHI^T  (whitened projection, symmetric).
`fit_theta` finds the theta that minimizes ||g - P A theta||^2 (+ damp^2 ||theta||^2) with LSQR, using only the
field's own operators (field.assemble, field.whiten_project: CPU or GPU), and the adjoint
(P A)^T z = [T_i^T (P z)[elem_i]]_i.  With 6000 coefficients and a range of at most M global modes the problem is
under-determined; LSQR (damp = 0) returns the minimum-norm minimizer.  The residual says how well the model space
can represent g (the floor for any sampler); ||theta||^2 / n_theta is about 1 for a typical prior draw.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr


def field_operator(field):
    """LinearOperator (N x n_theta) of theta (flat, blocks concatenated) -> YP, with its adjoint."""
    dims = [int(T.shape[1]) for T in field.T]
    offs = np.concatenate([[0], np.cumsum(dims)])
    N = field.N

    def split(theta):
        return [np.asarray(theta[offs[i]:offs[i + 1]], dtype=field.dtype) for i in range(len(dims))]

    def matvec(theta):
        return np.asarray(field.whiten_project(field.assemble(split(np.asarray(theta, float).ravel()))), float)

    def rmatvec(g):
        z = np.asarray(field.whiten_project(np.asarray(g, dtype=field.dtype).ravel()), float)
        return np.concatenate([np.asarray(T, float).T @ z[e] for T, e in zip(field.T, field.elem)])

    return LinearOperator((N, int(offs[-1])), matvec=matvec, rmatvec=rmatvec, dtype=float), dims


def fit_theta(field, g, iter_lim=300, atol=1e-10, btol=1e-10, damp=0.0):
    """Returns dict(theta (flat), blocks, yp, residual (relative, over all cells), iterations, istop)."""
    op, dims = field_operator(field)
    g = np.asarray(g, float).ravel()
    sol = lsqr(op, g, damp=damp, atol=atol, btol=btol, iter_lim=int(iter_lim))
    theta = sol[0]
    yp = op.matvec(theta)
    offs = np.concatenate([[0], np.cumsum(dims)])
    return dict(theta=theta, blocks=[theta[offs[i]:offs[i + 1]] for i in range(len(dims))], yp=yp,
                residual=float(np.linalg.norm(g - yp) / max(np.linalg.norm(g), 1e-300)),
                iterations=int(sol[2]), istop=int(sol[1]), theta_msq=float(np.mean(theta ** 2)))
