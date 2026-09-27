"""
Quadrature rules on boxes [lo, hi]^d (port of KLE_matlab/+kle/quadrature.m).

Nodes are returned in 'ij' order (first coordinate slowest), weights sum to
the box volume.  Gauss-Legendre nodes/weights come from
numpy.polynomial.legendre.leggauss (Golub-Welsch, same as the MATLAB port).
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def gauss_legendre_1d(nq, lo=0.0, hi=1.0):
    xi, wi = leggauss(int(nq))
    return lo + (hi - lo) / 2 * (xi + 1), wi * (hi - lo) / 2


def midpoint_1d(nq, lo=0.0, hi=1.0):
    h = (hi - lo) / nq
    return lo + h * (np.arange(nq) + 0.5), np.full(nq, h)


RULES_1D = {"gauss_legendre": gauss_legendre_1d, "midpoint": midpoint_1d}


def rule_1d(name, nq, lo=0.0, hi=1.0):
    return RULES_1D[name](nq, lo, hi)


def tensor_rule(name, nq, dim, lo=0.0, hi=1.0):
    """Tensor-product rule: nodes (nq^dim, dim) in 'ij' order, weights."""
    x, w = rule_1d(name, nq, lo, hi)
    grids = np.meshgrid(*([x] * dim), indexing="ij")
    wgrids = np.meshgrid(*([w] * dim), indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], 1)
    weights = np.prod(np.stack([g.ravel() for g in wgrids], 1), 1)
    return nodes, weights
