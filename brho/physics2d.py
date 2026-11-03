"""
2D permeameter: a thin wrapper over the grid-generic `fv.Permeameter`
(numba-compiled assembly when available, SuperLU with MMD_AT_PLUS_A).

    -div(K grad p) = 0 on (0,1)^2
    -K dp/dn = q_in  on the left face  x = 0   (prescribed unit influx)
     p = 0           on the right face x = 1   (outflow)
     no flow         on top and bottom

Kept for backward compatibility: `Darcy2D(n)` is `Permeameter((n, n))` plus
the attribute `n`.  Cell index m = ix * n + iy; reshape(n, n) gives [ix, iy].
The July scripts' solver (mcmc_study.py) is reproduced to ~1e-14
(tests/test_fv.py).
"""
from .fv import Permeameter


class Darcy2D(Permeameter):
    def __init__(self, n=32, q_in=1.0, solver="auto", backend="auto",
                 bc="influx", p_left=None, p_right=None):
        # bc="dirichlet": p = p_left (1) on x = 0, p = p_right (0) on x = 1
        # (see brho/fv.py); the default is the influx problem above.
        super().__init__((n, n), q_in=q_in, solver=solver, backend=backend,
                         bc=bc, p_left=p_left, p_right=p_right)
        self.n = n
