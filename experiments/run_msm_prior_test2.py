"""
Prior test (likelihood off, Phi = 0, every proposal accepted, no solves) of the 2026-11-03 samplers:
alt (L = 1, 10, 100; with and without the frozen offset), mixture_inn, mixture with 8 modes per block (Pi
lift), sequence_exact.  Fixed step s = 0.5 (no adaptation).  Reports the same statistics as
run_msm_prior_test.py: covariance error of theta against I_64, pointwise variance ratio against the exact
truncated-KLE prior variance diag(T T').
    BRHO_ROOT=runs/msm_local_kle_2026-11 python experiments/run_msm_prior_test2.py
Writes tables/msm_prior_test2/msm_prior_test2.{json,md}.
"""
import _common  # noqa: F401
import os
import sys

import numpy as np

from brho import io, msm_local as ML, msm_sampler as MS


class ZeroData:                                   # likelihood off
    def potential(self, theta):
        return 0.0, 0.0


def stats(th, T):
    flat = th.reshape(-1, th.shape[-1]); C = np.cov(flat.T)
    var = np.einsum("ij,jk,ik->i", T, C, T) / (T ** 2).sum(1)
    return dict(cov_err=float(np.linalg.norm(C - np.eye(64)) / 8), var_ratio_mean=float(var.mean()),
                var_ratio_min=float(var.min()), var_ratio_max=float(var.max()))


def main():
    S4, S8 = ML.MSMSetup(), ML.MSMSetup(primal_local_dim=8)
    T = S4.kle.T
    runs = [("alt L=1 (offset)", S4, "alt", dict(alt_L=1)), ("alt L=10 (offset)", S4, "alt", dict(alt_L=10)),
            ("alt L=100 (offset)", S4, "alt", dict(alt_L=100)),
            ("alt L=1, no offset", S4, "alt", dict(alt_L=1, alt_offset=False)),
            ("mixture_inn (never project)", S4, "mixture_inn", {}),
            ("mixture, Pi lift, 8 modes", S8, "mixture", {}),
            ("mixture, Pi lift, 4 modes (exp 4)", S4, "mixture", {}),
            ("sequence_exact", S4, "sequence_exact", dict(k_phase=10000))]
    res = {}
    for name, S, kind, kw in runs:
        n_iter = 20000
        outs = [MS.run_chain(kind, S, ZeroData(), 5, c, n_iter, 0, s0=0.5, save_every=50, **kw) for c in range(24)]
        if kind == "sequence_exact":
            res[name + " phase 1"] = stats(np.stack([o["theta"][:200] for o in outs])[:, 20:], T)
            res[name + " phase 2"] = stats(np.stack([o["theta"][200:] for o in outs])[:, 20:], T)
        else:
            res[name] = stats(np.stack([o["theta"] for o in outs])[:, 20:], T)
        print(name, "done", flush=True)
    tdir = io.tables_dir("msm_prior_test2"); os.makedirs(tdir, exist_ok=True)
    io.save_json(os.path.join(tdir, "msm_prior_test2.json"), res)
    with open(os.path.join(tdir, "msm_prior_test2.md"), "w") as f:
        f.write("| sampler | cov err (Frob/sqrt r) | pointwise var ratio mean (min-max) |\n|---|---|---|\n")
        for k, v in res.items():
            f.write(f"| {k} | {v['cov_err']:.3f} | {v['var_ratio_mean']:.3f} ({v['var_ratio_min']:.2f}-{v['var_ratio_max']:.2f}) |\n")
    print(open(os.path.join(tdir, "msm_prior_test2.md")).read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
