"""
Oracle-interface sweep (docs/dd_gibbs_experiment.md): how does the accuracy of the domain-decomposed posterior mean depend on the
accuracy of the interface data?  The blocks get the EXACT interface pressure g and flux h of the truth (Robin condition
alpha p + K dp/dn = alpha g + h, alpha 0.1), with a synthetic SMOOTH error added: a Gaussian-process draw along each interface line
(squared-exponential, correlation length --corr-len cells), shared by the two blocks that touch the line and scaled so that
||error|| / ||exact datum|| = delta on every block side (the relative measure brho.msm_dd.DDGrid.interface_report uses for the spline).
    --dp   relative pressure errors delta_g to sweep (flux exact)
    --dq   relative flux errors delta_q to sweep (pressure exact)
plus the actual setup (spline pressure, flux interpolated from the 224 measured faces) as reference, and the relative errors of that
setup so the sweep range can be read against it.  Blocks are independent given the interface, so independent block chains are the
exact sampler of this problem.  Output: a table line per configuration (stdout) and <out>.json.

    python experiments/oracle_interface_sweep.py --grid primal --primal-local-dim 10 --dual-energy 0.99 --out sweep_primal
"""
import _common  # noqa: F401
import argparse, json, sys, time
from multiprocessing import Pool
import numpy as np
from brho import msm_local as ML, msm_sampler as MS, msm_dd as DD, msm_flux as MF
from brho.mpsrf import mpsrf_curve
import run_dd_chains as R

_G = {}
SIGN = dict(L=+1.0, R=-1.0, B=+1.0, T=-1.0)


class OracleFlux:
    """Exact Darcy flux at every face of the interface lines (same interface as MF.FluxData for DD.Block.set_robin)."""

    def __init__(self, S, data):
        n = S.n
        self.n = n
        self.K2 = np.exp(S.kle.field(data.theta_true)).reshape(n, n)
        self.p2 = S.phys.solve(self.K2.ravel()).reshape(n, n)
        self.lines = MF.interface_lines(n)

    def has_line(self, orient, line):
        return line in self.lines

    def line_flux(self, orient, line, kind="cubic"):
        n, t = self.n, np.arange(self.n)
        if orient == "v":
            return MF.face_flux(self.K2, self.p2, np.full(n, line - 1), t, np.full(n, line), t)
        return MF.face_flux(self.K2, self.p2, t, np.full(n, line - 1), t, np.full(n, line))


def _line_noise(n, corr_len, key):
    """Unit-variance smooth GP draw on the n faces of one line (deterministic in `key`)."""
    t = np.arange(n)
    C = np.exp(-0.5 * ((t[:, None] - t[None, :]) / corr_len) ** 2) + 1e-8 * np.eye(n)
    return np.linalg.cholesky(C) @ np.random.default_rng(np.random.SeedSequence(key)).standard_normal(n)


def _side_line(blk, side):
    xy, (ia, ja, ib, jb) = blk.faces[side]
    vert = side in ("L", "R")
    return ("v" if vert else "h"), int(ib[0]) if vert else int(jb[0]), (ja if vert else ia)


def perturb(grid, S, dp, dq, corr_len, seed=5):
    """Add the smooth interface errors (relative size dp for g, dq for the flux) to every Robin side of every block, in place."""
    n = S.n
    for blk in grid.blocks:
        if not blk.robin:
            continue
        bc = {}
        for side, v in blk.bc.items():
            if not isinstance(v, dict):
                bc[side] = v
                continue
            orient, line, along = _side_line(blk, side)
            g, h = np.array(v["g"], float), np.array(v["h"], float)
            for amp, key, arr, sgn in ((dp, [seed, 1, ord(orient), line], g, 1.0), (dq, [seed, 2, ord(orient), line], h, SIGN[side])):
                if amp:
                    e = _line_noise(n, corr_len, key)[along]
                    arr += sgn * amp * np.linalg.norm(arr) / np.linalg.norm(e) * e
            bc[side] = dict(g=g, h=h, ainv=v["ainv"])
        blk.bc = bc
        blk.solver = DD.BlockSolver(blk.nxl, blk.nyl, bc)


def interface_errors(actual, exact):
    """Relative errors of the actual grid's g and h against the exact ones, over all Robin sides."""
    eg, eh = [], []
    for a, b in zip(actual.blocks, exact.blocks):
        for side, v in a.bc.items():
            if isinstance(v, dict):
                w = b.bc[side]
                eg.append(np.linalg.norm(v["g"] - w["g"]) / np.linalg.norm(w["g"]))
                eh.append(np.linalg.norm(v["h"] - w["h"]) / np.linalg.norm(w["h"]))
    return dict(g_mean=float(np.mean(eg)), g_max=float(np.max(eg)), h_mean=float(np.mean(eh)), h_max=float(np.max(eh)))


def _task(t):
    k, ch = t
    a = _G["a"]
    return DD.run_block_chain(_G["grid"].blocks[k], [7, k, ch], a.n_iter, a.adapt, 0.2, 50)


def run_config(S, data, grid, a, label):
    _G.update(grid=grid, a=a)
    nb, nch = len(grid.blocks), a.chains
    t0 = time.time()
    with Pool(a.workers) as P:
        res = P.map(_task, [(k, ch) for k in range(nb) for ch in range(nch)])
    c = [np.stack([res[k * nch + ch]["c"] for ch in range(nch)]) for k in range(nb)]
    M = a.adapt // 50
    lt = S.kle.field(data.theta_true)
    rng = np.random.default_rng(0)
    cb = []
    for k in range(nb):
        f = c[k][:, M:].reshape(-1, c[k].shape[2]).astype(float)
        cb.append(f[rng.choice(len(f), 600, replace=len(f) < 600)])
    F = S.kle.field(grid.theta(cb))
    m, sd = F.mean(0), F.std(0)
    sup = max(float(np.asarray(mpsrf_curve(c[k][:, 20:].astype(float), skip=2, step=2, iter_burn=1)["mpsrf"], float)[-1]) for k in range(nb))
    out = dict(label=label, err=float(np.sqrt(np.mean((m - lt) ** 2))), sd=float(sd.mean()),
               z=float(np.median(np.abs(m - lt) / sd)), acc=float(np.mean([r["acc"][a.adapt:].mean() for r in res])),
               mpsrf=sup, sec=time.time() - t0)
    print(f"{a.grid} {label:28s} err {out['err']:.4f} sd {out['sd']:.4f} med|e|/sd {out['z']:.2f} acc {out['acc']:.3f} "
          f"sup-MPSRF {out['mpsrf']:.2f} ({out['sec']:.0f}s)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, choices=["primal", "dual"])
    ap.add_argument("--primal-local-dim", type=int, default=6)
    ap.add_argument("--dual-energy", type=float, default=0.95)
    ap.add_argument("--dp", type=float, nargs="*", default=[0.0, 0.001, 0.002, 0.004, 0.008])
    ap.add_argument("--dq", type=float, nargs="*", default=[0.003, 0.01, 0.03, 0.07])
    ap.add_argument("--corr-len", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=20000)
    ap.add_argument("--adapt", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-actual", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base = argparse.Namespace(grid=a.grid, sigma_rel=0.00013258252, data_seed=2026, data_relerr=0.003, smoothing=1e-3, neighbors=60,
                              flux_noise_rel=0.003, flux_sigma_rel=0.08, overlap_halo=0, robin_alpha=a.alpha, robin_interp="cubic",
                              primal_local_dim=a.primal_local_dim, dual_energy=a.dual_energy)
    S, data, spline, actual = R.build(base)
    ms = S.primal if a.grid == "primal" else S.dual
    orc = OracleFlux(S, data)

    def oracle(dp, dq):
        g = DD.DDGrid(S, ms, data, exact=(orc.K2, orc.p2))
        for b in g.blocks:
            b.set_robin(orc, a.alpha, "cubic")
        perturb(g, S, dp, dq, a.corr_len)
        return g

    results = dict(settings=vars(a), interface_errors_actual=interface_errors(actual, oracle(0.0, 0.0)), runs=[])
    print("relative interface errors of the actual setup (spline g, interpolated measured flux h):", results["interface_errors_actual"], flush=True)
    for dp in a.dp:
        results["runs"].append(dict(dp=dp, dq=0.0, **run_config(S, data, oracle(dp, 0.0), a, f"dp={dp:g} dq=0")))
    for dq in a.dq:
        results["runs"].append(dict(dp=0.0, dq=dq, **run_config(S, data, oracle(0.0, dq), a, f"dp=0 dq={dq:g}")))
    if not a.no_actual:
        results["runs"].append(dict(dp=None, dq=None, **run_config(S, data, actual, a, "actual (spline + measured)")))
    json.dump(results, open(a.out + ".json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
