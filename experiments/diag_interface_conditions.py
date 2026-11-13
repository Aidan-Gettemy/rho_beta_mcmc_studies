"""
Deterministic check of the interface condition of the domain-decomposed blocks (docs/dd_flux_experiment.md, Robin variant).
For the TRUE permeability, every block is solved with (a) the exact face pressures (zero reference), (b) the spline face
pressures as Dirichlet data (the method of experiments 7/8) and (c) a mixed condition: Robin at the faces where flux is
measured, spline Dirichlet elsewhere, alpha p + K dp/dn = alpha g + h with g the spline pressure and h the noisy flux datum
(same noise realisation as the studies).  Reported, aggregated over the blocks of a grid:
  p err   : ||p_block - p_true|| / ||p_true|| over all cells (global norm)
  sens    : ||d_loc - p_block(sensors)|| / ||d||  (the local data misfit that the true K leaves; noise floor ~0.003)
  q err   : one-sided block flux vs true flux, relative L2, on the interface faces with flux data / without
  info    : mean over blocks of the linearised posterior/prior sd ratio of the local coefficients,
            sqrt(tr[(I + w J^T J)^-1] / r_k), J = d p(sensors) / d c at the projection of the truth; w = the study's
            global-norm weight (information as the studies see it) and w = 1/sigma_p^2 (information of the real noise)
    python experiments/diag_interface_conditions.py
"""
import _common  # noqa: F401
import numpy as np

from brho import msm_local as ML, msm_sampler as MS, msm_dd as DD, msm_flux as MF

ALPHAS = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SIGN = dict(L=+1.0, R=-1.0, B=+1.0, T=-1.0)           # h = K dp/dn_out = -(outward Darcy flux) = SIGN * q (+x / +y flux)
OUTW = dict(L=-1.0, R=+1.0, B=-1.0, T=+1.0)           # +x/+y flux = OUTW * outward flux


def main():
    S = ML.MSMSetup(primal_local_dim=6)
    data = MS.Data(S, 0.003, 2026)
    sp = DD.InterfaceSpline(S, data)
    fd = MF.FluxData(S, data, 0.003, 0.08)
    n = S.n
    lt = S.kle.field(data.theta_true)
    K2 = np.exp(lt).reshape(n, n)
    p2 = S.phys.solve(K2.ravel()).reshape(n, n)
    pt, Kt = p2.ravel(), K2.ravel()
    w_study, w_true = data.w, 1.0 / data.sigma_p ** 2
    for gname, ms in (("primal", S.primal), ("dual", S.dual)):
        g_sp = DD.DDGrid(S, ms, data, spline=sp)
        g_ex = DD.DDGrid(S, ms, data, exact=(K2, p2))
        print(f"\n== {gname} grid ({len(g_sp.blocks)} blocks)")
        print("%-22s %9s %9s %10s %10s %11s %11s" % ("interface condition", "p err", "sens", "q err meas", "q err unm",
                                                     "info (w_st)", "info (w_tr)"))

        def run(label, bcs):
            e2 = pt2 = s2 = d2 = 0.0
            qm, qu = [[], []], [[], []]
            infos = [[], []]
            for b, bc in zip(g_sp.blocks, bcs):
                sol = DD.BlockSolver(b.nxl, b.nyl, bc)
                K = Kt[b.cells]
                pl = sol.solve(K)
                e2 += np.sum((pl - pt[b.cells]) ** 2)
                s2 += np.sum((b.d_loc - pl[b.obs_loc]) ** 2)
                P = pl.reshape(b.nxl, b.nyl)
                Kl = K.reshape(b.nxl, b.nyl)
                for side in DD.SIDES:
                    if b.kinds[side] != "iface":
                        continue
                    xy, c = b.faces[side]
                    ia, ja, ib, jb = c
                    qtrue = MF.face_flux(K2, p2, ia, ja, ib, jb)
                    cell_p = dict(L=P[0, :], R=P[-1, :], B=P[:, 0], T=P[:, -1])[side]
                    cell_K = dict(L=Kl[0, :], R=Kl[-1, :], B=Kl[:, 0], T=Kl[:, -1])[side]
                    v = bc[side]
                    g, h, ainv = (np.asarray(v[k], float) for k in ("g", "h", "ainv")) if isinstance(v, dict) \
                        else (np.asarray(v, float), 0 * np.asarray(v, float), 0 * np.asarray(v, float))
                    t = 2.0 * cell_K
                    G = t / (1.0 + t * ainv)
                    q_out = G * (cell_p - (g + ainv * h))              # outward Darcy flux through the face
                    qmod = OUTW[side] * q_out
                    vertical = side in ("L", "R")
                    line = int(ib[0]) if vertical else int(jb[0])
                    along = ja if vertical else ia
                    meas = np.array([("v" if vertical else "h", line, int(a)) in fd.index for a in along])
                    for sel, acc in ((meas, qm), (~meas, qu)):
                        if sel.any():
                            acc[0].append(np.sum((qmod[sel] - qtrue[sel]) ** 2)); acc[1].append(np.sum(qtrue[sel] ** 2))
                # linearised information of the local coefficients at the projection of the truth
                A = b.psi * b.sqmu
                c_hat = np.linalg.lstsq(A, lt[b.cells], rcond=None)[0]
                J = np.empty((len(b.obs_loc), b.r_k))
                for j in range(b.r_k):
                    dc = np.zeros(b.r_k); dc[j] = 1e-5
                    pp = sol.solve(np.exp(b.log_field(c_hat + dc)))[b.obs_loc]
                    pm = sol.solve(np.exp(b.log_field(c_hat - dc)))[b.obs_loc]
                    J[:, j] = (pp - pm) / 2e-5
                for i, wt in enumerate((w_study, w_true)):
                    H = np.eye(b.r_k) + wt * J.T @ J
                    infos[i].append(np.sqrt(np.trace(np.linalg.inv(H)) / b.r_k))
            pe, se = np.sqrt(e2) / np.linalg.norm(pt), np.sqrt(s2) / data.dnorm
            qme = np.sqrt(sum(qm[0]) / sum(qm[1])) if qm[1] else float("nan")
            que = np.sqrt(sum(qu[0]) / sum(qu[1])) if qu[1] else float("nan")
            print("%-22s %9.4f %9.4f %10.4f %10.4f %11.3f %11.3f" % (label, pe, se, qme, que, np.mean(infos[0]), np.mean(infos[1])))

        run("exact faces", [b.bc for b in g_ex.blocks])
        run("Dirichlet (spline)", [b.bc for b in g_sp.blocks])
        for al in ALPHAS:
            bcs = []
            for b in g_sp.blocks:
                bc = {}
                for side, v in b.bc.items():
                    if v is None or b.kinds[side] != "iface":
                        bc[side] = v
                        continue
                    xy, c = b.faces[side]
                    ia, ja, ib, jb = c
                    vertical = side in ("L", "R")
                    line = int(ib[0]) if vertical else int(jb[0])
                    along = ja if vertical else ia
                    h = np.zeros(len(v)); ainv = np.zeros(len(v))
                    for m, a in enumerate(along):
                        i = fd.index.get(("v" if vertical else "h", line, int(a)))
                        if i is not None:
                            h[m] = SIGN[side] * fd.q[i]; ainv[m] = 1.0 / al
                    bc[side] = dict(g=np.asarray(v, float), h=h, ainv=ainv)
                bcs.append(bc)
            run(f"Robin alpha={al:g}", bcs)
        # Robin on EVERY interface face, flux interpolated along the line from the measured faces
        for kind in ("linear", "cubic"):
            for al in (0.1, 1.0, 3.0):
                bcs = []
                for b in g_sp.blocks:
                    bc = {}
                    for side, v in b.bc.items():
                        if v is None or b.kinds[side] != "iface":
                            bc[side] = v
                            continue
                        xy, c = b.faces[side]
                        ia, ja, ib, jb = c
                        vertical = side in ("L", "R")
                        line = int(ib[0]) if vertical else int(jb[0])
                        along = ja if vertical else ia
                        qi = fd.line_flux("v" if vertical else "h", line, kind)[along]
                        bc[side] = dict(g=np.asarray(v, float), h=SIGN[side] * qi, ainv=np.full(len(v), 1.0 / al))
                    bcs.append(bc)
                run(f"Robin all, {kind[:3]}, a={al:g}", bcs)


if __name__ == "__main__":
    main()
