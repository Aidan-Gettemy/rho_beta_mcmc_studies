"""Post-processing of GLAM runs: strided samples of the chains and pointwise field statistics.

Chains are repetition-encoded (a record covers iterations [start, start + count); rejected moves are repeats of the
same record).  The tail starts at the end of window 1 (the adaptation window).  A strided sample takes, for every rank,
the state at iterations tail_start, tail_start + stride, ... < n_iter, so rejections are counted with their true weight.

  load_chain(out, rank)                       -> dict of arrays (records + per-iteration traces)
  strided_samples(out, nranks, tail_start, stride, n_iter)
                                              -> theta (n, d), w (n,), rank (n,), iteration (n,), phi (n,)
  best_record(out, nranks, tail_start=0)      -> the lowest-Phi state of all chains (theta, w, phi, rank, iteration)
  best_fields(model, best, rank)              -> forward model on that state: log K, concentration (cells x times),
                                                 pressure, pressure drop, recomputed Phi
  plug_mask(nx, ny, nz)                       -> boolean (cells,) mask of the cylindrical plug (solver cell order)
  field_statistics(model, samples, comm, ...) -> pointwise mean / std (Welford, merged across ranks) of the pressure,
                                                 concentration and log-permeability fields obtained by re-running the
                                                 forward model on every strided sample (samples split over ranks)
"""
import glob
import os

import numpy as np


def load_chain(out, rank):
    parts = sorted(glob.glob(os.path.join(out, "chains", f"rank{rank:03d}_part*.npz")))
    if not parts:
        raise FileNotFoundError(f"no chain parts for rank {rank} in {out}/chains")
    z = [np.load(p) for p in parts]
    cat = lambda k: np.concatenate([a[k] for a in z])  # noqa: E731
    return dict(start=cat("start"), count=cat("count"), theta=np.concatenate([a["theta"] for a in z]),
                w=cat("w"), phi=cat("phi"), kind=cat("trace_kind"), acc=cat("trace_acc"),
                beta_g=cat("trace_beta_g"), beta_l=cat("trace_beta_l"), phi_trace=cat("trace_phi"))


def n_iterations(ch):
    return int(ch["start"][-1] + ch["count"][-1])


def strided_samples(out, nranks, tail_start, stride, n_iter=None, dtype=np.float64):
    th, w, rk, its, phi = [], [], [], [], []
    for r in range(nranks):
        ch = load_chain(out, r)
        n = n_iterations(ch) if n_iter is None else int(n_iter)
        if n_iterations(ch) < n:
            raise ValueError(f"rank {r}: chain has {n_iterations(ch)} iterations, expected {n}")
        t = np.arange(int(tail_start), n, int(stride))
        j = np.searchsorted(ch["start"], t, side="right") - 1
        if np.any(j < 0) or np.any(t >= ch["start"][j] + ch["count"][j]):
            raise ValueError(f"rank {r}: chain records do not cover the tail")
        th.append(ch["theta"][j].astype(dtype)); w.append(ch["w"][j]); phi.append(ch["phi"][j])
        rk.append(np.full(t.size, r)); its.append(t)
    return dict(theta=np.concatenate(th), w=np.concatenate(w), rank=np.concatenate(rk),
                iteration=np.concatenate(its), phi=np.concatenate(phi))


def best_record(out, nranks, tail_start=0):
    """Lowest-Phi record over all ranks.  Only records that cover an iteration >= tail_start count (default: the whole
    run).  Returns theta (float64), w, phi, rank, iteration (first iteration of the record)."""
    best = None
    for r in range(nranks):
        ch = load_chain(out, r)
        ok = (ch["start"] + ch["count"]) > int(tail_start)
        if not np.any(ok):
            continue
        phi = np.where(ok & np.isfinite(ch["phi"]), ch["phi"], np.inf)
        j = int(np.argmin(phi))
        if np.isfinite(phi[j]) and (best is None or phi[j] < best["phi"]):
            best = dict(theta=np.asarray(ch["theta"][j], float), w=float(ch["w"][j]), phi=float(phi[j]), rank=r,
                        iteration=int(max(ch["start"][j], int(tail_start))))
    if best is None:
        raise ValueError("no finite-Phi record found")
    return best


def best_fields(model, best, rank=0):
    """Forward model on the best state: log K (cells), concentration (cells x times), pressure, pressure drop, Phi."""
    st = split_state(model, best["theta"], best["w"])
    if not model.rho_ok(st):
        raise ValueError("best state has rho outside the allowed range")
    field = model.build(st)
    sim = model.simulate(field, rank, want_pressure=True)
    if sim is None or sim.conc is None or not np.isfinite(sim.dp_pa):
        raise RuntimeError("forward model failed on the best state")
    return dict(logperm=np.asarray(field.prior_field["logK"]), conc=np.asarray(sim.conc),
                pressure=np.asarray(sim.pressure), dp_pa=float(sim.dp_pa),
                phi=float(model.phi(st, field, sim)), rho=float(field.prior_field["rho"]),
                beta=float(model.beta.beta(field.prior_field["rho"])), w=float(best["w"]),
                phi_chain=float(best["phi"]), rank=int(best["rank"]), iteration=int(best["iteration"]))


def plug_mask(nx, ny, nz):
    """Cells inside the cylindrical core plug (same construction as the group's avg_3D.m): cell centre (x, y)
    within r - 0.4 h of the axis.  Vector in solver order (x fastest)."""
    i = np.arange(1, nx + 1)[:, None]
    j = np.arange(1, ny + 1)[None, :]
    inside = ((i - 0.5 - nx / 2.0) ** 2 + (j - 0.5 - ny / 2.0) ** 2) <= (nx / 2.0 - 0.4) ** 2
    return np.repeat(inside[:, :, None], nz, axis=2).reshape(-1, order="F")


class Welford:
    """Running mean and variance of arrays; `merge` combines accumulators (Chan et al.)."""

    def __init__(self, shape=None):
        self.n, self.mean, self.M2 = 0, None, None
        if shape is not None:
            self.mean, self.M2 = np.zeros(shape), np.zeros(shape)

    def add(self, x):
        x = np.asarray(x, float)
        if self.mean is None:
            self.mean, self.M2 = np.zeros_like(x), np.zeros_like(x)
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)

    def merge(self, o):
        if o.n == 0:
            return self
        if self.n == 0:
            self.n, self.mean, self.M2 = o.n, o.mean.copy(), o.M2.copy()
            return self
        n = self.n + o.n
        d = o.mean - self.mean
        self.M2 = self.M2 + o.M2 + d * d * (self.n * o.n / n)
        self.mean = self.mean + d * (o.n / n)
        self.n = n
        return self

    def std(self, ddof=1):
        return np.sqrt(self.M2 / max(self.n - ddof, 1))


def split_state(model, theta_flat, w):
    out, k = [], 0
    for d in model.block_dims:
        out.append(np.asarray(theta_flat[k:k + d], float)); k += d
    return out + [np.array([float(w)])]


def field_statistics(model, samples, comm, log=None, every=50, plug=None, grid=None):
    """Collective.  Samples are dealt round-robin to the ranks; every rank accumulates Welford statistics of
    pressure, concentration, log-permeability, and the pressure drop, and rank 0 merges them.  Returns a dict on
    rank 0 and None elsewhere."""
    rank, P = comm.Get_rank(), comm.Get_size()
    acc = dict(pressure=Welford(), conc=Welford(), logperm=Welford(), dp=Welford())
    if plug is not None:
        acc["conc_plug"], acc["conc_z"] = Welford(), Welford()
        nx, ny, nz = grid
        disc = np.asarray(plug).reshape((nx, ny, nz), order="F")[:, :, 0]
    failed = 0
    for i in range(rank, len(samples["w"]), P):
        st = split_state(model, samples["theta"][i], samples["w"][i])
        if not model.rho_ok(st):
            failed += 1
            continue
        field = model.build(st)
        sim = model.simulate(field, rank, want_pressure=True)
        if sim is None or sim.conc is None or not np.isfinite(sim.dp_pa) or sim.pressure is None:
            failed += 1
            continue
        acc["pressure"].add(sim.pressure); acc["conc"].add(sim.conc)
        acc["logperm"].add(field.prior_field["logK"]); acc["dp"].add(sim.dp_pa)
        if plug is not None:
            c4 = np.asarray(sim.conc, float).reshape((nx, ny, nz, -1), order="F")
            acc["conc_z"].add(c4[disc].mean(axis=0))            # (nz, times): mean over the disc of each layer
            acc["conc_plug"].add(c4[disc].mean(axis=(0, 1)))      # (times,)
        if log and rank == 0 and acc["dp"].n % every == 0:
            log(f"[glam-post] rank 0 processed {acc['dp'].n} samples")
    gathered = comm.gather((acc, failed), root=0)
    if rank != 0:
        return None
    tot = {k: Welford() for k in acc}
    nfail = 0
    for a, f in gathered:
        nfail += f
        for k in tot:
            tot[k].merge(a[k])
    res = {f"{k}_mean": v.mean for k, v in tot.items()}
    res.update({f"{k}_std": v.std() for k, v in tot.items()})
    res["n_used"], res["n_failed"] = tot["dp"].n, nfail
    return res
