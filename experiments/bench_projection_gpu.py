"""GPU projection benchmark: P processes share the node's GPUs round-robin (the GLAM layout: 64 ranks on 2-4 GPUs).

Every process uploads its own copy of PHI (1.2 GB) and times YP = PHI U diag(s) U^T PHI^T Y including the host<->device
copies of Y and YP (what a likelihood evaluation pays).  Needs CuPy and a GPU allocation (--gres=gpu:N).

    python experiments/bench_projection_gpu.py --procs 1 2 8 16 32 64 --phi external/matrices/phi_full_f32.npy
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _work(args):
    phi_path, m, reps, barrier, q, rank, devs = args
    os.environ["CUDA_VISIBLE_DEVICES"] = devs[rank % len(devs)]     # before CuPy is imported
    import cupy
    from brho.multiscale_mat import GpuProjector
    rng = np.random.default_rng(rank)
    phi = np.load(phi_path, mmap_mode="r")
    U = rng.standard_normal((m, m)).astype(np.float32) / np.sqrt(m)
    s = rng.random(m).astype(np.float32) + 0.5
    t0 = time.perf_counter()
    proj = GpuProjector(phi, U, s, cupy)
    cupy.cuda.Stream.null.synchronize()
    t_up = time.perf_counter() - t0
    Y = rng.standard_normal((phi.shape[0], 1)).astype(np.float32)
    proj(Y)
    barrier.wait()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); proj(Y); t.append(time.perf_counter() - t0)
    free, total = cupy.cuda.runtime.memGetInfo()
    q.put((t, t_up, (total - free) / 1e9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", required=True)
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 8, 16, 32, 64])
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devs = [d for d in vis.split(",") if d.strip()] or [str(i) for i in range(int(os.environ.get("SLURM_GPUS_ON_NODE", "1") or 1))]
    print(f"GPUs: {devs}  PHI {a.phi}")
    m = np.load(a.phi, mmap_mode="r").shape[1]
    base = None
    ctx = mp.get_context("spawn")
    for P in a.procs:
        barrier = ctx.Barrier(P); q = ctx.Queue()
        ps = [ctx.Process(target=_work, args=((a.phi, m, a.reps, barrier, q, r, devs),)) for r in range(P)]
        [p.start() for p in ps]
        res = [q.get() for _ in ps]
        [p.join() for p in ps]
        med = float(np.median(np.concatenate([r[0] for r in res])))
        base = base or med
        print(f"procs={P:3d}  median {med*1e3:8.2f} ms/call   slowdown x{med/base:5.2f}   "
              f"upload {np.median([r[1] for r in res]):5.1f} s   GPU mem in use (max) {max(r[2] for r in res):6.1f} GB", flush=True)


if __name__ == "__main__":
    main()
