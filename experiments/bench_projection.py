"""Benchmark the whitened projection YP = PHI U diag(s) U^T PHI^T Y  (V0: 'check before correcting').

One process per CPU is the Juno layout (64 chains), all reading the same PHI (n_cells x n_modes, float32).
The product is memory-bandwidth bound, so the question is how the per-call time grows with the number of
concurrent processes on one node.  Run with the real PHI (--phi path.npy) on a Juno node, or with a random
one of the same size (default) to measure the machine.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/bench_projection.py --procs 1 2 4 8 16 32 64
"""
import argparse, multiprocessing as mp, os, time
import numpy as np

def _work(args):
    phi_path, m, reps, barrier, q, private = args
    # private: the real --phi-mode ram layout (every process owns a first-touch copy); default: shared memory map
    phi = np.load(phi_path) if private else np.load(phi_path, mmap_mode="r")
    rng = np.random.default_rng(os.getpid())
    n = phi.shape[0]
    U = rng.standard_normal((m, m)).astype(np.float32) / np.sqrt(m)   # stands in for eigvec
    s = rng.random(m).astype(np.float32) + 0.5
    Y = rng.standard_normal(n).astype(np.float32)
    def proj(Y):
        z = phi.T @ Y
        z = U @ (s * (U.T @ z))
        return phi @ z
    proj(Y)                                   # warm up / page in
    barrier.wait()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); proj(Y); t.append(time.perf_counter() - t0)
    q.put(t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", default=None, help="PHI .npy (cells x modes); default: random of the real size")
    ap.add_argument("--n", type=int, default=64980); ap.add_argument("--m", type=int, default=4699)
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tmp", default="/tmp")
    ap.add_argument("--private", action="store_true", help="each process loads its own copy of PHI (= --phi-mode ram); 64 x 1.2 GB")
    a = ap.parse_args()
    path = a.phi
    if path is None:
        path = os.path.join(a.tmp, "phi_bench.npy")
        if not os.path.exists(path):
            mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(a.n, a.m))
            rng = np.random.default_rng(0)
            for i in range(0, a.n, 4096):
                mm[i:i + 4096] = rng.standard_normal((min(4096, a.n - i), a.m)).astype(np.float32)
            mm.flush(); del mm
    m = np.load(path, mmap_mode="r").shape[1]
    gb = 2 * np.load(path, mmap_mode="r").nbytes / 1e9 + 2 * 4 * m * m / 1e9
    print(f"PHI {path}  {'private copies' if a.private else 'shared mmap'}  traffic/call ~ {gb:.2f} GB (PHI twice + U twice)")
    base = None
    for P in a.procs:
        ctx = mp.get_context("fork"); barrier = ctx.Barrier(P); q = ctx.Queue()
        ps = [ctx.Process(target=_work, args=((path, m, a.reps, barrier, q, a.private),)) for _ in range(P)]
        [p.start() for p in ps]; ts = [q.get() for _ in ps]; [p.join() for p in ps]
        med = float(np.median(np.concatenate(ts)))
        base = base or med
        print(f"procs={P:3d}  median {med*1e3:8.1f} ms/call   slowdown x{med/base:5.2f}   aggregate {P*gb/med:7.1f} GB/s")

if __name__ == "__main__":
    main()
