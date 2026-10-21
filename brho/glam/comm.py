"""Communicator layer for GLAM.

The sampler and the anchor manager only use the small subset of the mpi4py communicator API listed in
`REQUIRED`.  On the cluster that is `mpi4py.MPI.COMM_WORLD`; for tests and single-process debugging this module provides

  * SerialComm      one rank
  * ThreadWorld     P ranks as threads of one process, with MPI semantics (objects are copied on send)

so the exchange / restart / adaptation logic can be tested without an MPI installation.  Reductions use the lower-case
(pickle) `allreduce`, which for NumPy arrays is the elementwise sum under both mpi4py and ThreadComm.
"""
import copy
import threading

REQUIRED = ("Get_rank", "Get_size", "Barrier", "gather", "bcast", "allreduce", "allgather")


def world():
    """mpi4py COMM_WORLD if mpi4py is importable, else a SerialComm."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except Exception:
        return SerialComm()


class SerialComm:
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def Barrier(self): pass
    def gather(self, obj, root=0): return [copy.deepcopy(obj)]
    def allgather(self, obj): return [copy.deepcopy(obj)]
    def bcast(self, obj, root=0): return copy.deepcopy(obj)
    def allreduce(self, obj, op=None): return copy.deepcopy(obj)


class ThreadWorld:
    """P ranks as threads.  `run(fn, *args)` calls fn(comm, *args) on every rank and returns the list of results
    (an exception on any rank aborts the barrier on all ranks and is re-raised)."""

    def __init__(self, size, timeout=120.0):
        self.size = int(size)
        self.timeout = timeout
        self._barrier = threading.Barrier(self.size, timeout=timeout)
        self._slots = [None] * self.size

    def comm(self, rank):
        return ThreadComm(self, rank)

    def run(self, fn, *args, **kw):
        out, errs = [None] * self.size, [None] * self.size

        def target(r):
            try:
                out[r] = fn(self.comm(r), *args, **kw)
            except BaseException as e:      # noqa: BLE001
                errs[r] = e
                self._barrier.abort()

        ts = [threading.Thread(target=target, args=(r,), daemon=True) for r in range(self.size)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        real = [e for e in errs if e is not None and not isinstance(e, threading.BrokenBarrierError)]
        if real:
            raise real[0]
        if any(e is not None for e in errs):
            raise errs[[e is not None for e in errs].index(True)]
        return out


class ThreadComm:
    def __init__(self, world, rank):
        self._w, self._r = world, int(rank)

    def Get_rank(self): return self._r
    def Get_size(self): return self._w.size
    def Barrier(self): self._w._barrier.wait()

    def _exchange(self, obj):
        w = self._w
        w._slots[self._r] = copy.deepcopy(obj)
        w._barrier.wait()
        vals = [copy.deepcopy(v) for v in w._slots]
        w._barrier.wait()                    # nobody overwrites a slot before everyone has read
        return vals

    def allgather(self, obj): return self._exchange(obj)
    def gather(self, obj, root=0):
        v = self._exchange(obj)
        return v if self._r == root else None
    def bcast(self, obj, root=0): return self._exchange(obj if self._r == root else None)[root]
    def allreduce(self, obj, op=None):
        vals = self._exchange(obj)
        acc = vals[0]
        for v in vals[1:]:
            acc = acc + v
        return acc
