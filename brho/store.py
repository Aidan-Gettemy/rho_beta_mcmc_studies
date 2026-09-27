"""
Build-once store for expensive, deterministic objects of a study: the KLE
basis and the synthetic reference field.

    $BRHO_ROOT/store/<kind>/<hash>.npz     the object
    $BRHO_ROOT/store/<kind>/<hash>.json    the settings it was built from

The hash covers exactly the settings that determine the object (e.g. grid,
kernel, ell, energy, method, quadrature for a KLE), so every experiment of a
study (E2, E3, E4 setups) that asks for the same thing loads the same file:
the curve and the sampler share one basis by construction.  A lock file makes
concurrent setup jobs wait for the first builder instead of racing.
"""
import hashlib
import json
import os
import time


def key_hash(key):
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def get_or_build(root, kind, key, build, save, load, wait_s=4 * 3600, poll_s=2.0,
                 log=print):
    """Return load(path) if the object exists, else build(), save(obj, path)."""
    d = os.path.join(root, kind)
    os.makedirs(d, exist_ok=True)
    h = key_hash(key)
    path = os.path.join(d, h + ".npz")
    lock = os.path.join(d, h + ".lock")
    t0 = time.time()
    while True:
        if os.path.exists(path):
            log(f"store: loaded {kind} {h}")
            return load(path), path
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"{lock} held for {wait_s} s; remove it if "
                                   "no build is running")
            time.sleep(poll_s)
            continue
        try:
            os.write(fd, f"{os.getpid()} {time.ctime()}\n".encode())
            if os.path.exists(path):                 # built while we waited
                continue
            log(f"store: building {kind} {h} ...")
            tb = time.time()
            obj = build()
            tmp = path + ".tmp.npz"
            save(obj, tmp)
            os.replace(tmp, path)
            with open(os.path.join(d, h + ".json"), "w") as fh:
                json.dump(dict(key=key, seconds=time.time() - tb,
                               built=time.ctime()), fh, indent=1)
            log(f"store: built {kind} {h} in {time.time() - tb:.1f} s")
            return obj, path
        finally:
            os.close(fd)
            if os.path.exists(lock):
                os.remove(lock)
