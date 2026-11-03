"""Minimal runner (no pytest needed): executes every test_* function in tests/test_*.py,
or only the files named on the command line, e.g.  python tests/run_tests.py test_dirichlet test_msm_local"""
import glob
import sys
import importlib.util
import os
import time
import traceback

here = os.path.dirname(os.path.abspath(__file__))
fails = 0
want = [a.replace(".py", "") for a in sys.argv[1:]]
for path in sorted(glob.glob(os.path.join(here, "test_*.py"))):
    if want and os.path.basename(path)[:-3] not in want:
        continue
    spec = importlib.util.spec_from_file_location("t", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        t0 = time.time()
        try:
            getattr(mod, name)()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)")
        except Exception:
            fails += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
print("all passed" if not fails else f"{fails} failed")
raise SystemExit(bool(fails))
