"""Bootstrap shared by the experiment runners."""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from brho import config as C  # noqa: E402


def parser(desc, *cfg_classes):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--quick", action="store_true",
                    help="tiny budgets for a smoke test")
    C.add_cli(ap, C.ProblemConfig)
    for cls in cfg_classes:
        C.add_cli(ap, cls)
    return ap


def configs(args, *cfg_classes):
    pc = C.from_cli(args, C.ProblemConfig)
    return (pc,) + tuple(C.from_cli(args, cls, quick=args.quick)
                         for cls in cfg_classes)
