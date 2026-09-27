"""
Where things go.

    results/<exp>/     raw outputs (npz, checkpoints, manifest.json)
    figures/<exp>/     PNGs written by reports/*
    tables/<exp>/      CSV + Markdown + JSON stats written by reports/*

Override the roots with BRHO_ROOT (e.g. to keep quick runs separate:
BRHO_ROOT=runs/quick).
"""
import json
import os
import platform
import sys
import time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXP_GRID = "e2_grid"
EXP_CURVE = "e3_curve"
EXP_SAMPLER = "e4_sampler"
EXP_PRIOR = "e4_prior_test"
EXP_SETUP = "setup"


def root():
    return os.environ.get("BRHO_ROOT", REPO)


def _dir(kind, exp):
    d = os.path.join(root(), kind, exp)
    os.makedirs(d, exist_ok=True)
    return d


def results_dir(exp):
    return _dir("results", exp)


def figures_dir(exp):
    return _dir("figures", exp)


def tables_dir(exp):
    return _dir("tables", exp)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(_jsonable(obj), fh, indent=1)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def write_manifest(exp, configs, extra=None):
    """configs: dict name -> asdict(config)."""
    man = dict(experiment=exp, time=time.strftime("%Y-%m-%d %H:%M:%S"),
               argv=sys.argv, python=platform.python_version(),
               numpy=np.__version__, configs=configs)
    if extra:
        man.update(extra)
    save_json(os.path.join(results_dir(exp), "manifest.json"), man)
    return man


def read_manifest(exp):
    return load_json(os.path.join(results_dir(exp), "manifest.json"))


def write_table(exp, name, rows, columns=None, floatfmt="{:.4g}"):
    """rows: list of dicts -> tables/<exp>/<name>.csv and .md"""
    if not rows:
        return
    columns = columns or list(rows[0].keys())
    fmt = lambda v: (floatfmt.format(v) if isinstance(v, (float, np.floating))
                     else str(v))
    d = tables_dir(exp)
    with open(os.path.join(d, name + ".csv"), "w") as fh:
        fh.write(",".join(columns) + "\n")
        for r in rows:
            fh.write(",".join(fmt(r.get(c, "")) for c in columns) + "\n")
    with open(os.path.join(d, name + ".md"), "w") as fh:
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("|" + "---|" * len(columns) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |\n")
