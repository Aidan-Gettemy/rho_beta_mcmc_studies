"""Replay the GLAM archive selection offline (no MPI, no solver, no cluster time) on the anchor set a run saved.

    python experiments/replay_archive_selection.py --exp-tag r24m2
    python experiments/replay_archive_selection.py --exp-tag r24m2 --eps 1e-6,1e-2,1e-1 --scale 1,10 --mind 3,12 --cap none,60

It loads the newest <results>/anchor_state/anchor_state_ite*_global.pkl (the archive after the last exchange), takes the
settings of the run from <results>/manifest.json, and reruns GlobalLocalAnchorManager._build_diversity_anchor_set on the
saved anchors for every combination of the settings below (the cross product), always starting with the run's own settings
("baseline").  The pool is the saved archive, so a replay can only THIN it: it answers what a different covariance
regularization (eps), log-det gain scale, minimum distance, minimum score or cap would have kept, not what other
candidates the chains would have produced.

  --eps       covariance_regularization of the log-det gain          (run: 1e-6)
  --scale     logdet_gain_scale                                        (run: 10)
  --mind      minimum_anchor_distance                                  (run: 3)
  --cap       max_anchors, none = unlimited (write --cap none,60)       (run: none)
  --minscore  minimum_diversity_score                                  (run: 0.5)
  --alpha     diversity_alpha (weight of the log-det novelty)          (run: 0.85)

Per setting it prints and writes tables/<study>/replay_selection.csv/.md:
  n, cap_hit, rej_score           archive size; stopped by the cap; admissible candidates left when the score cut-off stopped it
  score_min/mean/last             score of the selected anchors (last = mean over the last 10 % picked: how close the cut-off is)
  lognov_mean, gain_last          mean log-det novelty; log-det gain (nats) of the last picks = -scale*ln(1-novelty)
  eff_rank, top_share             effective rank of the selected set's covariance; share of its largest eigenvalue
  dmin, dmed, dmax                pairwise distances of the selected anchors
  cover_med, cover_max            distance from every anchor of the pool to the nearest selected one (small = the pool is spanned)
  phi_med, phi_max                Phi of the selected anchors
  secs                            wall time of the selection
BLAS threads are pinned to 1 by _common; export OMP_NUM_THREADS=8 (or more) first to speed up the uncapped settings.
"""
import _common  # noqa: F401
import argparse
import itertools
import os
import pickle
import re
import subprocess
import time

import numpy as np

from brho import config as C
from brho import io
from brho.glam.anchors import flatten_theta
from brho.glam.sampler import make_manager

STATE_RE = re.compile(r"^anchor_state_ite(\d+)_global\.pkl$")
SETTING_KEYS = ("eps", "scale", "mind", "cap", "minscore", "alpha")


def load_state(state_dir, iteration=None):
    """Newest (or the requested) global anchor-state pickle in state_dir -> (state dict, iteration)."""
    found = {}
    for fn in os.listdir(state_dir):
        m = STATE_RE.match(fn)
        if m:
            found[int(m.group(1))] = fn
    if not found:
        raise SystemExit(f"no anchor_state_ite*_global.pkl in {state_dir}")
    it = max(found) if iteration is None else int(iteration)
    if it not in found:
        raise SystemExit(f"iteration {it} not in {state_dir}; available: {sorted(found)}")
    with open(os.path.join(state_dir, found[it]), "rb") as f:
        return pickle.load(f), it


def cfg_from_sbatch(path):
    """The GlamConfig a run was launched with, from the GLAM_FLAGS array of its sbatch script (used when the run's
    manifest.json is missing from the results folder)."""
    import run_glam
    out = subprocess.run(["bash", "-c", f'source <(sed -n "/^GLAM_FLAGS=(/,/^)/p" "{path}"); printf "%s\\n" "${{GLAM_FLAGS[@]}}"'],
                         capture_output=True, text=True, check=True).stdout.split("\n")[:-1]
    return C.from_cli(run_glam.build_parser().parse_args(out), C.GlamConfig)


def archive_metrics(X, pool_X):
    """Geometry of the selected vectors X (n x d) and how well they span the pool (N x d)."""
    n = X.shape[0]
    out = dict(eff_rank=np.nan, top_share=np.nan, dmin=np.nan, dmed=np.nan, dmax=np.nan, cover_med=np.nan, cover_max=np.nan)
    if n >= 3:
        Xc = X - X.mean(axis=0)
        lam = np.clip(np.linalg.eigvalsh(Xc @ Xc.T), 0.0, None)
        if lam.sum() > 0:
            p = lam / lam.sum()
            q = p[p > 0]
            out["eff_rank"] = float(np.exp(-np.sum(q * np.log(q))))
            out["top_share"] = float(p.max())
        sq = np.sum(X ** 2, axis=1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
        iu = np.triu_indices(n, 1)
        d = np.sqrt(d2[iu])
        out.update(dmin=float(d.min()), dmed=float(np.median(d)), dmax=float(d.max()))
    c2 = np.maximum(np.sum(pool_X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2.0 * pool_X @ X.T, 0.0).min(axis=1)
    c = np.sqrt(c2)
    out.update(cover_med=float(np.median(c)), cover_max=float(c.max()))
    return out


def apply_setting(mgr, s, run_cfg):
    mgr.covariance_regularization = float(s["eps"])
    mgr.logdet_gain_scale = float(s["scale"])
    mgr.minimum_anchor_distance = float(s["mind"])
    mgr.max_anchors = None if s["cap"] < 0 else int(s["cap"])
    mgr.minimum_diversity_score = float(s["minscore"])
    mgr.diversity_alpha = float(s["alpha"])
    mgr.archive_selection_temperature = float(run_cfg.archive_selection_temperature)


def replay(cfg, state, settings, pool_size=0, seed=0, log=print):
    """Run the selection once per setting on the saved archive.  cfg = the run's GlamConfig; settings = list of dicts
    with SETTING_KEYS (+ "label").  Returns a list of result rows."""
    n_sub = int(state["numsubd"])
    pool = [a.clone() for a in state["anchors"]]
    best = state["best_anchor"].clone()
    rng = np.random.default_rng(seed)
    if pool_size and len(pool) > pool_size:
        keep = rng.choice(len(pool), size=pool_size, replace=False)
        pool = [pool[i] for i in sorted(keep)]
    mgr = make_manager(cfg, n_sub, rank=0)
    nstate, weights = mgr.nstate, mgr.block_weights
    pool_X = np.vstack([flatten_theta(a.theta, nstate, weights) for a in pool])
    rows = []
    for k, s in enumerate(settings):
        apply_setting(mgr, s, cfg)
        mgr.rng = np.random.default_rng(seed + 1)
        t0 = time.perf_counter()
        sel = mgr._build_diversity_anchor_set(best.clone(), [a.clone() for a in pool])
        secs = time.perf_counter() - t0
        X = np.vstack([flatten_theta(a.theta, nstate, weights) for a in sel])
        dg = mgr._selection_diagnostics
        sc = np.asarray(dg["selected_scores"], dtype=float)
        ln = np.asarray(dg["selected_logdet_novelty"], dtype=float)
        tail = max(1, len(sc) // 10)
        gain_last = (-s["scale"] * np.log(np.maximum(1.0 - ln[-tail:], 1e-300))).mean() if len(ln) else np.nan
        phis = np.array([a.phi for a in sel])
        row = dict(label=s["label"], **{key: s[key] for key in SETTING_KEYS}, n=len(sel), pool=len(pool),
                   cap_hit=int(bool(dg["archive_hit_cap"])), rej_score=int(dg["score_rejected_final"]),
                   score_min=float(sc.min()) if len(sc) else np.nan, score_mean=float(sc.mean()) if len(sc) else np.nan,
                   score_last=float(sc[-tail:].mean()) if len(sc) else np.nan,
                   lognov_mean=float(ln.mean()) if len(ln) else np.nan, gain_last=float(gain_last),
                   **archive_metrics(X, pool_X), phi_med=float(np.median(phis)), phi_max=float(phis.max()), secs=secs)
        rows.append(row)
        log(f"[{k + 1}/{len(settings)}] {s['label']:<9} eps={s['eps']:g} scale={s['scale']:g} mind={s['mind']:g} cap={s['cap']:g} "
            f"minscore={s['minscore']:g} -> n={row['n']} rej_score={row['rej_score']} score_last={row['score_last']:.3f} "
            f"eff_rank={row['eff_rank']:.2f} top={row['top_share']:.2f} cover_max={row['cover_max']:.1f} ({secs:.0f}s)")
    return rows


def grid(args, cfg):
    def parse(text, default):
        return [-1.0 if x.strip().lower() in ("none", "inf") else float(x) for x in text.split(",")] if text else [default]
    vals = dict(eps=parse(args.eps, cfg.covariance_regularization), scale=parse(args.scale, cfg.logdet_gain_scale),
                mind=parse(args.mind, cfg.minimum_anchor_distance), cap=parse(args.cap, float(cfg.max_anchors)),
                minscore=parse(args.minscore, cfg.minimum_diversity_score), alpha=parse(args.alpha, cfg.diversity_alpha))
    base = dict(label="baseline", eps=cfg.covariance_regularization, scale=cfg.logdet_gain_scale, mind=cfg.minimum_anchor_distance,
                cap=float(cfg.max_anchors), minscore=cfg.minimum_diversity_score, alpha=cfg.diversity_alpha)
    out, seen = [base], {tuple(base[k] for k in SETTING_KEYS)}
    for combo in itertools.product(*(vals[k] for k in SETTING_KEYS)):
        if combo not in seen:
            seen.add(combo)
            out.append(dict(label=f"s{len(out)}", **dict(zip(SETTING_KEYS, combo))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-tag", default="")
    ap.add_argument("--root", default=None, help="BRHO_ROOT (default: $BRHO_ROOT, else runs/cluster if it exists, else the repo)")
    ap.add_argument("--state-dir", default=None, help="folder with anchor_state_ite*_global.pkl (default <results>/anchor_state)")
    ap.add_argument("--iteration", type=int, default=None, help="snapshot iteration (default: the newest)")
    ap.add_argument("--sbatch", default=None, help="sbatch script holding the run's GLAM_FLAGS, used when the results folder has no "
                                                   "manifest.json (default cluster/slurm/glam_<exp-tag>.sbatch)")
    ap.add_argument("--pool-size", type=int, default=0, help="subsample the pool to this many anchors (speed); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", default="1e-6,1e-2,1e-1")
    ap.add_argument("--scale", default="1,10")
    ap.add_argument("--mind", default="3,12")
    ap.add_argument("--cap", default="none,60", help="max_anchors list; none = unlimited (a bare -1 would be read as an option: use --cap=-1,60)")
    ap.add_argument("--minscore", default="")
    ap.add_argument("--alpha", default="")
    args = ap.parse_args()
    if args.root:
        os.environ["BRHO_ROOT"] = args.root
    elif "BRHO_ROOT" not in os.environ and os.path.isdir(os.path.join(io.REPO, "runs", "cluster", "results")):
        os.environ["BRHO_ROOT"] = os.path.join(io.REPO, "runs", "cluster")
    exp = io.EXP_GLAM + (f"_{args.exp_tag}" if args.exp_tag else "")
    rdir = io.results_dir(exp)
    manifest = os.path.join(rdir, "manifest.json")
    if os.path.exists(manifest):
        cfg = C.from_dict(C.GlamConfig, io.read_manifest(exp)["configs"]["GlamConfig"])
        print(f"settings of the run: {manifest}")
    else:
        sb = args.sbatch or os.path.join(io.REPO, "cluster", "slurm", f"glam_{args.exp_tag}.sbatch")
        if not os.path.exists(sb):
            raise SystemExit(f"{manifest} not found and no sbatch script {sb} to read the run's settings from (pass --sbatch)")
        cfg = cfg_from_sbatch(sb)
        print(f"{manifest} not found; settings of the run read from {sb}")
    state_dir = args.state_dir or os.path.join(rdir, "anchor_state")
    if not os.path.isdir(state_dir):
        raise SystemExit(f"{state_dir} not found (wrong folder? pass --root or --state-dir; push it from Juno with git add -f)")
    state, it = load_state(state_dir, args.iteration)
    print(f"{exp}: anchor state of iteration {it}: {len(state['anchors'])} anchors, best Phi {state['best_anchor'].phi:.1f}, "
          f"numsubd {state['numsubd']}")
    settings = grid(args, cfg)
    print(f"{len(settings)} settings (baseline first)\n")
    rows = replay(cfg, state, settings, pool_size=args.pool_size, seed=args.seed)
    cols = ["label", *SETTING_KEYS, "n", "pool", "cap_hit", "rej_score", "score_min", "score_mean", "score_last", "lognov_mean",
            "gain_last", "eff_rank", "top_share", "dmin", "dmed", "dmax", "cover_med", "cover_max", "phi_med", "phi_max", "secs"]
    os.makedirs(io.tables_dir(exp), exist_ok=True)
    io.write_table(exp, "replay_selection", rows, columns=cols)
    print("\nwritten:", os.path.join(io.tables_dir(exp), "replay_selection.csv"))


if __name__ == "__main__":
    main()
