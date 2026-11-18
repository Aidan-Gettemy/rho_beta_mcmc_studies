"""Chain-level analyses of one s80k arm, run where the chains are (Juno); writes only small tables.

    python experiments/s80k_chain_analysis.py --exp-tag s80k_glam1_mask --mask runs/cluster/tables/e6_glam_s80k_pcn/keep_mask_12chains.npz

Writes into tables/e6_glam_<tag>/ :
  variance_by_index.csv   block, index, mean, variance, between_chain_var of every coordinate over the tail (pooled over chains); rho is block 12
  variance_summary.json   the same split by a common partition of the coordinates (the mask's `installed from the anchor` = constrained, `keep own` =
                          unconstrained), per block and overall: mean / median variance and the mean between-chain variance
  exchange_phi.csv        one row per exchange (found from the chains: every exchange flips the stage kind between global and local):
                          iteration, Phi of the chains just before the restart (min / median / max over chains), Phi of the restart states
                          (the record that starts at the exchange iteration; n_restart chains have one), and, when exchanges.pkl has the
                          exchange, the anchor Phi (best / median / max), the number of anchors and the candidate Phi range
  leakage_summary.json    mean over exchanges of restart Phi - anchor Phi (best and median), overall and for the last 25 % of the exchanges
The variance pass loads the coefficient arrays of one chain at a time (about 1 GB each); the exchange pass loads only the traces.
exchanges.pkl: a resumed run starts its exchange list empty and rewrites the file, so after an extension it holds only the exchanges since the
resume point; older ones can be added with --old-exchanges <pkl> (iterations present in the current file win).
"""
import _common  # noqa: F401
import argparse
import csv
import json
import os
import pickle

import numpy as np

from brho import io
from brho.glam.post import load_chain, n_iterations
from coef_variance_by_index import tail_moments


def load_exchanges(paths):
    """{iteration: archive-diagnostics dict} from one or more exchanges.pkl files (earlier files first, later ones override)."""
    out = {}
    for p in paths:
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                for e in pickle.load(f):
                    out[int(e["iteration"])] = e["archive"]
    return out


def exchange_iterations(ch):
    """Iterations at which an exchange happened: the first iteration of every stage after the first.  Stages alternate between global and local
    moves, so every exchange (the end of the burn-in included, which is global-kind) shows as a change of the per-iteration kind trace."""
    kind = np.asarray(ch["kind"])
    return (np.nonzero(np.diff(kind))[0] + 1).astype(int)


def per_exchange_phi(out, ranks, exchanges, n_expected=None):
    """Rows (one per exchange) from the traces and records of every chain; no coefficient arrays are loaded."""
    pre, post, its = [], [], None
    for r in ranks:
        ch = load_chain(out, r, theta=False)
        N = n_iterations(ch)
        if n_expected is not None and N != n_expected:
            raise ValueError(f"rank {r}: {N} iterations, expected {n_expected}")
        tr = np.asarray(ch["phi_trace"], float)
        if tr.size != N:
            raise ValueError(f"rank {r}: trace has {tr.size} entries for {N} iterations; exchange iterations cannot be read from it")
        it = exchange_iterations(ch)
        its = it if its is None else its
        if not np.array_equal(it, its):
            raise ValueError(f"rank {r}: exchange iterations differ from rank {ranks[0]}'s (chains of one run share the schedule)")
        pre.append(tr[it - 1])
        j = np.searchsorted(ch["start"], it)
        j = np.minimum(j, ch["start"].size - 1)
        ok = ch["start"][j] == it                          # no record when the first move of the stage was accepted at once
        post.append(np.where(ok, np.asarray(ch["phi"], float)[j], np.nan))
        del ch
    pre, post = np.array(pre), np.array(post)               # (chains, exchanges)
    rows = []
    for k, it in enumerate(its):
        a = exchanges.get(int(it), {})
        q = lambda v, f: float(f(v)) if np.isfinite(v).any() else np.nan  # noqa: E731
        rows.append(dict(
            exchange=k, iteration=int(it),
            pre_min=float(pre[:, k].min()), pre_median=float(np.median(pre[:, k])), pre_max=float(pre[:, k].max()),
            n_restart=int(np.isfinite(post[:, k]).sum()),
            restart_min=q(post[:, k], np.nanmin), restart_median=q(post[:, k], np.nanmedian), restart_max=q(post[:, k], np.nanmax),
            anchor_count=a.get("anchor_count", np.nan), anchor_phi_best=a.get("anchor_phi_best", np.nan),
            anchor_phi_median=a.get("anchor_phi_median", np.nan), anchor_phi_max=a.get("anchor_phi_max", np.nan),
            candidate_phi_min=a.get("candidate_phi_min", np.nan), candidate_phi_max=a.get("candidate_phi_max", np.nan)))
    return rows


def leakage_summary(rows):
    """Restart Phi minus anchor Phi, where both are known.  With one anchor the best anchor is the one every chain restarts from."""
    ok = [r for r in rows if np.isfinite(r["anchor_phi_best"]) and np.isfinite(r["restart_median"])]
    if not ok:
        return dict(n_exchanges_with_anchor=0)

    def stats(sub):
        g_best = np.array([r["restart_median"] - r["anchor_phi_best"] for r in sub])
        g_med = np.array([r["restart_median"] - r["anchor_phi_median"] for r in sub])
        return dict(n=len(sub), first_iteration=sub[0]["iteration"], last_iteration=sub[-1]["iteration"],
                    gap_to_best_mean=float(g_best.mean()), gap_to_best_median=float(np.median(g_best)), gap_to_best_max=float(g_best.max()),
                    gap_to_median_mean=float(g_med.mean()),
                    anchor_phi_best_mean=float(np.mean([r["anchor_phi_best"] for r in sub])),
                    restart_median_mean=float(np.mean([r["restart_median"] for r in sub])),
                    gain_vs_pre_mean=float(np.mean([r["pre_median"] - r["restart_median"] for r in sub])))
    last = ok[int(0.75 * len(ok)):]
    return dict(all=stats(ok), last_quarter=stats(last), single_anchor=bool(all(r["anchor_count"] == 1 for r in ok)))


def variance_summary(var, mean, between, mask_path, n_sub, m):
    z = np.load(mask_path)
    keep = np.concatenate([np.asarray(z[f"m{b}"], bool).ravel() for b in range(n_sub)])        # True = unconstrained (the chain keeps its own value)
    v, bt = var[:n_sub * m], between[:n_sub * m]

    def part(sel):
        return dict(n=int(sel.sum()), variance_mean=float(v[sel].mean()), variance_median=float(np.median(v[sel])),
                    between_chain_var_mean=float(bt[sel].mean()))
    per_block = []
    for b in range(n_sub):
        s = np.zeros(n_sub * m, bool); s[b * m:(b + 1) * m] = True
        per_block.append(dict(block=b, constrained=part(s & ~keep), unconstrained=part(s & keep)))
    return dict(mask=mask_path, overall=dict(constrained=part(~keep), unconstrained=part(keep)), per_block=per_block,
                rho=dict(variance=float(var[-1]), mean=float(mean[-1]), between_chain_var=float(between[-1])))


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-tag", required=True, help="e.g. s80k_glam1_mask (results folder e6_glam_<tag>)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out-dir", default=None, help="folder with chains/ (default results/e6_glam_<tag>)")
    ap.add_argument("--mask", required=True, help="keep_mask npz that defines the constrained / unconstrained partition (same for every arm)")
    ap.add_argument("--n-sub", type=int, default=12)
    ap.add_argument("--chains", type=int, default=24)
    ap.add_argument("--tail-start", type=int, default=40000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=None, help="check that every chain has exactly this many iterations")
    ap.add_argument("--old-exchanges", nargs="*", default=[], help="older exchanges.pkl files whose exchanges are added")
    ap.add_argument("--skip-variance", action="store_true")
    args = ap.parse_args()
    if args.root:
        os.environ["BRHO_ROOT"] = args.root
    elif "BRHO_ROOT" not in os.environ and os.path.isdir(os.path.join(io.REPO, "runs", "cluster", "results")):
        os.environ["BRHO_ROOT"] = os.path.join(io.REPO, "runs", "cluster")
    exp = io.EXP_GLAM + f"_{args.exp_tag}"
    out = args.out_dir or io.results_dir(exp)
    tables = io.tables_dir(exp)
    os.makedirs(tables, exist_ok=True)
    ranks = list(range(args.chains))

    if not args.skip_variance:
        mean, var, between = tail_moments(out, ranks, args.tail_start, args.stride, args.n_sub)
        m = (len(var) - 1) // args.n_sub
        rows = [dict(block=(j // m if j < args.n_sub * m else args.n_sub), index=(j % m if j < args.n_sub * m else 0),
                     mean=mean[j], variance=var[j], between_chain_var=between[j]) for j in range(len(var))]
        _write_csv(os.path.join(tables, "variance_by_index.csv"), rows)
        s = variance_summary(var, mean, between, args.mask, args.n_sub, m)
        with open(os.path.join(tables, "variance_summary.json"), "w") as f:
            json.dump(dict(tail_start=args.tail_start, stride=args.stride, chains=args.chains, **s), f, indent=1)
        o = s["overall"]
        print(f"{exp}: variance, constrained {o['constrained']['variance_mean']:.3f} ({o['constrained']['n']} coords), "
              f"unconstrained {o['unconstrained']['variance_mean']:.3f} ({o['unconstrained']['n']} coords)")

    exch = load_exchanges(list(args.old_exchanges) + [os.path.join(out, "exchanges.pkl")])
    rows = per_exchange_phi(out, ranks, exch, args.n_iter)
    if len(rows) == 0:
        print(f"{exp}: no exchanges in the chains (plain pCN): exchange and leakage tables skipped")
        return
    _write_csv(os.path.join(tables, "exchange_phi.csv"), rows)
    leak = leakage_summary(rows)
    leak["exchanges_in_chains"] = len(rows)
    leak["exchanges_with_anchor_info"] = int(sum(np.isfinite(r["anchor_phi_best"]) for r in rows))
    with open(os.path.join(tables, "leakage_summary.json"), "w") as f:
        json.dump(leak, f, indent=1)
    print(f"{exp}: {len(rows)} exchanges from the chains, {leak['exchanges_with_anchor_info']} with anchor information")
    if "all" in leak:
        print("  restart Phi - best anchor Phi (mean over exchanges):", round(leak["all"]["gap_to_best_mean"], 2),
              "; last quarter:", round(leak["last_quarter"]["gap_to_best_mean"], 2))
    print("written:", tables)


if __name__ == "__main__":
    main()
