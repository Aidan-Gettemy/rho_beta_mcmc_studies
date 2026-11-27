"""Writes the LaTeX tables of the first paper from runs/msm_local_kle_2026-11/dd_end/master_table.csv (no typed numbers).
Run: python3 paper/make_tables.py   -> paper/tab_ratios.tex, paper/tab_overlap.tex"""
import os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "runs", "msm_local_kle_2026-11", "dd_end")
d = pd.read_csv(os.path.join(D, "master_table.csv"))
d = d[(d.truth == 2026) & (d.eps_factor == 1.0)]

def f(x, n=2):
    return "--" if pd.isna(x) else f"{x:.{n}f}"

rows = []
for ell in (0.2, 0.1, 0.05):
    for flux in (False, True):
        for nb in (2, 4, 8):
            if ell == 0.2 and nb == 8:
                continue                                  # a block would have two cells
            s = d[(d.ell == ell) & (d.flux == flux) & (d.blocks == nb) & ~d.label.str.contains("-DD-H")]
            m, p = s[s.variant == "msm"], s[s.variant == "primal"]
            if m.empty or p.empty:
                continue
            m, p = m.iloc[0], p.iloc[0]
            rows.append((ell, "pressure + flux" if flux else "pressure", nb, m.err_ratio, m.sd_ratio, p.err_ratio, p.sd_ratio,
                         max(m.sup_mpsrf, p.sup_mpsrf)))
with open(os.path.join(HERE, "tab_ratios.tex"), "w") as o:
    o.write("\\begin{tabular}{llcrrrrr}\n\\toprule\n$\\ell$ & data & blocks & \\multicolumn{2}{c}{MSM basis (global pCN)} & "
            "\\multicolumn{2}{c}{DD primal} & sup-MPSRF \\\\\n & & per side & err & sd & err & sd & (max of the two) \\\\\n\\midrule\n")
    last = None
    for ell, dt, nb, me, ms, pe, ps, mp in rows:
        if last is not None and (ell, dt) != last:
            o.write("\\addlinespace\n")
        last = (ell, dt)
        o.write(f"{ell:g} & {dt} & {nb} & {f(me)} & {f(ms)} & {f(pe)} & {f(ps)} & {f(mp, 3)} \\\\\n")
    o.write("\\bottomrule\n\\end{tabular}\n")

ov = d[(d.ell.isin([0.1, 0.05])) & (d.flux == True) & (d.blocks == 4) & (d.variant == "primal")]
with open(os.path.join(HERE, "tab_overlap.tex"), "w") as o:
    o.write("\\begin{tabular}{llrrrrr}\n\\toprule\n$\\ell$ & run & err ratio & sd ratio & 90\\% cover & sup-MPSRF & iterations to MPSRF $<1.2$ \\\\\n\\midrule\n")
    for _, r in ov.sort_values(["ell", "label"], ascending=[False, True]).iterrows():
        if r.label.endswith("-H4"):
            continue                                       # withdrawn: no flux on its boundaries (LOG.md 2026-11-18)
        name = "overlap, halo reaches dual lines" if r.label.endswith("-H") else "non-overlapping"
        ci = "--" if pd.isna(r.conv_it) else f"{int(r.conv_it):,}".replace(",", "\\,")
        o.write(f"{r.ell:g} & {name} & {f(r.err_ratio)} & {f(r.sd_ratio)} & {f(r.cover90)} & {f(r.sup_mpsrf, 3)} & {ci} \\\\\n")
    o.write("\\bottomrule\n\\end{tabular}\n")
print("written")


# ---------------------------------------------------------------------------------------------------------------
# Iterations to convergence for every method of an experiment group (same ell, data type, blocks), truth 2026.
# Columns: iterations to sup-MPSRF < 1.2, ratio to the global pCN of the group, frozen pCN step, acceptance after burn-in,
# error ratio of the posterior mean, and the average relative pressure-data error of the posterior fields with its ratio
# to the global run (global/MSM runs: median over saved chain states of the relative data error, summary.json
# post_relerr_median_of_chain_medians; DD runs: median over assembled fields, post_summary assembled_global_relerr).
import json
ROOT = os.path.abspath(os.path.join(HERE, ".."))

def run_stats(r):
    """step, acceptance, avg pressure relative error of one master_table row."""
    s = json.load(open(os.path.join(ROOT, r.dir, "summary.json")))
    if r.variant in ("global", "msm"):
        return float(s["s_frozen"]), float(s["post_acc_mean"]), float(s["post_relerr_median_of_chain_medians"])
    steps = [b["s_used"] for b in s["blocks"]]
    ps = json.load(open(os.path.join(ROOT, r.dir, "post", "post_summary.json")))
    g = ps["grids"][r.variant]
    return float(np.median(steps)), float(g["acceptance"]["mean"]), float(g["assembled_global_relerr"]["median"])

def it(x):
    return "--" if pd.isna(x) else f"{int(round(x)):,}".replace(",", "\\,")

def sci(x):
    return f"{x:.3f}"

NAMES = {"global": "global pCN", "msm": "MSM basis, global pCN", "primal": "DD primal", "dual": "DD dual"}
def method_name(r):
    return "DD primal, overlap" if r.label.endswith("-DD-H") else NAMES[r.variant]

def iter_table(blocks_wanted, fname):
    out = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
           "group & method & iters to & ratio to & step & accept & err & avg pressure & ratio to \\\\",
           " & & MPSRF $<1.2$ & global & $\\beta$ & rate & ratio & rel.\\ error & global \\\\", "\\midrule"]
    for ell in (0.2, 0.1, 0.05):
        for flux in (False, True):
            g = d[(d.ell == ell) & (d.flux == flux) & (d.variant == "global")]
            if g.empty:
                continue
            g = g.iloc[0]
            gst = run_stats(g)
            first = True
            for nb in blocks_wanted:
                s = d[(d.ell == ell) & (d.flux == flux) & (d.blocks == nb) & d.variant.isin(["msm", "primal", "dual"])
                      & ~d.label.str.endswith("-DD-H4")]
                if s.empty or (ell == 0.2 and nb == 8):
                    continue
                rows = [("global pCN", g, gst)] if first else []
                first = False
                order = {"msm": 0, "primal": 1, "dual": 2}
                for _, r in s.assign(o=s.variant.map(order) + s.label.str.endswith("-DD-H") * 0.5).sort_values("o").iterrows():
                    rows.append((method_name(r), r, run_stats(r)))
                grp = f"$\\ell={ell:g}$, {'pressure+flux' if flux else 'pressure'}, ${int(nb)}\\times{int(nb)}$"
                for i, (nm, r, (b_, a_, e_)) in enumerate(rows):
                    isg = r.variant == "global"
                    ratio = "--" if (pd.isna(r.conv_it) or isg) else f"{r.conv_it / g.conv_it:.3g}"
                    out.append(f"{grp if i == 0 else ''} & {nm} & {it(r.conv_it)} & {ratio} & {b_:.3g} & {a_:.3f} & "
                               f"{'1' if isg else f(r.err_ratio)} & {sci(e_)} & {'1' if isg else f'{e_ / gst[2]:.2f}'} \\\\")
                out.append("\\addlinespace")
    out += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(HERE, fname), "w").write("\n".join(out) + "\n")

iter_table([4], "tab_iterations.tex")
iter_table([2, 4, 8], "tab_iterations_full.tex")
print("iteration tables written")
