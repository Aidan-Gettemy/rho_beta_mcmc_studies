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
