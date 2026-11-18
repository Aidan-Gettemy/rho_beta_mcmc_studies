"""Tables of the +20000-iteration extension (all six arms) for glam_study80k.tex:  python extension_tables.py   (from this folder)
80000-iteration values come from the git commit before the 100000 results (35ce049~1), the new ones from the working tree."""
import json, os, subprocess
ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
OLD = "35ce049~1"
ARMS = [("1", "pcn", "pCN"), ("2", "glam1", "GLAM, 1 anchor"), ("2b", "glam1_best", "GLAM, 1 anchor + memory"), ("3", "glamN", "GLAM, many anchors"),
        ("4", "glam1_mask", "1 anchor + mask"), ("5", "glamN_mask", "many anchors + mask")]


def load(rev, tag, name):
    rel = f"runs/cluster/tables/e6_glam_s80k_{tag}/{name}"
    if rev is None:
        return json.load(open(os.path.join(ROOT, rel)))
    return json.loads(subprocess.check_output(["git", "-C", ROOT, "show", f"{rev}:{rel}"], text=True))


def row(rev, tag):
    m, f = load(rev, tag, "mpsrf.json"), load(rev, tag, "field_comparison.json")
    c, p, po = m["curves"], m["psrf_final"], f["posterior"]
    return dict(n=m["n_samples"], pmax=p["max"], b0=c["block 0"]["final_mpsrf"], f100=c["first 100"]["final_mpsrf"], rmse=f["mean"]["rmse"],
                rz=po["rms_z"], c2=po["coverage_2sd"], dre=f.get("data_re"), drm=f.get("data_rmse"))


g = lambda x: f"{x:.2f}" if x < 10 else f"{x:.1f}"
with open("tables_ext.tex", "w") as fh:
    fh.write("\\begin{tabular}{llccccccc}\\toprule\n")
    fh.write("arm & & samples/chain & max PSRF & MPSRF first 100 & MPSRF block 0 & RMSE of mean & rms $z$ & coverage $2\\sigma$ \\\\ \\midrule\n")
    for a, tag, nm in ARMS:
        o, n = row(OLD, tag), row(None, tag)
        fh.write(f"{a} & {nm} & {o['n']} $\\to$ {n['n']} & {g(o['pmax'])} $\\to$ {g(n['pmax'])} & {g(o['f100'])} $\\to$ {g(n['f100'])} & "
                 f"{g(o['b0'])} $\\to$ {g(n['b0'])} & {o['rmse']:.3f} $\\to$ {n['rmse']:.3f} & {o['rz']:.2f} $\\to$ {n['rz']:.2f} & {o['c2']:.2f} $\\to$ {n['c2']:.2f} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\n")
with open("tables_ext_data.tex", "w") as fh:
    fh.write("\\begin{tabular}{llcccc}\\toprule\n")
    fh.write("arm & & RE best & RE posterior mean & RMSE best & RMSE posterior mean \\\\ \\midrule\n")
    for a, tag, nm in ARMS:
        n = row(None, tag)
        fh.write(f"{a} & {nm} & {n['dre']['best']:.4f} & {n['dre']['mean']:.4f} & {n['drm']['best']:.4f} & {n['drm']['mean']:.4f} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\n")
print(open("tables_ext.tex").read()); print(open("tables_ext_data.tex").read())

# per-block MPSRF at 100000 (final value of the growing-window curve), as in Section A
with open("tables_ext_blocks.tex", "w") as fh:
    fh.write("\\begin{tabular}{ll" + "r" * 14 + "}\\toprule\n")
    fh.write("arm & & $\\rho$ & first 100 & " + " & ".join(f"b{i}" for i in range(12)) + " \\\\ \\midrule\n")
    for a, tag, nm in ARMS:
        c = load(None, tag, "mpsrf.json")["curves"]
        vals = [c["rho (w)"]["final_mpsrf"], c["first 100"]["final_mpsrf"]] + [c[f"block {i}"]["final_mpsrf"] for i in range(12)]
        fh.write(f"{a} & {nm} & " + " & ".join(g(v) for v in vals) + " \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\n")
print(open("tables_ext_blocks.tex").read())
