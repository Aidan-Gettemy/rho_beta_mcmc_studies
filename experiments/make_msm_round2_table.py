"""Collect every MSM run's summary.json + post/post_summary.json into one table
(tables/msm_round2/msm_round2_table.{json,md,tex}). Reads only committed files."""
import json, os, sys
R = os.path.join(os.environ.get("BRHO_ROOT", "runs/msm_local_kle_2026-11"))
ORDER = [("exp1_global_s50","1 global"),
 ("exp2_primal_raw_s50","2 primal raw d4"),
 ("exp2_6_primal_raw_d6","2.6 primal raw d6"),("exp2_8_primal_raw_d8","2.8 primal raw d8"),
 ("exp3a_primal_pinv_s50","3a pinv"),
 ("exp3b_primal_aug_s50","3b aug"),
 ("exp4_mixture","4 mixture d4"),("exp4_8_mixture_d8","4.8 mixture d8"),
 ("exp4b_mixture_innovation","4b mixture innov."),
 ("exp5_sequence","5 sequence"),("exp5b_sequence_exact","5b sequence exact"),
 ("exp6a_alt_L1","6a alt L=1"),("exp6b_alt_L10","6b alt L=10"),
 ("exp6c_alt_L100","6c alt L=100"),("exp6d_alt_L1_nooffset","6d alt L=1 no offset")]
rows = []
for d, lab in ORDER:
    sp, pp = os.path.join(R, d, "summary.json"), os.path.join(R, d, "post", "post_summary.json")
    if not (os.path.exists(sp) and os.path.exists(pp)): continue
    s, p = json.load(open(sp)), json.load(open(pp))
    st = s["settings"]
    acc = {k: round(v["mean"], 3) for k, v in p["acceptance"].items()}
    f = next(iter(p["fields"].values())) if p.get("fields") else {}
    row = dict(dir=d, label=lab, save_every=st.get("save_every"), acc=acc,
               relerr=p.get("relerr_post_M_median"), sec_chain=s.get("sec_per_chain"),
               has_method=("mpsrf_method" in p), mpsrf=p["mpsrf"], fields=p.get("fields"))
    rows.append(row)
json.dump(rows, open("tables/msm_round2/msm_round2_table.json", "w"), indent=1)
for r in rows:
    print(f"\n{r['label']}  [{r['dir']}] save_every={r['save_every']} relerr={r['relerr']:.4f} acc={r['acc']} newMPSRF={r['has_method']}")
    for k, v in r["mpsrf"].items():
        print(f"   {k:34s} dim={v['dim']:3d} start={v['window_start_iteration']:6d} MPSRF={v['mpsrf_final']:.3f} PSRFmax={v['psrf_max_final']:.3f} conv={v['convergence_iteration']}")
    print("   fields:", {k: {a: round(b, 4) for a, b in v.items()} for k, v in r["fields"].items()})

# ---- LaTeX table (report) ----
def g(r, *keys):
    for k in keys:
        if k in r["mpsrf"]: return r["mpsrf"][k]
    return None
ref_sd = rows[0]["fields"]["run"]["logk_sd_mean"]
lines = []
for r in rows:
    fk = "pooled" if "pooled" in r["fields"] else "run"
    m58 = g(r, "theta_range58_common", "theta_phase2_range58_common", "theta_range58", "theta")
    m64 = g(r, "theta", "theta_phase2")
    sw = 50000 if "sequence" in r["dir"] else 0
    cv = lambda m: "--" if (m is None or m["convergence_iteration"] is None) else f"{(m['convergence_iteration']-sw)/1e3:.1f}"
    acc = "/".join(f"{v:.2f}" for v in r["acc"].values())
    f = r["fields"][fk]
    lab = r["label"].replace("&", r"\&")
    lines.append(f"{lab} & {acc} & {r['relerr']:.4f} & {cv(m58)} & {m58['mpsrf_final']:.3f} & "
                 f"{m58['psrf_max_final']:.3f} & {f['logk_sd_mean']:.3f} & {f['logk_sd_mean']/ref_sd:.2f} & {f['logk_mean_error_rms']:.3f} \\\\")
open("tables/msm_round2/results_table.tex", "w").write("\n".join(lines) + "\n")
print("wrote tex table", len(lines))
