"""V1: GLAM likelihood (hand-computed values, failure handling, real Syn_2mm data if available) + runner cfg patch."""
import glob, os, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from brho.glam_likelihood import GlamData, phi_like, PSI_TO_PA  # noqa: E402
from brho.darcy_external import DarcyRunner  # noqa: E402


def test_glam_like_hand_values():
    ref = np.array([[1.0, 2.0], [2.0, 0.0]])            # ||ref||^2 = 9
    sim = ref + np.array([[0.3, 0.0], [0.0, 0.0]])      # ||err||^2 = 0.09
    d = GlamData(ref, 1000.0)
    phi, (pc, pd) = phi_like(d, sim, 1000.0 * 1.01)     # dp rel err^2 = 1e-4
    assert abs(pc - 0.09 / 9 / (2 * 1e-6)) < 1e-6 * pc    # = 5000
    assert abs(pd - 1e-4 / (2 * 4e-6)) < 1e-9 * pd + 1e-9  # = 12.5
    assert abs(phi - (pc + pd)) < 1e-12
    assert phi_like(d, ref, 1000.0)[0] == 0.0


def test_glam_like_failure_is_inf():
    d = GlamData(np.ones((4, 2)), 1.0)
    bad = np.ones((4, 2)); bad[0, 0] = np.nan
    assert phi_like(d, bad, 1.0)[0] == np.inf
    assert phi_like(d, np.ones((4, 2)), np.nan)[0] == np.inf
    assert phi_like(d, None, 1.0)[0] == np.inf


def test_glam_like_real_data():
    cs = glob.glob("/root/.claude/uploads/*/*Syn_2mm_TC.dat") + [os.environ.get("GLAM_CONC", "")]
    ps = glob.glob("/root/.claude/uploads/*/*Syn_2mm_pDrop.dat") + [os.environ.get("GLAM_DP", "")]
    c = next((x for x in cs if os.path.exists(x)), None); p = next((x for x in ps if os.path.exists(x)), None)
    if c is None or p is None:
        return
    d = GlamData.load(c, p)
    assert d.conc.shape == (64980, 10) and abs(d.dp_pa - 0.480757 * PSI_TO_PA) < 1e-6
    assert phi_like(d, d.conc, d.dp_pa)[0] == 0.0
    # 1 sigma_conc-sized relative perturbation costs 0.5 in Phi_conc by construction
    rng = np.random.default_rng(0)
    e = rng.standard_normal(d.conc.shape); e *= 1e-3 * np.linalg.norm(d.conc) / np.linalg.norm(e)
    assert abs(phi_like(d, d.conc + e, d.dp_pa)[1][0] - 0.5) < 1e-9


def test_runner_keff_only_cfg_turns_velocity_off():
    w = tempfile.mkdtemp()
    r = DarcyRunner("/bin/true", os.path.join(REPO, "external", "DATA.cfg"), w)
    txt = open(r.cfg).read().split("*DUMP_VELOCITY")[1].split("\n")
    assert txt[1].split("!")[0].strip() == "0"
    r2 = DarcyRunner("/bin/true", os.path.join(REPO, "external", "DATA.cfg"), w, keep_velocity=True)
    assert r2.cfg.endswith("DATA.cfg") and "1" == open(r2.cfg).read().split("*DUMP_VELOCITY")[1].split("\n")[1].split("!")[0].strip()
