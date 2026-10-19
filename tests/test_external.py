"""External-solver machinery: driver, .mat field source, fits, runner (mock solver, no real PDE solves)."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from brho import curvefit  # noqa: E402
from brho.curve import BetaCurve  # noqa: E402
from brho.darcy_external import DarcyRunner, parse_cfg, physics_from_cfg, selfcheck  # noqa: E402
from brho.multiscale_mat import MultiscaleField, load_multiscale  # noqa: E402

MOCK = os.path.join(HERE, "mock_darcy_solver.py")
CFG = """*TITLE
test
*GLOBAL_NUMBER_OF_CELLS_IN_EACH_DIRECTION
6 6 8
*SCALE_IN_EACH_DIRECTION
0.06 0.06 0.08 ! m
*BOUNDARY_CONDITION_TYPES
0 0 0 0 0 1 ! l,r,d,u,b,t
*BOUNDARY_CONDITION_VALUES
0 0 0 0 -7.6313E-06 6894.76
*MATERIAL_PARAMETERS
4
0.001 ! viscosity water
0.00062
1000.0
*END
"""


def _write_cfg(d):
    path = os.path.join(d, "case.cfg")
    with open(path, "w") as f:
        f.write(CFG)
    return path


def _mock_exe(d):
    """Executable wrapper so the mock runs through the same path as a real solver binary."""
    path = os.path.join(d, "mock_solver")
    with open(path, "w") as f:
        f.write(f"#!/bin/sh\nexec {sys.executable} {MOCK} \"$@\"\n")
    os.chmod(path, 0o755)
    return path


def test_cfg_physics():
    with tempfile.TemporaryDirectory() as d:
        p = physics_from_cfg(_write_cfg(d))
        assert (p["nx"], p["ny"], p["nz"]) == (6, 6, 8)
        assert abs(p["Lz"] - 0.08) < 1e-15 and abs(p["v"] - 7.6313e-6) < 1e-15 and abs(p["mu"] - 1e-3) < 1e-15
        assert "*MATERIAL_PARAMETERS" in parse_cfg(os.path.join(d, "case.cfg"))


def test_driver_selfcheck_with_mock():
    with tempfile.TemporaryDirectory() as d:
        runner = DarcyRunner(_mock_exe(d), _write_cfg(d), os.path.join(d, "work"))
        lines = []
        assert selfcheck(runner, 1e-13, out=lines.append), "\n".join(lines)
        # a solver that fails must give NaN, not a number
        bad = DarcyRunner("/bin/false", _write_cfg(d), os.path.join(d, "work2"))
        assert np.isnan(bad.keff(np.full(6 * 6 * 8, 1e-13)))


def test_driver_diverged_is_failure():
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "div")
        with open(exe, "w") as f:
            f.write(f"#!/bin/sh\n{sys.executable} {MOCK} \"$@\" >/dev/null\n"
                    "echo 'KSP converged reason: DIVERGED_ITS, iterations: 10000'\n")
        os.chmod(exe, 0o755)
        r = DarcyRunner(exe, _write_cfg(d), os.path.join(d, "w"))
        assert np.isnan(r.keff(np.full(6 * 6 * 8, 1e-13)))


def _synthetic_multiscale(rng, N=600, M=40, n_sub=3):
    PHI = np.linalg.qr(rng.standard_normal((N, M)))[0]
    U = np.linalg.qr(rng.standard_normal((M, M)))[0]
    s = np.linspace(0.2, 5.0, M)
    cells = np.array_split(rng.permutation(N), n_sub)
    S_half = U @ np.diag(np.sqrt(s)) @ U.T
    return PHI, U, s, cells, [PHI[c] @ S_half for c in cells]


def test_multiscale_algebra_and_mismatch():
    rng = np.random.default_rng(1)
    PHI, U, s, cells, T_list = _synthetic_multiscale(rng)
    M = PHI.shape[1]
    f = MultiscaleField(T_list, cells, PHI, U, 1.0 / np.sqrt(s), dtype=np.float64)
    a = f.field(f.sample_theta(np.random.default_rng(5)))
    b = f.field(f.sample_theta(np.random.default_rng(5)))
    assert np.allclose(a, b)
    try:
        MultiscaleField(T_list, cells, PHI, np.eye(M + 3), np.ones(M + 3), dtype=np.float64)
        raise AssertionError("whitening/PHI mismatch not detected")
    except ValueError:
        pass
    try:
        MultiscaleField(T_list, [c[:-1] for c in cells], PHI, U, 1.0 / np.sqrt(s), dtype=np.float64)
        raise AssertionError("incomplete partition not detected")
    except ValueError:
        pass
    # Y = PHI z, z ~ N(0, Sigma)  =>  PHI^T YP = W z has covariance I
    Sigma = U @ np.diag(s) @ U.T
    Ys = rng.multivariate_normal(np.zeros(M), Sigma, size=4000) @ PHI.T
    N = PHI.shape[0]
    f1 = MultiscaleField([np.eye(N)], [np.arange(N)], PHI, U, 1.0 / np.sqrt(s), dtype=np.float64)
    coef = np.array([PHI.T @ f1.whiten_project(y) for y in Ys[:2500]])
    assert np.max(np.abs(np.cov(coef.T) - np.eye(M))) < 0.12


def _write_mat_set(d, tag, rng, N=600, M=40, n_sub=3):
    """Synthetic files in the cluster's naming (v5 .mat, read through scipy)."""
    from scipy.io import savemat
    PHI, U, s, cells, T_list = _synthetic_multiscale(rng, N, M, n_sub)
    for i, (T, c) in enumerate(zip(T_list, cells), start=1):
        savemat(os.path.join(d, f"KLE_matrix_{tag}_{i}.mat"), {"T": T})            # MATLAB: cells x modes
        savemat(os.path.join(d, f"SG_elem_{tag}_{i}.mat"), {"elem": (c + 1)[None, :]})   # 1-based row vector
    savemat(os.path.join(d, f"phi_matrix_full_{tag}.mat"), {"PHI": PHI})
    np.savez(os.path.join(d, "whiten.npz"), eigvec=U, invsqrt=1.0 / np.sqrt(s))
    return PHI, U, s, cells, T_list


def test_load_multiscale_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(2)
        PHI, U, s, cells, T_list = _write_mat_set(d, "tg", rng)
        f = load_multiscale(d, "tg", os.path.join(d, "whiten.npz"), n_sub=3, dtype=np.float64)
        ref = MultiscaleField(T_list, cells, PHI, U, 1.0 / np.sqrt(s), dtype=np.float64)
        th = [np.random.default_rng(7 + i).standard_normal(T.shape[1]) for i, T in enumerate(T_list)]
        assert np.allclose(f.field(th), ref.field(th), atol=1e-12)


def test_curvefit_recovers_and_beta_curve_compatible():
    rhos, keff, K0, h_true = curvefit.synthetic_data(seed=0, n_rho=40, n=1024)
    out, extra = curvefit.fit_all(rhos, keff, K0, Kexp=1e-13)
    m4 = next(m for m in out["models"] if m["degree"] == 4)
    assert abs(m4["coef"][1] - 0.15) < 0.01 and abs(m4["coef"][2] + 0.004) < 0.002
    assert abs(m4["coef"][0]) < 0.02
    assert out["loo_rms_over_se"]["poly4"] < 1.0 and out["loo_rms_over_se"]["poly2"] > 5
    # PCHIP passes through the anchor and the data, NaN outside
    p = extra["pchip"]
    assert abs(float(p(0.0))) < 1e-15 and np.allclose(p(rhos), extra["h"]) and np.isnan(p(rhos.max() + 1))
    # same data as BetaCurve input (keff1 = keff / K0)
    bc = BetaCurve(rhos, keff / K0)
    assert np.isfinite(bc.beta_bar(2.0, 1e-13)) and np.isnan(bc.beta_bar(6.0, 1e-13))
    try:
        bad = keff.copy()
        bad[3, 5] = np.nan
        curvefit.fit_all(rhos, bad, K0, Kexp=1e-13)
        raise AssertionError("NaN keff not rejected")
    except ValueError:
        pass


def test_runner_end_to_end_with_mock():
    """Whole pipeline on a tiny problem: .mat fields -> parallel solves -> npz -> fits."""
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(3)
        N = 6 * 6 * 8
        _write_mat_set(d, "tg", rng, N=N, M=12, n_sub=3)
        cfg, exe = _write_cfg(d), _mock_exe(d)
        env = dict(os.environ, BRHO_ROOT=os.path.join(d, "root"), ALLOW_FEW_RHO="1")
        cmd = [sys.executable, os.path.join(REPO, "experiments", "run_external_curve.py"),
               "--exe", exe, "--cfg", cfg, "--matrix-dir", d, "--tag", "tg",
               "--whitening", os.path.join(d, "whiten.npz"), "--n-sub", "3", "--workers", "2",
               "--n-real", "6", "--n-rho", "8", "--n-save", "2", "--Kexp", "1e-13", "--degrees", "2", "4"]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        res = os.path.join(d, "root", "results", "e3_external")
        z = np.load(os.path.join(res, "curve_external.npz"))
        assert z["keff"].shape == (8, 6) and np.all(np.isfinite(z["keff"]))
        assert np.allclose(z["keff1"], z["keff"] / float(z["K0"]))
        bc = BetaCurve.load(os.path.join(res, "curve_external.npz"))
        assert np.isfinite(bc.beta_bar(1.0, 1e-13))
        assert os.path.exists(os.path.join(d, "root", "tables", "e3_external", "fits.json"))
        assert os.path.exists(os.path.join(d, "root", "figures", "e3_external", "fit_curves.png"))
        assert len(os.listdir(os.path.join(res, "fields"))) == 2
        # resume: a second run does no new solves and gives identical numbers
        r2 = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert r2.returncode == 0
        assert np.array_equal(np.load(os.path.join(res, "curve_external.npz"))["keff"], z["keff"])
        # a failed solve (NaN) must still be saved, and the fits must be skipped, not crash
        part = os.path.join(res, "keff_parts", "keff_00002.npy")
        bad = np.load(part); bad[3] = np.nan; np.save(part, bad)
        r3 = subprocess.run(cmd + ["--fit"], env=env, capture_output=True, text=True)
        assert r3.returncode == 0, r3.stdout + r3.stderr
        z3 = np.load(os.path.join(res, "curve_external.npz"))
        assert np.isnan(z3["keff"]).sum() == 1 and "fits skipped" in r3.stdout
