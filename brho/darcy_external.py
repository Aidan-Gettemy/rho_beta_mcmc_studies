"""
External Darcy solver driver (the solver is NOT part of this repository).

Solver contract (what any solver must provide to be used here)
    <exe> <case.cfg> -permeability_file K.bin -output_dir DIR [-ksp_rtol r] [-output_* false ...]
    K.bin   little-endian float64, one value per cell, x fastest:
            index = i + j*nx + k*nx*ny
    DIR/pressure_drop.txt   first number = pressure drop in Pa (inlet minus outlet)
    stdout may contain "KSP converged reason: <REASON>"; a DIVERGED_* reason counts as a failed solve.
The case file supplies the mesh, the flux boundary value, the viscosity and the cylinder plug.
Effective permeability: Keff = |v| mu Lz / dP  (v = inlet Darcy flux, mu viscosity, Lz length; the
cross-section cancels because the flux is imposed per unit area).

Point the code at a solver with the SOLVER_DIR/--exe setting (external/README.md).  A faster solver
that honours the contract can be dropped in and validated with `selfcheck` (uniform field,
scaling, reference field against the measured pressure drop).
"""
import os
import shutil
import re
import shlex
import subprocess
import sys

import numpy as np

PSI_TO_PA = 6894.757293168



# ----------------------------------------------------------------------------- cfg
def solver_exe(exe=None, repo=None):
    """Path of the solver executable: the explicit argument, else $SOLVER_EXE, else
    $SOLVER_DIR/darcy_solver, else SOLVER_DIR from external/solver.local (KEY=VALUE lines, not in git)."""
    if exe:
        return os.path.expanduser(exe)
    if os.environ.get("SOLVER_EXE"):
        return os.path.expanduser(os.environ["SOLVER_EXE"])
    if os.environ.get("SOLVER_DIR"):
        return os.path.join(os.path.expanduser(os.environ["SOLVER_DIR"]), "darcy_solver")
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(repo, "external", "solver.local")
    if os.path.exists(local):
        for line in open(local):
            line = line.split("#")[0].strip()
            if line.startswith("SOLVER_DIR="):
                d = os.path.expandvars(os.path.expanduser(line.split("=", 1)[1].strip().strip('"')))
                return os.path.join(d, "darcy_solver")
    raise SystemExit("no solver given: pass --exe, set SOLVER_EXE or SOLVER_DIR, or create "
                     "external/solver.local (see external/README.md)")


def parse_cfg(path):
    """Read the *SECTION blocks of the case file (the '!' starts a comment)."""
    secs, cur = {}, None
    for raw in open(path):
        line = raw.split("!")[0].strip()
        if not line:
            continue
        if line.startswith("*"):
            if line.upper().startswith("*END"):
                break
            cur = line.upper()
            secs[cur] = []
        elif cur:
            secs[cur].append(line)
    return secs


def physics_from_cfg(path):
    s = parse_cfg(path)
    nx, ny, nz = (int(t) for t in s["*GLOBAL_NUMBER_OF_CELLS_IN_EACH_DIRECTION"][0].split())
    Lx, Ly, Lz = (float(t) for t in s["*SCALE_IN_EACH_DIRECTION"][0].split())
    types = [int(t) for t in s["*BOUNDARY_CONDITION_TYPES"][0].split()]
    vals = [float(t) for t in s["*BOUNDARY_CONDITION_VALUES"][0].split()]
    flux = [(f, v) for f, (t, v) in enumerate(zip(types, vals)) if t == 0 and v != 0.0]
    if len(flux) != 1:
        raise ValueError(f"expected exactly one nonzero flux face in the cfg, found {flux}")
    mu = float(s["*MATERIAL_PARAMETERS"][1].split()[0])
    return dict(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, v=abs(flux[0][1]), mu=mu)


# ----------------------------------------------------------------------------- solver
def clean_env():
    """Environment for a solver started from inside an MPI rank (GLAM runs one chain per rank): drop the parent's
    OpenMPI/PMIx/PMI process-manager variables so the solver starts as an independent single-rank (singleton) job
    instead of trying to join the parent's MPI world.  Tuning variables OMPI_MCA_* are kept."""
    drop = ("PMIX_", "PMI_", "HYDRA_", "MPIEXEC_", "OMPI_COMM_", "OMPI_UNIVERSE", "OMPI_NUM_APP", "OMPI_FIRST", "OMPI_APP", "OMPI_MCA_ess",
            "OMPI_MCA_orte", "OMPI_MCA_pmix", "OMPI_MCA_mca_base_env", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID",
            "SLURM_STEP", "SLURM_TASK_PID", "SLURM_GTIDS", "SLURM_NTASKS", "SLURM_NPROCS", "SLURM_STEPID")
    return {k: v for k, v in os.environ.items() if not k.startswith(drop)}


class DarcyRunner:
    def __init__(self, exe, cfg, work, launcher="", rtol=1e-8, timeout=900, extra="", keep_velocity=False,
                 clean_mpi_env=False):
        self.exe, self.cfg, self.work = os.path.abspath(exe), os.path.abspath(cfg), os.path.abspath(work)
        self.launcher, self.rtol, self.timeout, self.extra = launcher, rtol, timeout, extra
        self.env = clean_env() if clean_mpi_env else None
        self.phys = physics_from_cfg(cfg)
        os.makedirs(self.work, exist_ok=True)
        if not keep_velocity:
            # The solver ignores "-output_velocity false" when the cfg says *DUMP_VELOCITY 1 (needed by the
            # tracer / GLAM). Keff-only users (E3 curve) run a copy with the dump switched off.
            txt = open(self.cfg).read()
            txt = re.sub(r"(\*DUMP_VELOCITY[ \t]*\n)\s*\d+", r"\g<1>0", txt, count=1)
            self.cfg = os.path.join(self.work, "DATA_keff_only.cfg")
            open(self.cfg, "w").write(txt)

    def solve(self, K, tag="solve", pressure=False, velocity=False):
        """Run the solver on K (N,) [m^2, solver order].  Returns dict(dp=<pressure drop, Pa>, outdir=<dir>,
        pressure_file, velocity_file) or None after two failed attempts (message on stderr).
        pressure=True also writes solution_pressure (PETSc binary); velocity=True writes solution_velocity.dat for
        the tracer (needs a runner made with keep_velocity=True, since the cfg switch overrides the command line).
        The caller removes `outdir` when done (see `cleanup`)."""
        if velocity and self.cfg.endswith("DATA_keff_only.cfg"):
            raise ValueError("velocity=True needs DarcyRunner(..., keep_velocity=True)")
        kfile = os.path.join(self.work, f"K_{tag}.bin")
        outdir = os.path.join(self.work, f"out_{tag}")
        os.makedirs(outdir, exist_ok=True)
        np.ascontiguousarray(K, dtype="<f8").tofile(kfile)
        cmd = shlex.split(self.launcher) + [self.exe, self.cfg,
               "-permeability_file", kfile, "-output_dir", outdir,
               "-ksp_rtol", repr(self.rtol),
               "-output_pressure", "true" if pressure else "false",
               "-output_velocity", "true" if velocity else "false",
               "-output_permeability", "false", "-output_porosity", "false", "-output_mask", "false",
               "-output_pressure_drop", "true", "-output_pressure_drop_format", "ascii"]
        if pressure:
            cmd += ["-output_pressure_format", "binary"]
        cmd += shlex.split(self.extra)
        why = ""
        for attempt in (1, 2):
            try:
                r = subprocess.run(cmd, cwd=self.work, capture_output=True, text=True, timeout=self.timeout, env=self.env)
                tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-15:])
                dpf = os.path.join(outdir, "pressure_drop.txt")
                if not os.path.exists(dpf):
                    why = f"exit code {r.returncode}; no pressure_drop.txt written. Last solver output:\n{tail}"
                    continue
                m = re.search(r"KSP converged reason:\s*(\w+)", r.stdout)
                if m and m.group(1).startswith("DIVERGED"):
                    why = f"linear solver did not converge ({m.group(1)}). Last solver output:\n{tail}"
                    continue
                dp = float(open(dpf).read().split()[0])
                if r.returncode == 0 and np.isfinite(dp) and dp != 0.0:
                    os.remove(kfile)
                    return dict(dp=dp, outdir=outdir,
                                pressure_file=os.path.join(outdir, "solution_pressure.dat"),
                                velocity_file=os.path.join(outdir, "solution_velocity.dat"))
                why = f"exit code {r.returncode}; pressure drop = {dp}. Last solver output:\n{tail}"
            except subprocess.TimeoutExpired:
                why = f"timeout after {self.timeout} s"
            except (FileNotFoundError, ValueError, IndexError) as e:
                why = f"{type(e).__name__}: {e}"
        sys.stderr.write(f"[solver failed twice] {' '.join(cmd)}\n  reason: {why}\n")
        return None

    @staticmethod
    def cleanup(res):
        if res and os.path.isdir(res["outdir"]):
            shutil.rmtree(res["outdir"], ignore_errors=True)

    def keff(self, K, tag="solve"):
        """K: (N,) array of permeabilities in solver order (x fastest). Returns Keff [m^2] (NaN on failure)."""
        res = self.solve(K, tag)
        if res is None:
            return float("nan")
        p = self.phys
        return p["v"] * p["mu"] * p["Lz"] / abs(res["dp"])




def selfcheck(runner, K0=1.0e-13, ref_field=None, ref_pdrop=None, out=print):
    """Solver acceptance test.  Returns True if the uniform-field and scaling checks pass.
      [1] uniform K0            -> Keff/K0 = 1
      [2] Keff(3.7 K) / Keff(K) -> 3.7         (the scaling lemma behind beta(rho))
      [3] reference field       -> Keff against K* = |v| mu Lz / dP_observed  (dP in psi in the data file)
    Check [3] is printed, not asserted: how close it must be depends on the data."""
    p = runner.phys
    N = p["nx"] * p["ny"] * p["nz"]
    ok = True
    k1 = runner.keff(np.full(N, K0), "uniform")
    out(f"[1] uniform K0 = {K0:.4e}:  Keff = {k1:.6e}   ratio Keff/K0 = {k1 / K0:.6f}   (expect 1)")
    ok &= bool(abs(k1 / K0 - 1) < 1e-3)
    rng = np.random.default_rng(0)
    Kr = K0 * np.exp(0.8 * rng.standard_normal(N))
    ka, kb = runner.keff(Kr, "a"), runner.keff(3.7 * Kr, "b")
    out(f"[2] scaling: Keff(3.7 K)/Keff(K) = {kb / ka:.6f}   (expect 3.7)")
    ok &= bool(abs(kb / ka / 3.7 - 1) < 1e-4)
    if ref_field:
        Kref = np.loadtxt(ref_field)
        kref = runner.keff(Kref, "ref")
        out(f"[3] reference field: geometric mean {np.exp(np.log(Kref).mean()):.4e}, "
            f"arithmetic mean {Kref.mean():.4e}, Keff = {kref:.4e}")
        if ref_pdrop:
            dp_psi = float(open(ref_pdrop).read().split()[0])
            kobs = p["v"] * p["mu"] * p["Lz"] / (dp_psi * PSI_TO_PA)
            out(f"    observed pressure drop {dp_psi} psi -> K* = {kobs:.4e};  Keff(ref)/K* = {kref / kobs:.4f}")
    out("selfcheck: " + ("PASSED" if ok else "FAILED (see lines above)"))
    return ok
