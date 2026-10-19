# External solver, fields and data

Everything here is **about** a Darcy solver, the multiscale fields and the measured data that live
outside this repository. Only pointers, our own case file and a few small scripts are committed; the
solver, the matrices and the data are not (`.gitignore`).

| in git | not in git (see `.gitignore`) |
|---|---|
| `README.md`, `solver.local.example`, `DATA.cfg` (our case file), `fetch_matrices.sh` | `solver.local` (your paths), `solver/`, `matrices/`, `data/`, `fields/` |

## 1. The solver (kept out of the repository on purpose)

The solver (`darcy_solver`, C + PETSc, TPFA finite volumes, CG + GAMG) belongs to its author and carries
its own license; the repository only needs to know **where it is**:

```
cp external/solver.local.example external/solver.local     # then edit SOLVER_DIR (and SOLVER_ENV)
```

`brho/darcy_external.py` also accepts `--exe PATH`, `$SOLVER_EXE` or `$SOLVER_DIR`.

**Contract** (what any solver must provide, so a faster replacement can be dropped in):

```
<exe> case.cfg -permeability_file K.bin -output_dir DIR [-ksp_rtol r] [-output_pressure false ...]
```

- `K.bin`: little-endian float64, one value per cell, x fastest (`index = i + j*nx + k*nx*ny`).
- `DIR/pressure_drop.txt`: first number = pressure drop in Pa (inlet minus outlet).
- stdout may contain `KSP converged reason: <REASON>`; `DIVERGED_*` counts as a failed solve.
- The case file gives mesh, flux boundary value, viscosity and the plug mask; the driver reads
  `nx ny nz`, the length scale `Lz`, the one nonzero flux `v` and the viscosity `mu` from it and sets
  `Keff = |v| mu Lz / dP`.

**Acceptance test**, for any solver, on the machine that will run the study:

```
python experiments/run_external_curve.py --selfcheck --cfg external/DATA.cfg \
    --ref-field external/data/Kref.dat --ref-pdrop external/data/Syn_2mm_pDrop.dat
```

It checks that a uniform field gives `Keff/K0 = 1`, that `Keff(3.7 K)/Keff(K) = 3.7`, and prints
`Keff` of the reference field against `K* = |v| mu Lz / dP_obs` (the data file is in psi). The Juno run
of 2026-10-18 gave 1.000000, 3.700000 and `Keff(ref)/K* = 1.0000` (see `LOG.md`).

### Building the solver on Juno (UT Dallas)

PETSc is a module there. The solver's own docs assume PETSc 3.23; Juno's is 3.18.1, which needs one
compile flag and an empty `PETSC_ARCH` (a prefix install):

```
cd $SOLVER_DIR
cat > ev_arch-openmpi <<'EOT'
module load gnu14 openmpi5 petsc/3.18.1
export PETSC_ARCH=
export C_INCLUDE_PATH=$PETSC_DIR/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$PETSC_DIR/lib:$LD_LIBRARY_PATH
EOT
source ev_arch-openmpi
make CPPFLAGS="-DPETSC_SUCCESS=0"       # PETSC_SUCCESS only exists from PETSc 3.19 on (it is 0)
```

Every solve is a single MPI rank. Set `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader` (the job script does)
or OpenMPI tries InfiniBand and prints "max locked memory" errors on the login node.

## 2. The matrices (multiscale fields)

`fetch_matrices.sh` copies them from the cluster that holds them (`--list`, `--size`, `--dry-run` first;
`REMOTE_USER`, `REMOTE_HOST` and `REMOTE_BASE` have no defaults). For the 38x38x45 problem with 500 local
modes (`<tag>` = `38x38x45_38x38x45_l2.0x2.0x2.0_m500_IND`) the study needs, in `external/matrices/`:

- `whitening_operator_SP1.npz`: `eigvec`, `invsqrt` of size **4699** (the global basis). The SP2 file has
  5313 modes and does **not** match this PHI; the loader raises a clear error.
- `phi_matrix_full_<tag>.mat` (`PHI`, 64980 cells x 4699 modes, about 1.1 GB; transfer it with
  `rsync -avP --partial --append-verify` in a retry loop if the link drops).
- `KLE_matrix_<tag>_<i>.mat` (`T`) and `SG_elem_<tag>_<i>.mat` (`elem`) for the 12 subdomains, i = 1..12.

The other files the script copies (`CondMat*`, `pdom*`, `pnode*`, `SG_ext*`, `SG_elext*`) are for other
parts of the group's pipeline and are not used here. Reading needs `h5py` (v7.3 `.mat`) or `scipy`.
`brho/multiscale_mat.py` documents the recipe (assemble, whiten, project).

## 3. Data (`external/data/`)

- `Kref.dat`: the reference permeability field, one value per cell (a plain text file; check that
  `head -c 100` shows numbers, not a Git LFS pointer starting with `version`).
- `Syn_2mm_pDrop.dat`: the observed pressure drop in psi; `Kexp = |v| mu Lz / (dP_psi * 6894.757)`.
- `Syn_2mm_TC.dat`: tracer concentration data, for the full sampler (not used by the curve).

## 4. Running

`cluster/slurm/node_e3_external.sbatch` (settings in `cluster/env.sh`, block "external-solver curve"):
pilot first (`E3EXT_ARGS="--n-real 64 --n-rho 6 --exp-tag pilot" ALLOW_FEW_RHO=1 sbatch ...`), then the
study. Results go to `$BRHO_ROOT/results/e3_external/` (`curve_external.npz` has the layout of
`curve_kle.npz`, so `brho.curve.BetaCurve.load` reads it), fits to `tables/e3_external/`, figures to
`figures/e3_external/`.
