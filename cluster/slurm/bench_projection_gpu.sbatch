#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --partition=h200
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH -J brho-benchgpu
#SBATCH -o logs/brho-benchgpu-%j.out
#
# GPU projection benchmark (P processes share the GPUs round-robin):  sbatch cluster/slurm/bench_projection_gpu.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# the GPU tracer and CuPy live in the darcyMCMC environment (cupy 13.4.1, petsc4py 3.21.4); override with BRHO_CONDA_ENV=...
export BRHO_CONDA_ENV="${BRHO_CONDA_ENV-/home/arg210006/.conda/envs/darcyMCMC}"
source cluster/env.sh
set -a; source external/solver.local; set +a
set +u
[ -n "${SOLVER_ENV:-}" ] && source "$SOLVER_ENV"      # whatever environment provides CuPy for the tracer
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
echo "node $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import cupy; print('cupy', cupy.__version__)"
python experiments/bench_projection_gpu.py --procs 1 2 8 16 32 64 --phi "$EXT_MATRICES/phi_full_f32.npy"
