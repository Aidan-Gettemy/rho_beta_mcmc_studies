#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive                   # the whole node: the benchmark measures memory-bandwidth sharing between 64 processes
#SBATCH --mem=0
#SBATCH --partition=dev               # 2 h limit, short queue; the node type is the same as normal
#SBATCH --time=00:30:00
#SBATCH -J brho-benchproj
#SBATCH -o logs/brho-benchproj-%j.out
#
# Projection benchmark on a COMPUTE node (the login node is shared and capped, so it cannot answer this):
#   sbatch cluster/slurm/bench_projection.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
echo "node $(hostname)  nproc=$(nproc)  allowed cpus: $(taskset -cp $$ | sed 's/.*: //')"
lscpu | grep -E "Model name|Socket|Thread|NUMA node\(s\)|^CPU\(s\)"
free -g | head -2
F="$EXT_MATRICES/phi_full_f32.npy"
echo "== shared memory map (what the first benchmark did)"
python experiments/bench_projection.py --procs 1 8 16 32 64 --phi "$F"
echo "== private copy per process (= --phi-mode ram, the layout of the CPU GLAM run)"
python experiments/bench_projection.py --procs 1 8 16 32 64 --phi "$F" --private
