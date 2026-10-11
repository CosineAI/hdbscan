#!/bin/bash
#SBATCH --job-name=hdbscan-cuda-parity
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load cuda/12.6
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUPY_CACHE_DIR="$SLURM_SUBMIT_DIR/results/cupy-cache"
mkdir -p results

srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
  .venv/bin/python scripts/run_cuda_parity.py --output-dir results
