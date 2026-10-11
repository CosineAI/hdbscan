#!/bin/bash
#SBATCH --job-name=hdbscan-optimisation
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
: "${HDBSCAN_CORPUS:?Set HDBSCAN_CORPUS to the frozen embedding NPY file}"
module load cuda/12.6
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUPY_CACHE_DIR="$SLURM_SUBMIT_DIR/results/cupy-cache"
output="results/optim-$SLURM_JOB_ID"
mkdir -p "$output"
sha256sum hdbscan_gpu/*.py benchmarks/*.py tests/*.py scripts/*.py scripts/*.sbatch \
  pyproject.toml uv.lock > "$output/source.sha256"
tar --exclude=__pycache__ -czf "$output/source-snapshot.tar.gz" \
  hdbscan_gpu benchmarks tests scripts pyproject.toml uv.lock

run() {
  srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores "$@"
}

run .venv/bin/python scripts/run_cuda_parity.py --output-dir "$output"
for precision in float64 float32; do
  run .venv/bin/python -m scripts.profile_cuda \
    --input "$HDBSCAN_CORPUS" --precision "$precision" --output "$output/profile-$precision.json"
  run .venv/bin/python -m benchmarks.benchmark \
    --backend cuda --precision "$precision" --input "$HDBSCAN_CORPUS" \
    --batch-sizes 1 8 32 --min-cluster-size 2 --epsilon 0.1 \
    --cpu-workers 8 --cpu-executor processes --warmup 5 --repeats 30 \
    --output "$output/benchmark-$precision.json"
done

sanitizer="${CUDA_SANITIZER:-$(dirname "$(command -v nvcc)")/../compute-sanitizer/compute-sanitizer}"
test -x "$sanitizer"
for tool in memcheck racecheck initcheck synccheck; do
  run "$sanitizer" --tool "$tool" --error-exitcode 99 \
    --log-file "$output/sanitizer-$tool.log" \
    .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_kernels.py tests/test_parity.py \
    -k "(cuda-float32 or cuda-float64) and (unit_core or runtime_shared or cached_shared_kernel or overflow_in_partial or (parallel_sort and 129) or (minimum_two_preserves and 33) or (shared_workspace and 0.1 and (32-0 or 1024-1)) or (storage_boundary and 4096) or (eom_adjacency and 5-0.1-0.0-32) or real_shape)" \
    --junitxml="$output/sanitizer-$tool.xml"
done
printf 'Parity, benchmarks and sanitizer checks completed\n' > "$output/complete"
