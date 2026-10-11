# HDBSCAN GPU

A correctness-first, **device-resident batched HDBSCAN** implementation for Apple Metal (MLX) and NVIDIA CUDA (CuPy). The Python package is `hdbscan_gpu`.

The CPU oracle is **`sklearn.cluster.HDBSCAN` from scikit-learn 1.9.1 with NumPy 2.3.5**, using dense Euclidean input and `algorithm="auto"`. This is not the separate scikit-learn-contrib `hdbscan` package.

**Status:** Both backends have cooperative graph kernels, exact intermediate-state regression tests and measurements on frozen learned embeddings. See the [benchmark results](benchmarks/RESULTS.md) for CPU-relative speedups by device, precision and batch size, and the [validation record](tests/README.md) for parity and GPU instrumentation coverage. These are clustering-only results, not a universal speedup or proof of floating-point equivalence. Validate end-to-end performance and parity on your own workload.

## Scope

- Input: one dense window `[N, D]` or independent windows `[B, N, D]`.
- Output: device-resident `int32` labels `[N]` or `[B, N]`; `-1` means noise.
- Euclidean distance, self-inclusive `min_samples`, EOM selection, configurable epsilon and minimum cluster size.
- Fixed `alpha=1`, `allow_single_cluster=False`, and no maximum-cluster-size constraint.
- Metal uses float32. CUDA offers float32 and float64.
- At most 4,096 samples per window, with a 512 MiB limit on the two dense distance buffers combined; this is not a bound on all runtime/device memory.

Windows never share graph edges or clusters. No sparse inputs, arbitrary metrics, probabilities, out-of-sample prediction, gradients, or sklearn estimator wrapper are implemented. Unsupported options fail explicitly. Scikit-learn is a testing/benchmark dependency only: there is **no CPU clustering fallback**.

## Install

Use Python 3.12 and `uv`. Clone the repository first:

```sh
git clone https://github.com/CosineAI/hdbscan.git
cd hdbscan
```

On an Apple-silicon Mac with Metal support:

```sh
uv sync --python 3.12 --extra mlx --extra test
uv run pytest -q
```

On a supported Linux NVIDIA/CUDA machine, select exactly one CuPy distribution:

```sh
uv sync --python 3.12 --extra cuda12 --extra test
# Or use --extra cuda13 for a compatible CUDA 13 environment.
uv run pytest -q
```

The CUDA and Metal tests share a contract. Tests skip only when the relevant optional dependency or GPU prerequisite is absent. Kernel failures and unexpected parity differences fail the suite. Inspect skipped counts before claiming a backend works.

## Use

```python
import mlx.core as mx
from hdbscan_gpu import hdbscan

windows = mx.random.normal((8, 1024, 8))
labels = hdbscan(
    windows,
    backend="mlx",
    min_cluster_size=2,
    min_samples=2,
    cluster_selection_epsilon=0.1,
)
mx.eval(labels)  # MLX is lazy; evaluate before timing or consuming results.
```

For CUDA, pass a CuPy array and `backend="cuda", precision="float64"` to use double precision. Torch CUDA tensors can be wrapped through `cupy.from_dlpack(tensor.detach())` before calling this API; there is no need for `.cpu().numpy()`. Torch tensors are not accepted directly.

NumPy arrays and nested numeric lists are host inputs and are explicitly transferred. Native MLX/CuPy arrays stay on the selected device. Outputs stay there too. Input/distance validation synchronizes scalar status with the host: twice on CUDA, once on Metal, where the checks are fused with distance computation. The implementation is device-resident, **not completely asynchronous**. No clustering decisions are computed on the CPU.

## Implementation

1. A GPU kernel computes coordinate-wise Euclidean distances for one triangle and mirrors it exactly. Coordinate accumulation order and precise square roots are preserved.
2. GPU partitioning finds self-inclusive core distances. For `min_samples <= 2`, both core radii are bounded by every off-diagonal edge, so mutual reachability is exactly Euclidean distance; partitioning and core loads can be omitted without approximation.
3. Cooperative Prim keeps its iterations ordered but parallelises vertex updates and candidate reduction. Comparisons are strict, ties resolve by global vertex index, and equal edges retain their old source.
4. Indirect introsort preserves the pinned NumPy tie order. Balanced leaf-representative union-find emits the same linkage IDs, child orientation, weights and subtree sizes as the original algorithm.
5. Breadth-first condensation preserves stability accumulation order. Ordered EOM adjacency, ancestor suppression, epsilon selection and label propagation avoid repeated whole-tree scans.

**CUDA:** one block per window. Up to 1,024 points, the block width is the next power of two (at least 32), each thread retains one vertex's Prim state in registers, and independent introsort partitions execute cooperatively. Larger windows retain a 256-thread general path. The packed graph/tree arena uses 31 bytes per point in float32 or 43 in float64, plus kernel-static shared storage; it uses the device's reported opt-in shared-memory capacity or falls back to device-global storage. Configured kernel objects survive compilation-cache eviction. Minimum-size-two condensation omits general subtree traversal.

**Metal, up to 1,024 points:** packed threadgroup workspace, register-held Prim state, native SIMD reductions and parallel execution of independent sorting partitions. Two dependent GPU kernels separate Prim/sort from tree processing: sorted int16 endpoint pairs and unchanged float32 weights feed a 32-thread hierarchy kernel with cooperative final label writes. When the minimum cluster size is two, every pruned child must be a leaf, so a separately tested specialization omits general subtree traversal. Larger supported windows use the cooperative device-global workspace path.

The Metal hierarchy cooperatively prepares condensed child pairs and epsilon predicates. Every split produces exactly two consecutive child IDs; EOM still adds their stabilities in the original order. Strict starting `< epsilon` and ancestor `> epsilon` comparisons retain the original reciprocal round trip, including equality, infinities and NaNs.

Both backends share the graph/tree algorithm source through backend-specific headers. Packed storage changes integer representation and scratch lifetimes, not arithmetic precision. No approximate neighbours, jitter, fast-math substitution, epsilon tolerance or CPU clustering fallback is used.

`_kernels.py` derives the tree algorithms from sklearn and indirect introsort from NumPy. Source pins, licensing notices and attribution are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## What parity means

The suite checks **exact noise masks and exact partitions modulo cluster-number renaming**, not approximate agreement scores. It includes:

- Separated/overlapping and varying-density clusters, isolated points, duplicate-heavy and all-identical inputs.
- Symmetric/tied geometry, tiny clusters and all-noise outcomes.
- Exact/below/above epsilon boundaries.
- All 91 multisets of four or five points from a four-vertex grid, under four parameter combinations: 364 finite exhaustive cases per backend.
- Seeded randomized sweeps, selected input permutations, 1,024×8 fixtures and batch/position invariance.
- Device residency, mutation protection, strided inputs, validation errors and blocking sklearn imports in the production path.
- GPU intermediate edge-sort comparisons with the pinned NumPy oracle.
- Exact ordered Prim-edge tests across thread-block boundaries, including equal-edge source retention.
- Bit-exact distance symmetry/specialisation and exact linkage-tree comparisons against the previous device implementation.
- Shared/global workspace, register-Prim, capacity-boundary and forced introsort-depth comparisons.
- CUDA opt-in configuration after cache eviction and forced global fallback with nontrivial minimum sizes; exact linkage checks catch sort scratch storage corrupting leaf cardinalities.
- Exact Metal split-stage endpoint/weight transport, fresh-buffer hierarchy processing and unevaluated producer/consumer pipelines.
- Exact EOM stability, selected-cluster and numeric-label comparisons with the previous device-side algorithm, including suppression through a losing intermediate cluster.
- Bitwise Metal condensation and final-state checks, asserted NaN/Inf coverage, and epsilon-boundary fixtures whose reference partitions change at the strict comparisons.
- CUDA-only unquantized float64 fixtures, verified on NVIDIA GH200 hardware.

This is not a proof over every floating-point input. In particular:

**Float32 can disagree with sklearn's float64 arithmetic.** There is an explicit regression demonstrating that an epsilon one float64 step above `0.125` merges clusters in sklearn but rounds to the unmerged boundary in float32. Passing that test documents the limitation; it does not count as exact parity for that case. Metal float64 is rejected rather than silently moved to CPU.

**Tie ordering targets NumPy's scalar indirect quicksort.** Architecture-specific NumPy SIMD sorting can differ. CUDA parity must be checked against the actual pinned reference environment on the target machine, not inferred from Metal results.

## Benchmark

```sh
uv run python -m benchmarks.benchmark \
  --backend mlx --samples 1024 --dimensions 8 \
  --batch-sizes 1 8 32 --warmup 5 --repeats 30 \
  --output benchmark-results/metal.json
```

Use `--input embeddings.npy` for frozen real embeddings instead of the synthetic corpus, `--cpu-workers` to compare parallel CPU window processing, and `--backend cuda --precision float64` for NVIDIA. Multi-worker CPU runs use threads by default; add `--cpu-executor processes` for persistent independent processes started with `spawn`. Process timings include per-call serialization/dispatch but exclude pool startup after warmup.

For a supplied corpus, the harness checks **every window**, including the last partial batch, before recording any timings. A mismatch writes its window index and diagnostics, exits nonzero and prevents benchmarking. Timings use the requested prefixes of the same corpus. The report records the input SHA-256, device-resident and host-to-host timings, first-call time, warmup, versions, shape, CPU affinity and worker configuration. Timings cover clustering only, not encoder inference. See [the measurements](benchmarks/RESULTS.md).

For the full optimisation gate on Isambard, set `HDBSCAN_CORPUS` to the frozen `.npy` file and submit `scripts/optimise-cuda.sbatch`. It runs both CUDA test suites, stage profiling, whole-corpus parity, five warmups and 30 timed repetitions against eight CPU processes, then targeted `memcheck`, `racecheck`, `initcheck` and `synccheck` runs. `CUDA_SANITIZER` can select the Compute Sanitizer executable when it is not beside the CUDA toolkit. Reports and exact source snapshots go under `results/optim-<job-id>/`.

For Metal stage investigation, use `python -m scripts.profile_mlx --input embeddings.npy --output profile.json`. Its production split-kernel measurements and fused diagnostic alternatives are identified separately; timings of differently compiled prefix kernels must not be subtracted and presented as exact production-stage costs.

Add `--cooperative-variants` to compare the Metal hierarchy preparation on/off, or `--variants` to `scripts.profile_cuda` to compare exact CUDA launch/sort specializations. These are internal diagnostic ablations, not public clustering options. CUDA's `graph_and_hierarchy` and `full_api` use the production dispatch; `prim_only` and `graph_global_workspace` remain explicitly labelled legacy reference variants.

### Benchmarking your embeddings

Save independent, fixed-size windows as a dense numeric `.npy` array with shape `[B, N, D]`. Record the model, preprocessing, precision and input hashes separately, and compare every backend on the **same saved corpus**, not newly generated embeddings. The private learned-embedding corpus used in the recorded benchmarks is not distributed; synthetic tests and benchmarks run without it.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python -m benchmarks.benchmark \
  --backend cuda --precision float64 --input /path/to/corpus/embeddings.npy \
  --batch-sizes 1 8 32 --cpu-workers 8 --cpu-executor processes \
  --warmup 5 --repeats 30 --output benchmark-results/cuda-processes.json
```

## Licence

Released under the [BSD 3-Clause License](LICENSE). The upstream-derived portions retain the copyrights and notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
