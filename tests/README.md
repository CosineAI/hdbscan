# Validation record

Initial verification on 10 October 2026, Apple M2 Ultra / macOS:

```text
627 passed, 1114 skipped, 137 warnings in 3.57s
```

The skips were CUDA prerequisite skips: 556 float32 and 558 float64 cases could not execute without CuPy/NVIDIA hardware. They are not passing CUDA results. The warnings came from the pinned sklearn implementation's NumPy scalar-conversion deprecation.

Run from the repository root:

```sh
uv sync --python 3.12 --extra mlx --extra test
uv run pytest -q
```

For NVIDIA, install either `cuda12` or `cuda13` instead of `mlx` and run the same suite. Kernel compilation or execution errors are test failures, not skip reasons. GPU-less environments can run only the CPU-reference/validation tests; a largely skipped suite is not a GPU validation result.

## NVIDIA validation

Verified on 10 October 2026 on Isambard, NVIDIA GH200 120GB (compute capability 9.0), Linux aarch64, Python 3.12.13, CuPy 14.2.0, NumPy 2.3.5 and scikit-learn 1.9.1. The job used the `interactive` partition and reservation, one GPU, and the `cuda/12.6` module. CuPy reported CUDA runtime 12.9 and driver API 13.0; NVRTC was 12.6.

Initial implementation verification: Slurm job `7239795`. Its JSON report and both JUnit files were retrieved locally, and all 15 recorded source/configuration hashes matched that implementation, tests and job scripts.

| Gate | Passed | CUDA skips | Failures |
| --- | ---: | ---: | ---: |
| CUDA float64 | 558 | 0 | 0 |
| Additional CPU oracle checks | 10 | — | 0 |
| CUDA float32 | 556 | 0 | 0 |

The first run exposed an NVRTC compilation error from including host `math.h`. The CUDA header now uses NVRTC's built-in device math and an explicit infinity intrinsic instead. The subsequent hardware run passed both precision gates. Tests of all-identical inputs exercise infinite-density handling; the shared GPU sort tests also pass on Metal after this change.

Reproduce from the repository root after installing the locked CUDA environment:

```sh
uv sync --locked --python 3.12 --extra cuda12 --extra test
mkdir -p results
sbatch --account=YOUR_ACCOUNT --output=results/slurm-%j.out scripts/cuda-parity.sbatch
```

The Slurm script targets Isambard's interactive reservation; adapt the scheduler directives and CUDA module for other clusters. For an already allocated, single-visible-GPU environment:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python scripts/run_cuda_parity.py
```

The runner checks the real device and pinned reference versions, executes float64 before float32, and fails if any selected CUDA case skips or fails. It writes `results/cuda-parity.json` with hardware, versions, source hashes and phase counts, plus JUnit XML for each precision. Full logs and reports are retained locally under the ignored `results/` directory; failed-run evidence is retained separately.

Passing includes the explicitly documented float32 epsilon-resolution counterexample below, not a claim that float32 always reproduces float64 decisions. This initial run validates the synthetic corpus on one NVIDIA architecture. Other GPU architectures and CUDA 13 remain outstanding; targeted sanitizer validation of the optimised implementation is recorded below.

## Private learned-embedding corpus

The subsequent GH200 run exported **256 held-out windows from a private learned-embedding corpus**, each containing 1,024 eight-dimensional metric embeddings, from a trained metric-learning checkpoint. The corpus, checkpoint and raw input data are not distributed in this repository:

- Checkpoint SHA-256: `575e0070dafefcfe16b519f7de63f53d310776e06415ab87eb06874779b0795e`.
- Frozen embedding file SHA-256: `f14d5c9a2dc578d3c0d3c44288923eb81d3af6ad68d051fa5429c9c5ca3bdc55`.
- Selection: seed 42, 128 validation and 128 test windows sampled without replacement among windows with at least two ground-truth classes. The pools contain 21,379 and 25,494 eligible windows respectively.
- Coverage: 87 validation source IDs and 97 test source IDs, with 2–24 ground-truth classes per sampled window.
- Encoder inference: PyTorch 2.9.1+cu128, eval mode, bfloat16 autocast, batches of eight; embeddings saved as float32. Both clustering implementations consume the same saved arrays.
- Clustering: pinned sklearn reference, `min_cluster_size=2`, default `min_samples`, epsilon 0.1.

**CUDA float64 and float32 both matched exact noise masks and partitions on all 256 windows.** This checks every corpus window, not just the prefixes used for timing. It is a sampled private learned-embedding result, not full-dataset coverage, coverage of unrepresented operating modes, or proof for all floating-point embeddings.

Corpus export and initial benchmarks ran in job `7240292`. The corpus, exact window identities, inference/source/data hashes, reports and log are retained locally under ignored `results/`, not bundled with the repository. The corpus hash and all 256 unique window identities were independently checked after local retrieval. See [benchmark results](../benchmarks/RESULTS.md) for CPU-process comparisons and performance conclusions.

After adding whole-corpus gates and startup-isolated CPU-process benchmarks, the local suite passed with **632 passed, 1,120 CUDA prerequisite skips and 137 warnings**. The revised benchmark harness separately passed all **26 selected tests on GH200**, with no selected-test skips, in job `7240336`; three Metal-specific tests were deliberately deselected there. The original CUDA clustering kernels were unchanged during this follow-up.

## Cooperative-backend test coverage

The current tests go beyond final partitions:

- Distance symmetry, the eight-coordinate specialization and the 1,024-point index shortcut are bit-exact against the original device distance kernel.
- Ordered Prim edges check global-index ties and old-source retention across 32–1,024-thread launches. CUDA diagnostic and production launches use explicit matching launch bounds; production scales block width up to 1,024 threads.
- Packed/shared and global workspaces, register-held Prim state, dynamic CUDA shared-memory boundaries and 1,023/1,024/1,025/2,048/4,096-point storage boundaries are exercised.
- Indirect sort order matches pinned NumPy, including forced depth exhaustion; parallel partitions preserve NumPy's rule for continuing the smaller child without rechecking depth.
- Balanced union-find matches the frozen original linkage tree exactly: child orientation, insertion IDs, sizes and weights.
- A separately frozen pre-cache condensation implementation checks parent/point-parent state, stability, selected clusters and labels, including finite, infinite and NaN stability, minimum sizes two/five/int32 maximum and zero/nonzero epsilon.
- Production Metal split-stage outputs match exact sorted endpoint pairs and float32 weight bits. Fresh-buffer hierarchy tests cover both general and minimum-size-two specializations; separate unevaluated-pipeline tests check producer/consumer dependencies.
- Packed CUDA arena alignment covers odd sizes and full capacity. Configured opt-in kernels are retained across compilation-cache eviction, including a subsequent launch needing more than 48 KiB. Forced global fallback uses finite random data and minimum size five, not only infinite-stability duplicate pairs.
- Cooperative sorting is tested on global as well as packed linkage storage. The sorter restores global leaf cardinalities after reusing them as task-queue scratch.
- Metal paired-child preparation is tested enabled and disabled against frozen pre-EOM and final state. Floating state is compared as uint32 bits. Zero-edge-path fixtures assert NaN/Inf coverage; separate starting/ancestor epsilon fixtures assert reference partition changes below, at and above the boundary.

The optimized paths do not relax the partition comparator, add epsilon tolerance, change precision or invoke sklearn in production. Full-size Metal API validation and smaller shader-instrumented tests are separate gates. An attempted fully instrumented 1,024-point packed diagnostic needed **64,028 bytes** of threadgroup memory, exceeding this device's **32,768-byte** limit before execution; disabling the threadgroup-memory instrumentation option did not resolve that limit. The captured error is retained in `results/metal-shader-threadgroup-limit.log`. This is a coverage limit, not a successful full-size shader check or a general memory-safety proof.

See [Apple's validation documentation](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage) for the instrumentation controls. Reproduce the smaller instrumented set with:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 \
uv run pytest -sq tests/test_kernels.py \
  -k 'metal and (split_transport or unevaluated or shared_workspace or parallel_sort or overflow_in_partial or distance_symmetry) and not (513 or 1023 or 1024)'
```

The excluded cases have separate non-instrumented parity and API-validation coverage. CUDA uses all four Compute Sanitizer tools on a targeted subset; neither platform's instrumented subset is an exhaustive safety proof.

### Current source-matched verification

Verified 11 October 2026 after CUDA packing/launch/sort changes and Metal paired-child hierarchy preparation:

| Gate | Result |
| --- | --- |
| Complete local suite, Apple M2 Ultra | **977 passed**, 1,526 skips, 137 warnings |
| CUDA float64, GH200 | **763 passed**, plus 10 CPU oracle checks; no CUDA skips/failures |
| CUDA float32, GH200 | **761 passed**, no CUDA skips/failures |
| Metal API validation of split/state/boundary/real-shape paths | **157 passed**, no skips/failures, including 1,024-point production paths and larger storage boundaries |
| Metal shader-instrumented 33-point hierarchy state cases | **12 passed**, no skips/failures |
| CUDA `memcheck`, `racecheck`, `initcheck`, `synccheck` | **50 tests per tool**, no skips/failures, zero errors; racecheck also reports zero warnings |
| Whole frozen real corpus | **256/256 windows match** in both CUDA precisions and each of three final-source Metal invocations |

The local skips are 1,524 unavailable CUDA cases and two CUDA-only checks selected with a Metal fixture. Warnings remain the pinned sklearn scalar-conversion deprecations and the existing overflow-validation fixture. These skips do not count as CUDA passes; the separate GH200 run supplies that evidence.

CUDA job **`7245930`** completed all steps with exit code zero on Isambard's interactive partition/reservation. Its complete report, JUnit files, sanitizer logs, benchmark samples and source snapshot are in `results/optim-7245930/`. All **22 source/configuration digests** match local files.

Static review of the cooperative sorter found that its global fallback reused leaf sizes as queue scratch without restoring them. The fix restores leaf cardinalities before the existing publication barrier. Exact-tree tests now compare serial-global, parallel-global and parallel-packed storage, and forced-fallback tests include finite random data with minimum size five. Earlier gate `7245916` was cancelled rather than accepted as final evidence; the corrected source passed the entire gate above.

Metal's final-source suite is `results/round2-metal-corrected-tests.{log,xml}`. Its API and small shader gates are `results/round2-metal-api.{log,xml}` and `results/round2-metal-final-shader.{log,xml}`. Hashes/archive are `results/round2-final-source.{sha256,tar.gz}`; repeated parity/timing reports are `results/round2-metal-verified-{1,2,3}.json`. No local GPU benchmarks ran concurrently with local GPU validation.

The new Metal shader gate validates the cooperative hierarchy changes on small windows; **it does not resolve the full-size shader-instrumentation limits documented below**. No exhaustive memory-safety or floating-point-equivalence claim is made. The wheel/source distribution build, packaged attribution and targeted Ruff checks also pass.

### Previous source-matched verification (historical)

Verified 10 October 2026:

| Gate | Result |
| --- | --- |
| Local complete suite, Apple M2 Ultra | **930 passed**, 1,459 skips, 137 warnings |
| CUDA float64, GH200 | **730 passed**, plus 10 CPU oracle checks; no CUDA skips/failures |
| CUDA float32, GH200 | **728 passed**, no CUDA skips/failures |
| Metal API validation, including full-size kernels | **323 passed**, one deliberately selected CUDA-only skip |
| Smaller Metal shader-instrumented set | **95 passed**, no skips/failures; both validation layers explicitly enabled in the log |
| CUDA `memcheck`, `racecheck`, `initcheck`, `synccheck` | **30 tests per tool**, no skips/failures, zero errors; racecheck also reports zero warnings |
| Whole frozen real corpus | **256/256 windows match** in CUDA float64, CUDA float32 and each of three Metal invocations |

The local skips comprise 1,458 CUDA prerequisite skips and the CUDA-only shared-capacity check selected with a Metal fixture. They are not claimed CUDA passes. The 137 warnings are the pinned sklearn scalar-conversion deprecations and the existing overflow-validation fixture.

CUDA job **`7242060`** completed the entire gated workflow. All 22 source/configuration hashes in its retrieved `source.sha256` matched the tested revision. CUDA reports, JUnit XML, four sanitizer logs and the source archive are retained in `results/optim-7242060/`. The prior expanded run `7241703` exposed register-budget launch failures only in 1,024-thread diagnostics; explicit diagnostic launch bounds corrected those without changing that revision's production 256-thread launch or weakening assertions.

Metal's matching Python-source digests/archive are `results/metal-verified-source.{sha256,tar.gz}`. The full suite, API-validation and shader-validation evidence is under `results/metal-verified-{tests,api,shader}.{log,xml}`; repeated timing/corpus reports are `results/metal-verified-{a,b,c}.json`.

Further attempts to instrument the final **full-size production** Metal path were also blocked before completing parity checks:

- General core-distance selection: MLX raised `[Metal::copy] Must use 1024 sized block`.
- The unit-core fast path used for this corpus: the instrumented pipeline permitted at most **576 threads**, below the **1,024-thread** launch.

The exact logs are `results/metal-full-size-shader.log` and `results/metal-unit-core-full-size-shader.log`. These attempts are **not passing shader checks**. They are distinct from the earlier diagnostic threadgroup-memory limit and do not negate the successful uninstrumented full-size parity/API tests, but leave full-size shader instrumentation unverified on this hardware.

The wheel and source distribution build successfully, the wheel includes `THIRD_PARTY_NOTICES.md`, and targeted Ruff checks pass. No GPU timing was taken concurrently with local GPU validation.

## First optimisation milestone (historical)

This milestone introduced cooperative CUDA Prim updates and deterministic tie reductions with 256 threads per window. Shared CUDA/Metal hierarchy code gained path-compressed union-find, ascending-ID EOM adjacency and linear propagation of winning ancestors. No approximation, epsilon tolerance or CPU clustering fallback was introduced.

Validation on 10 October 2026:

- Local Metal/API suite: **649 passed**, 1,154 expected CUDA prerequisite skips, 137 warnings. The warnings comprise sklearn scalar-conversion deprecations and the existing overflow-validation fixture.
- CUDA gates: **578 float64 and 576 float32 cases**, plus ten CPU-oracle checks. No CUDA skips or failures.
- Both precisions: exact noise-mask and partition parity for **256/256 private frozen learned-embedding windows**.
- Exact ordered Prim edges across sizes 2, 255, 256, 257 and 513, including global-index ties and retaining old sources on equal edges.
- Exact EOM stability, parent/selection state and numeric labels against the previous exhaustive device algorithm. Fixtures cover deep trees, zero-distance/infinite-density cases, epsilon variants and an explicitly asserted winning ancestor above a losing intermediate and winning grandchildren.
- NVIDIA Compute Sanitizer `memcheck`, `racecheck`, `initcheck` and `synccheck`: **eight targeted tests per tool**, all passed, zero errors; racecheck also reported zero warnings. These exercise 257/513-point batched graph kernels and normal 1,024-point API calls in both precisions. This is targeted coverage, not sanitization of every test or possible input.

The full performance/sanitizer run is job `7240817`, with artifacts under `results/optim-7240817/`. Earlier run `7240669` correctly caught initialization errors in the *test harness*: it copied unused workspace padding to the host. The helper now transfers only defined regions; production buffers were not zero-filled to hide invalid reads, and the corrected checks pass. The failed log is retained.

The parity runner's final selection is explicitly `cuda-float64 or cpu_oracle or float64_oracle`, then `cuda-float32`. This avoids accidentally selecting Metal EOM tests merely because their names contain “oracle”. Selector-only verification in job `7240976` passed all 588 selected float64/CPU cases and all 576 float32 cases with no skips. Its reports are under `results/parity-final-7240976/`. The 21 recorded source digests match the local code; production kernel and benchmark digests also match job `7240817`, whose only subsequent source change was this runner filter.

See `scripts/optimise-cuda.sbatch` for the repeatable gated workflow and [benchmark results](../benchmarks/RESULTS.md) for measured gains and limitations.

## Test responsibilities

| File | Checks |
| --- | --- |
| `conftest.py` | Exact sklearn 1.9.1 / NumPy 2.3.5 pins, device prerequisites, noise/partition comparator |
| `test_parity.py` | Curated geometry, finite exhaustive domain, epsilon boundaries, seeded cases, 1,024×8 shape, batch invariance, explicit precision limitation |
| `test_float64.py` | True float64-only CUDA fixtures plus CPU confirmation of their expected partitions |
| `test_kernels.py` | Bit-exact distances, Prim edges, NumPy tie sorting/depth exhaustion, exact linkage/EOM/labels, packed/global and capacity boundaries, Metal split-stage transport/lazy dependencies |
| `test_api.py` | Inputs, mutation, device residency, no sklearn fallback/import, native-array host-conversion guard |
| `test_benchmark.py` | Full-corpus/tail-window gating, timing/warmup logic, CPU threads/processes, JSON output and real GPU CLI smoke |

The ordinary parity corpus is quantized to float32 before either implementation sees it; sklearn promotes those same values internally to float64. Strict float64 fixtures bypass that quantization.

## Known precision limitation is not a hidden success

`test_documented_float32_epsilon_resolution_limit` deliberately confirms a counterexample: epsilon `nextafter(float64(0.125), 1)` is above the merge boundary for sklearn, but rounds to `0.125` in float32. The float32 backend matches the rounded problem, not the original full-precision one. This expected divergence is documented explicitly rather than skipped or allowed through an approximate agreement threshold.

All other parity assertions require exact noise masks and identical partitions up to cluster-number permutation. This does not establish equivalence for every possible input, architecture or reference sorting implementation. The private learned-embedding sample described above adds real-data evidence without changing that limitation. Validate parity and performance on your own deployment data and hardware; neither synthetic tests nor these clustering-only measurements establish end-to-end application performance or general speedup guarantees.
