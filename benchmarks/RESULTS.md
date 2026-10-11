# Benchmark results

The learned-embedding measurements below use a private frozen corpus that is not distributed in this repository. Model checkpoints, raw inputs and the local artifacts referenced under ignored `results/` are not bundled either. These results are workload- and hardware-specific clustering timings, not end-to-end application measurements or general speedup guarantees. Validate parity and performance on your own data, hardware and deployment.

## Current: packed CUDA launch/sort and cooperative Metal hierarchy

Measured 11 October 2026 on the same private frozen 256-window learned-embedding corpus, with unchanged precision, parameters and **eight persistent CPU processes**, one numerical-library thread per process. Five warmups and 30 completed samples are used per batch size. Every invocation checks exact noise masks and partitions on all 256 windows before timing. These are clustering-only measurements; encoder inference and CPU-pool startup are excluded, while normal GPU validation and CPU dispatch/serialization remain included.

### NVIDIA GH200 120GB

Final source: interactive job **`7245930`**, eight allocated CPU cores. All 22 recorded source/configuration digests match the local code.

| Precision | Windows/batch | 8 CPU processes | GPU resident | GPU host-to-host | Resident speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| float64 | 1 | 17.541 ms | 2.718 ms | 2.758 ms | 6.45× |
| float64 | 8 | 18.637 ms | 3.127 ms | 3.204 ms | 5.96× |
| float64 | 32 | 74.914 ms | 3.697 ms | 3.883 ms | **20.27×** |
| float32 | 1 | 17.435 ms | 2.625 ms | 2.659 ms | 6.64× |
| float32 | 8 | 18.720 ms | 2.871 ms | 2.932 ms | 6.52× |
| float32 | 32 | 74.251 ms | 3.464 ms | 3.576 ms | **21.43×** |

Compared with the previous batch-32 GPU medians of 4.988/4.584 ms, this is **25.9% less time in float64 and 24.4% less in float32** (1.35×/1.32× additional GPU speedup). Smaller batches improve too, but do not reach 10× CPU-relative speedup.

Packing the workspace alone was approximately flat. The substantial additional gain came from register-held Prim state with a 1,024-thread block at this window size, followed by cooperative execution of independent exact-introsort partitions. The minimum-size-two shortcut adds a smaller saving. Source-matched reports, all timing samples/p95s and validation artifacts are in `results/optim-7245930/`.

### Apple M2 Ultra / MLX 0.32.3

Three final-source invocations ran sequentially, without concurrent local GPU tests. All matched **256/256 windows**. Run 2 is shown in full because it has the middle batch-32 CPU-relative speedup, not because it is the fastest:

| Windows/batch | 8 CPU processes | GPU resident | GPU host-to-host | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.256 ms | 4.159 ms | 4.118 ms | 3.67× |
| 8 | 16.112 ms | 4.675 ms | 4.668 ms | 3.45× |
| 32 | 60.375 ms | 5.558 ms | 5.553 ms | **10.86×** |
| 64 | 120.353 ms | 6.191 ms | 6.177 ms | **19.44×** |

| Run | Batch-32 GPU median | Batch-32 speedup | Batch-64 GPU median | Batch-64 speedup |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5.545 ms | 10.91× | 6.096 ms | 19.63× |
| 2 | 5.558 ms | 10.86× | 6.191 ms | 19.44× |
| 3 | 5.657 ms | 10.75× | 6.075 ms | 19.83× |

The middle run's batch-32 GPU time is **7.9% lower** than the previous representative 6.032 ms. All three invocations now exceed 10× at batch 32, but three repetitions on one device/corpus are not a universal guarantee. Repetitions use the same corpus prefixes, not independent data samples. Independently sampled resident and host-to-host medians can cross because of timing noise.

Cooperatively preparing child-pair adjacency, label-map initialization and strict epsilon predicates reduced the measured hierarchy stage from about 2.19 to 1.96 ms. The separate cooperative-lambda experiment added no reliable gain and was removed. EOM still performs the original ordered additions; epsilon checks preserve both the reciprocal round trip and the distinct strict comparisons.

Final reports are `results/round2-metal-verified-{1,2,3}.json`; code/configuration digests and the source archive are `results/round2-final-source.{sha256,tar.gz}`. Preliminary `round2-metal-benchmark-*` runs are retained but are not the final-source table above.

### Remaining limits

The approximately **3 ms CUDA / 4 ms Metal batch-32 stretch goals were not reached**. The gains above do not change the labels-only contract, float32 counterexample, CPU reference, corpus or comparator. No approximate neighbours, tolerance changes, jitter, fast-math substitution or CPU clustering fallback were introduced.

See the [validation record](../tests/README.md) for exact-state, capacity/fallback and sanitizer coverage. Full-size Metal shader instrumentation remains limited by the previously recorded hardware/compiler restrictions. Complete encoder-to-clustering wall time, broader held-out coverage, unrepresented operating modes and other hardware remain unverified.

## Previous cooperative milestone: private learned embeddings (historical)

Verified 10 October 2026. Both implementations consume the same frozen **private 256-window `[256,1024,8]` learned-embedding corpus**, with `min_cluster_size=2`, self-inclusive `min_samples=2` and epsilon 0.1. Every window matches the pinned sklearn reference's exact noise mask and partition before timing starts.

The comparator is **eight persistent CPU processes, one numerical-library thread each**, not serial or threaded sklearn. Each invocation uses five warmups and 30 synchronized samples per batch size, including separate resident and host-to-host GPU measurements. CPU process startup and encoder inference are excluded; CPU per-call dispatch/serialization is included. The GPU still performs normal input/distance validation. No approximation, tolerance change, precision reduction or CPU clustering fallback was introduced.

### NVIDIA GH200 120GB

Final Slurm job **`7242060`**, interactive partition/reservation, eight allocated CPU cores:

| Precision | Windows/batch | 8 CPU processes | GPU resident | GPU host-to-host | Resident speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| float64 | 1 | 17.529 ms | 3.445 ms | 3.487 ms | 5.09× |
| float64 | 8 | 18.557 ms | 4.449 ms | 4.531 ms | 4.17× |
| float64 | 32 | 75.042 ms | 4.988 ms | 5.182 ms | **15.04×** |
| float32 | 1 | 17.313 ms | 3.228 ms | 3.274 ms | 5.36× |
| float32 | 8 | 18.522 ms | 3.949 ms | 4.013 ms | 4.69× |
| float32 | 32 | 75.739 ms | 4.584 ms | 4.717 ms | **16.52×** |

This cleared the 10× target at batch 32 in both precisions. It did not imply a 10× latency improvement for single windows or batches of eight. Reports, source snapshot, all timing samples/p95s and sanitizer/JUnit evidence are in `results/optim-7242060/`; all 22 source/configuration digests matched the tested revision.

### Apple M2 Ultra / MLX 0.32.3

Three complete benchmark invocations were run sequentially, after validation and without competing GPU tests. They all matched **256/256 windows**. Each repeated the same corpus prefixes; these are repeatability checks, not additional independent data coverage.

Run C is shown in full below. It is the middle CPU-relative speedup of the three invocations at each batch size, not the fastest run:

| Windows/batch | 8 CPU processes | GPU resident | GPU host-to-host | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.328 ms | 4.324 ms | 4.349 ms | 3.54× |
| 8 | 16.050 ms | 4.884 ms | 4.947 ms | 3.29× |
| 32 | 60.665 ms | 6.032 ms | 5.860 ms | **10.06×** |
| 64 | 119.819 ms | 6.348 ms | 6.446 ms | **18.88×** |

The full batch-32/64 repeatability record:

| Run | Batch-32 GPU median | Batch-32 speedup | Batch-64 GPU median | Batch-64 speedup |
| --- | ---: | ---: | ---: | ---: |
| A | 6.594 ms | 9.23× | 6.312 ms | 19.08× |
| B | 5.760 ms | 10.56× | 7.251 ms | 16.61× |
| C | 6.032 ms | 10.06× | 6.348 ms | 18.88× |

**Metal's batch-32 median reaches the target, but it does not reliably exceed 10× in every invocation.** One run was 9.23×. Batch 64 cleared 10× in all three runs, at 16.61–19.08×. Smaller batches remained around 3.3–3.5×. Independently sampled host-to-host and resident medians can cross because of timing variation; this is not negative transfer cost.

Raw reports are `results/metal-verified-{a,b,c}.json`, accompanied by final source hashes/archive and validation logs. The corpus SHA-256 is `f14d5c9a2dc578d3c0d3c44288923eb81d3af6ad68d051fa5429c9c5ca3bdc55` on both devices.

### What produced these gains

- CUDA moves graph/tree workspace into opt-in shared memory when device capacity permits; larger supported windows retain a device-global path.
- Metal uses packed threadgroup storage, register-held cooperative Prim state, native SIMD reductions and parallel independent sorting partitions.
- Both skip core-distance partitioning when self-inclusive `min_samples <= 2`: the resulting mutual-reachability weights are mathematically identical to Euclidean weights.
- Balanced leaf-only union-find reduces pointer chasing while preserving the exact linkage tree. EOM, ancestor suppression, epsilon traversal and labels use ordered or topological linear passes rather than repeated whole-tree walks.
- Metal separates Prim/sort from its serial hierarchy tail using typed, device-resident sorted edges. Its hierarchy specializes minimum size two, preserves accumulation order and cooperatively writes final labels.
- Symmetric distances avoid duplicate computation without changing coordinate accumulation order.

Exact intermediate-state tests, frozen old linkage/condensation/EOM algorithms, real-corpus parity and device instrumentation guard these changes. See the [validation record](../tests/README.md), including full-size Metal shader-instrumentation limits.

Rejected experiments included a one-barrier double-buffered Prim reduction, alternative register blocking and consolidated validation submission: they did not reliably improve full-API timing. The slower pipelined-Prim profile and other intermediate reports remain under `results/`; its code is not in production.

**Boundary:** these are steady-state clustering measurements on one NVIDIA and one Apple GPU, using the private sampled corpus. Full encoder-to-clustering wall time, broader/held-out coverage, unrepresented operating modes, other devices, and a reliable margin above 10× at Metal batch 32 remain follow-up work.

## First CUDA optimisation milestone (historical)

Verified 10 October 2026 on Isambard's NVIDIA GH200 120GB, job `7240817`, using the same frozen 256-window corpus and pinned sklearn reference as the baseline below. Both float64 and float32 matched exact noise masks and partitions on **all 256 windows**.

The CPU comparator is eight independent, fully initialized processes on eight allocated cores, not the weaker serial/threaded baseline. Numerical-library threads are limited to one per process. Each median below uses **three warmups and 15 synchronized repetitions** on identical corpus prefixes; process startup and encoder inference are excluded.

| Precision | Windows/batch | 8 CPU processes | GPU resident | GPU host-to-host | Resident speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| float64 | 1 | 17.500 ms | 6.632 ms | 6.678 ms | 2.64× |
| float64 | 8 | 18.696 ms | 13.209 ms | 13.276 ms | 1.42× |
| float64 | 32 | 74.218 ms | 15.727 ms | 15.959 ms | 4.72× |
| float32 | 1 | 17.309 ms | 6.370 ms | 6.421 ms | 2.72× |
| float32 | 8 | 18.316 ms | 12.717 ms | 12.781 ms | 1.44× |
| float32 | 32 | 74.167 ms | 14.738 ms | 14.855 ms | 5.03× |

Compared with the original GPU medians below, batch-32 clustering is **20.49× faster in float64** and **20.19× faster in float32**. The CPU-relative improvement is useful at batch 32, but is not uniformly fivefold: batch 8 improves only about 1.4×. These are sampled, steady-state clustering results on one GPU architecture, not an end-to-end training/inference claim.

### What changed and why

Profiling the original float64 implementation at batch 32 found about 317 ms in graph/hierarchy work out of 319 ms total. Distance computation and core-distance selection were not the main bottlenecks.

1. **Cooperative CUDA Prim:** 256 threads update disjoint vertices and reduce `(weight, global vertex index)`. Strict source updates preserve ties. Prim fell to about 3 ms, but the remaining hierarchy still kept the full API around 199 ms.
2. **Path-compressed union-find:** preserves root IDs and linkage insertion order; this removed some repeated walks but saved only a few milliseconds on this corpus.
3. **Ordered EOM adjacency:** replaces full-range child scans with ascending-ID child lists, preserving floating-point accumulation order.
4. **Linear ancestor suppression:** replaces repeated descendant walks with a forward propagation of winning ancestors. The old EOM algorithm remains a test-only device oracle; exact stability, selection state and numeric labels are checked, including a losing intermediate between winning ancestor and grandchild.

At this milestone, the hierarchy changes were shared with Metal but cooperative Prim was CUDA-only. Metal's tests and real-corpus parity passed, but its corresponding measurements below remained substantially slower than CPU.

Results, full timing samples/p95 values, stage profiles and source snapshots are under `results/optim-7240817/`. Earlier iterations (`7240505`, `7240570`, `7240616`, `7240669`) and initial profiles (`7240438`, `7240542`) are retained for attribution. Run the validation and benchmark workflow on your own frozen embedding corpus using `scripts/optimise-cuda.sbatch` with `HDBSCAN_CORPUS` set to that corpus; the private corpus used here is not distributed. See [validation details](../tests/README.md) for test and sanitizer coverage.

Remaining work at this milestone: verify complete encoder-to-clustering wall time, broaden held-out and operating-mode coverage and profile the remaining sequential hierarchy. Any deployment needs its own parity and performance validation. No policy, precision tolerance or CPU fallback was introduced to obtain the speedup.

## Metal before cooperative Prim (historical)

Measured on Apple M2 Ultra using MLX 0.32.3 after the shared hierarchy optimisations, with Metal's Prim implementation still serial. The same private frozen 256-window learned-embedding corpus matched exact sklearn noise masks and partitions on **all 256 windows**. Corpus SHA-256 was checked against the export manifest.

Three warmups, 15 synchronized repetitions, float32 GPU arithmetic, eight persistent CPU processes and one numerical-library thread per process:

| Windows/batch | 8 CPU processes | GPU resident | GPU host-to-host | GPU slowdown |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.592 ms | 187.040 ms | 188.791 ms | 12.0× |
| 8 | 16.112 ms | 326.728 ms | 325.400 ms | 20.3× |
| 32 | 60.722 ms | 577.550 ms | 578.494 ms | 9.5× |

The report for eight CPU processes is retained locally under ignored `results/`. This was not an acceleration result: cooperative Prim had not yet been ported to Metal. It motivated the deterministic SIMD/threadgroup implementation and the same exact ordered-edge, EOM-state, sklearn and full-corpus gates. The historical synthetic Metal table below uses a different corpus and cannot establish the speedup from the shared hierarchy changes.

## Initial NVIDIA GH200 baseline: private learned embeddings

Measured 10 October 2026 on Isambard's NVIDIA GH200 120GB, with eight allocated CPU cores, CuPy 14.2.0, NumPy 2.3.5 and scikit-learn 1.9.1. Jobs used the `interactive` partition and reservation. The implementation was not optimised between measurements.

The private frozen corpus contains 256 real learned-embedding windows: 128 validation and 128 test windows, 1,024 points × 8 dimensions. Both CUDA precisions matched sklearn's exact noise masks and partitions on **all 256 windows**, not only the benchmark prefixes. Corpus selection, checkpoint/input hashes and coverage limitations are recorded in [the validation record](../tests/README.md).

Configuration: `min_cluster_size=2`, `min_samples=None`, epsilon 0.1. Each table entry is median completed batch time after two warmups and five repetitions. Batch sizes use the first 1, 8 and 32 windows of the same shuffled corpus. Numerical libraries are limited to one thread per CPU worker. GPU calls are synchronized.

| Precision | Windows/batch | Serial CPU | 8 CPU processes | GPU resident | GPU host-to-host |
| --- | ---: | ---: | ---: | ---: | ---: |
| float64 | 1 | 16.932 ms | 19.670 ms | 192.143 ms | 192.081 ms |
| float64 | 8 | 153.425 ms | 18.526 ms | 297.558 ms | 297.677 ms |
| float64 | 32 | 584.969 ms | 72.935 ms | 322.218 ms | 322.840 ms |
| float32 | 1 | 17.145 ms | 19.800 ms | 178.638 ms | 178.620 ms |
| float32 | 8 | 148.430 ms | 18.483 ms | 274.857 ms | 275.004 ms |
| float32 | 32 | 590.841 ms | 75.056 ms | 297.591 ms | 297.181 ms |

“Precision” is the GPU arithmetic and supplied-array dtype; sklearn calculates with its float64 implementation. The encoder originally emitted bfloat16-autocast outputs saved as float32; using CUDA float64 does not recover lost encoder precision.

**Initial baseline conclusion:** batch-32 GPU clustering beat serial sklearn, but lost to the independent-process CPU baseline by **4.42× (float64)** and **3.96× (float32)**. At smaller batches it also lost. This motivated the optimisations recorded above.

Eight CPU threads were also tested and were slower than serial execution: batch-32 CPU medians were 1,031.558 ms (float64 input) and 1,015.517 ms (float32 input). Those weaker threaded baselines are retained in the raw reports, not used to inflate the GPU result.

### Measurement controls and evidence

- Initial serial/threaded reports and frozen corpus: job `7240292`, retained locally under ignored `results/`.
- Final process/GPU measurements: job `7240336`, retained locally under ignored `results/`. All 26 selected harness tests passed on GH200, with no selected-test skips. Source digests, corpus digest and JUnit counts were verified after retrieval.
- Preliminary process run `7240322` is retained but superseded: lazy worker startup contaminated early small-batch measurements. The final harness uses a startup barrier so all workers finish importing before timing; startup is reported separately (2.28–2.36 seconds).
- Processes are persistent and use `spawn`, not a fork of the GPU runtime. CPU timings include per-call serialization and dispatch. Process startup and encoder inference are excluded; this is a steady-state clustering benchmark, not end-to-end application timing.
- Resident GPU inputs are uploaded before timing. Host-to-host timings include upload and returned labels. Tiny reversals between these independently measured medians are timing variation, not negative transfer overhead.
- JSON files retain all five timing samples, p95 values, affinity, thread limits, software versions and input hashes. Logs, exact job scripts and source snapshots accompany the reports under ignored `results/`; weights and raw datasets are not bundled.

This baseline identified the serial per-window graph/tree kernel as the optimisation target, without changing policy or weakening parity. Coverage of unrepresented operating modes, broader held-out sampling, additional architectures and complete encoder-to-clustering wall time remain outstanding.

## Initial Metal measurements

Measured 10 October 2026 on **Apple M2 Ultra**, using MLX 0.32.3, NumPy 2.3.5 and scikit-learn 1.9.1. Inputs were synthetic separated Gaussian mixtures, seed 42, 1,024 points × 8 dimensions per window. This is not a benchmark on the private learned-embedding corpus.

Configuration: `min_cluster_size=2`, `min_samples=None` (therefore 2), epsilon 0.1, float32 GPU arithmetic; sklearn uses its float64 `auto` path. CPU window processing was serial (`--cpu-workers 1`).

| Windows/batch | CPU median batch time | GPU device-resident median | CPU time / GPU time | Partition + noise parity |
| ---: | ---: | ---: | ---: | --- |
| 1 | 11.853 ms | 213.015 ms | 0.056× | Exact on tested window |
| 8 | 95.288 ms | 393.466 ms | 0.242× | Exact on all tested windows |
| 32 | 372.185 ms | 669.442 ms | 0.556× | Exact on all tested windows |

These are one warmup and three timed repetitions, with GPU completion synchronized. Lower times are better. **The prototype is slower than the CPU at every measured batch size.** There is no acceleration claim.

Reproduce with:

```sh
uv run python -m benchmarks.benchmark --backend mlx \
  --batch-sizes 1 8 32 --warmup 1 --repeats 3 \
  --output benchmark-results/metal.json
```

This historical implementation ran Prim and the hierarchy serially in one GPU thread per window, before the shared hierarchy improvements above. Current Metal Prim is cooperative. Compare against multi-worker CPU execution as well; serial sklearn is not the strongest possible CPU baseline.

These historical Metal measurements are synthetic, not the private learned-embedding corpus measured above on CUDA and Metal, and do not substantiate general speedup claims. Complete encoder-to-clustering wall time remains unmeasured. A first invocation in the benchmark process is not necessarily a genuinely cold compilation: driver caches may already be warm.
