import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import sklearn
from sklearn.cluster import HDBSCAN

from hdbscan_gpu import hdbscan


def partition_equal(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.array_equal(actual == -1, expected == -1):
        return False
    forward, reverse = {}, {}
    for observed, reference in zip(actual.flat, expected.flat, strict=True):
        if reference == -1:
            continue
        observed, reference = int(observed), int(reference)
        if forward.setdefault(observed, reference) != reference:
            return False
        if reverse.setdefault(reference, observed) != observed:
            return False
    return True


def make_windows(batch, samples, dimensions, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (8, dimensions))
    membership = rng.integers(0, len(centers), (batch, samples))
    return (centers[membership] + rng.normal(0, 0.15, (batch, samples, dimensions))).astype(np.float32)


def timings(operation, warmup, repeats, windows):
    for _ in range(warmup):
        operation()
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        elapsed.append(perf_counter() - start)
    median = float(np.median(elapsed))
    return {
        "median_batch_ms": median * 1000,
        "p95_batch_ms": float(np.quantile(elapsed, 0.95)) * 1000,
        "windows_per_second": windows / median,
        "samples_seconds": elapsed,
    }


def validate_windows(values, cpu_one, gpu_batch, batch_size):
    failures = []
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        actual = gpu_batch(batch)
        for offset, (observed, window) in enumerate(zip(actual, batch, strict=True)):
            expected = cpu_one(window)
            if not partition_equal(observed, expected):
                failures.append({
                    "window_index": start + offset,
                    "noise_disagreements": int(np.count_nonzero((observed == -1) != (expected == -1))),
                    "cpu_clusters": int(len(np.unique(expected[expected >= 0]))),
                    "gpu_clusters": int(len(np.unique(observed[observed >= 0]))),
                })
    return {
        "total_windows": len(values),
        "matching_windows": len(values) - len(failures),
        "partition_and_noise_parity": not failures,
        "failures": failures,
    }


def write_report(report, output):
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    print(encoded, end="")


def cpu_labels(window, parameters):
    return HDBSCAN(
        algorithm="auto", metric="euclidean", cluster_selection_method="eom",
        allow_single_cluster=False, alpha=1.0, copy=True, n_jobs=1, **parameters,
    ).fit_predict(window)


def initialize_cpu_worker(barrier):
    # Warmups can finish on early workers while later workers still import sklearn.
    barrier.wait(timeout=120)


def main():
    parser = argparse.ArgumentParser(description="Compare actual completed GPU clustering with pinned sklearn.")
    parser.add_argument("--backend", choices=("mlx", "cuda"), default="mlx")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--dimensions", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--cpu-executor", choices=("threads", "processes"), default="threads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=Path, help="Optional frozen [N,D] or [B,N,D] .npy embeddings")
    parser.add_argument("--output", type=Path, help="Write a JSON report; exit nonzero on partition mismatch")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.cpu_workers < 1:
        parser.error("warmup must be nonnegative; repeats and cpu-workers must be positive")
    if args.samples < 2 or args.dimensions < 1 or min(args.batch_sizes) < 1:
        parser.error("samples must be >=2; dimensions and batch sizes must be positive")
    if sklearn.__version__ != "1.9.1" or np.__version__ != "2.3.5":
        parser.error("the reference requires scikit-learn==1.9.1 and numpy==2.3.5")

    if args.backend == "mlx":
        if args.precision != "float32":
            parser.error("the Metal backend supports float32 only")
        import mlx.core as device

        def upload(values):
            with device.stream(device.gpu):
                result = device.array(values, dtype=device.float32)
                device.eval(result)
            return result

        def complete(result):
            device.eval(result)
            device.synchronize(device.gpu)
            return result

        def host(result):
            return np.asarray(complete(result))

        hardware = device.device_info()
        backend_version = importlib.metadata.version("mlx")
    else:
        import cupy as device

        def upload(values):
            result = device.asarray(values, dtype=getattr(device, args.precision))
            device.cuda.get_current_stream().synchronize()
            return result

        def complete(result):
            device.cuda.get_current_stream().synchronize()
            return result

        def host(result):
            return device.asnumpy(complete(result))

        properties = device.cuda.runtime.getDeviceProperties(device.cuda.Device().id)
        name = properties["name"]
        hardware = {"device_name": name.decode() if isinstance(name, bytes) else name}
        backend_version = device.__version__

    parameters = {
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_epsilon": args.epsilon,
    }

    cpu_one = partial(cpu_labels, parameters=parameters)

    supplied = None
    if args.input:
        supplied = np.load(args.input, allow_pickle=False)
        if supplied.ndim == 2:
            supplied = supplied[None]
        if supplied.ndim != 3 or len(supplied) < max(args.batch_sizes):
            parser.error("--input needs [B,N,D] with enough windows for every requested batch size")
        supplied = np.ascontiguousarray(supplied, dtype=getattr(np, args.precision))

    report = {
        "schema_version": 1,
        "backend": args.backend,
        "precision": args.precision,
        "backend_version": backend_version,
        "hardware": hardware,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "cpu_workers": args.cpu_workers,
        "cpu_executor": "serial" if args.cpu_workers == 1 else args.cpu_executor,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "parameters": parameters,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "input": str(args.input) if args.input else "synthetic separated Gaussian mixtures",
        "timing_scope": "clustering only, not encoder inference; GPU completion is synchronized",
        "cases": [],
    }
    if supplied is not None:
        with args.input.open("rb") as stream:
            report["input_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        report["corpus_parity"] = validate_windows(
            supplied, cpu_one,
            lambda values: host(hdbscan(
                upload(values), backend=args.backend, precision=args.precision, **parameters
            )),
            max(args.batch_sizes),
        )
        if not report["corpus_parity"]["partition_and_noise_parity"]:
            write_report(report, args.output)
            return 2
    # Spawn avoids forking an already-initialized CUDA or Metal runtime.
    context = multiprocessing.get_context("spawn")
    pool = (
        ProcessPoolExecutor(
            max_workers=args.cpu_workers, mp_context=context,
            initializer=initialize_cpu_worker, initargs=(context.Barrier(args.cpu_workers),),
        )
        if args.cpu_executor == "processes"
        else ThreadPoolExecutor(max_workers=args.cpu_workers)
    )
    with pool as executor:
        if args.cpu_executor == "processes":
            start = perf_counter()
            ready = [executor.submit(os.getpid) for _ in range(args.cpu_workers)]
            for future in ready:
                future.result()
            report["cpu_worker_startup_seconds"] = perf_counter() - start
        for batch in args.batch_sizes:
            values = supplied[:batch] if supplied is not None else make_windows(
                batch, args.samples, args.dimensions, args.seed
            )

            def cpu():
                if args.cpu_workers == 1:
                    return np.stack([cpu_one(window) for window in values])
                return np.stack(list(executor.map(cpu_one, values)))

            native = upload(values)

            def gpu():
                return complete(hdbscan(
                    native, backend=args.backend, precision=args.precision, **parameters
                ))

            def gpu_with_transfer():
                result = hdbscan(
                    upload(values), backend=args.backend, precision=args.precision, **parameters
                )
                return host(result)

            expected = cpu()
            start = perf_counter()
            initial = host(gpu())
            first_call = perf_counter() - start
            matches = [
                partition_equal(actual, reference)
                for actual, reference in zip(initial, expected, strict=True)
            ]
            case = {
                "shape": list(values.shape),
                "matching_windows": sum(matches),
                "total_windows": batch,
                "partition_and_noise_parity": all(matches),
                "first_gpu_call_seconds": first_call,
                "first_call_note": "first timed call for this batch; corpus validation and driver caches may already be warm",
            }
            report["cases"].append(case)
            if not all(matches):
                break
            case.update({
                "cpu": timings(cpu, args.warmup, args.repeats, batch),
                "gpu_device_resident": timings(gpu, args.warmup, args.repeats, batch),
                "gpu_host_to_host": timings(gpu_with_transfer, args.warmup, args.repeats, batch),
            })
            case["resident_speedup_vs_cpu"] = (
                case["cpu"]["median_batch_ms"] / case["gpu_device_resident"]["median_batch_ms"]
            )
    write_report(report, args.output)
    return 0 if all(case["partition_and_noise_parity"] for case in report["cases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
