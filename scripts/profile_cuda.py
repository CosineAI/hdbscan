"""Measure synchronized CUDA stages on a frozen embedding corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from benchmarks.benchmark import timings
from hdbscan_gpu import hdbscan
from hdbscan_gpu._cuda import CLUSTER_THREADS, _cluster_kernel, _kernels, _production_kernel
from hdbscan_gpu._kernels import INTEGER_WORKSPACE, PRIM_SOURCE, REAL_WORKSPACE, header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--variants", action="store_true", help="compare exact graph specializations")
    args = parser.parse_args()
    corpus = np.load(args.input, allow_pickle=False)
    records = []
    distance_kernel, global_kernel = _kernels(args.precision, unit_core=True)
    prim_kernel = cp.RawKernel(
        header("cuda", args.precision) + "#define PARALLEL_PRIM\n#define UNIT_CORE\n"
        + 'extern "C" __global__ void prim_profile(const Real* distances, const Real* core, '
        + "const int* params, const Real* options, int* labels, int* integers, Real* reals) {"
        + PRIM_SOURCE + "}",
        "prim_profile", options=("--fmad=false",),
    )
    dtype = getattr(cp, args.precision)
    for batch in args.batch_sizes:
        x = cp.ascontiguousarray(cp.asarray(corpus[:batch], dtype=dtype))
        if len(x) != batch:
            raise ValueError("not enough corpus windows")
        _, n, dimensions = x.shape
        cluster_kernel, shared_mem, cluster_threads = _production_kernel(args.precision, n, 2, 2)
        params = cp.asarray([n, dimensions, batch, 2], dtype=cp.int32)
        options = cp.asarray([0.1], dtype=dtype)
        distances = cp.empty((batch, n, n), dtype=dtype)
        labels = cp.empty((batch, n), dtype=cp.int32)
        integers = cp.empty((batch, INTEGER_WORKSPACE * n), dtype=cp.int32)
        reals = cp.empty((batch, REAL_WORKSPACE * n), dtype=dtype)

        def distance():
            distance_kernel(((distances.size + 255) // 256,), (256,), (x, params, distances))

        distance()
        core = cp.ascontiguousarray(cp.partition(distances, 1, axis=-1)[..., 1])

        def graph():
            cluster_kernel(
                (batch,), (cluster_threads,),
                (distances, core, params, options, labels, integers, reals),
                shared_mem=shared_mem,
            )

        def complete(operation):
            def measured():
                operation()
                cp.cuda.get_current_stream().synchronize()
            return measured

        operations = {
            "input_validation": lambda: cp.all(cp.isfinite(x)).item(),
            "distances": distance,
            "distance_validation": lambda: cp.all(cp.isfinite(distances)).item(),
            "core_distances_general": lambda: cp.ascontiguousarray(cp.partition(distances, 1, axis=-1)[..., 1]),
            "prim_only": lambda: prim_kernel(
                (batch,), (CLUSTER_THREADS,),
                (distances, core, params, options, labels, integers, reals),
            ),
            "graph_and_hierarchy": graph,
            "graph_global_workspace": lambda: global_kernel(
                (batch,), (CLUSTER_THREADS,),
                (distances, core, params, options, labels, integers, reals),
            ),
            "full_api": lambda: hdbscan(
                x, backend="cuda", precision=args.precision,
                min_cluster_size=2, cluster_selection_epsilon=0.1,
            ),
        }
        result = {
            name: timings(complete(operation), warmup=2, repeats=5, windows=batch)
            for name, operation in operations.items()
        }
        variants = {}
        if args.variants:
            graph()
            expected = labels.copy()
            for minimum_two, register_prim, threads, parallel_sort in (
                (False, 0, 256, False), (True, 0, 256, False), (False, 4, 256, False),
                (True, 4, 256, False), (True, 8, 128, False), (True, 2, 512, False),
                (True, 1, 1024, False), (True, 2, 512, True), (True, 1, 1024, True),
            ):
                if register_prim and n > register_prim * threads:
                    continue
                kernel, shared = _cluster_kernel(
                    args.precision, n, unit_core=True,
                    minimum_two=minimum_two, register_prim=register_prim, threads=threads, parallel_sort=parallel_sort,
                )

                def variant():
                    kernel(
                        (batch,), (threads,),
                        (distances, core, params, options, labels, integers, reals),
                        shared_mem=shared,
                    )

                variant()
                if not bool(cp.all(labels == expected).item()):
                    raise AssertionError("graph specialization changed numeric labels")
                variants[f"minimum_two={minimum_two},register_prim={register_prim},threads={threads},parallel_sort={parallel_sort}"] = {
                    **timings(complete(variant), warmup=5, repeats=30, windows=batch),
                    "exact_label_parity": True, "kernel_attributes": kernel.attributes,
                    "dynamic_shared_bytes": shared,
                }
        records.append({
            "shape": list(x.shape), "stages": result, "dynamic_shared_bytes": shared_mem,
            "kernel_attributes": cluster_kernel.attributes, "variants": variants, "cluster_threads": cluster_threads,
        })
        print(batch, {key: round(value["median_batch_ms"], 3) for key, value in result.items()}, flush=True)
        if variants:
            print("variants", {key: round(value["median_batch_ms"], 3) for key, value in variants.items()}, flush=True)
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "precision": args.precision, "input_sha256": digest, "reference_cluster_threads": CLUSTER_THREADS,
        "cupy": cp.__version__, "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cases": records,
        "unit_core": True,
        "scope": (
            "synchronized warm stage timings; preallocated buffers except core selection and full API; "
            "prim_only and graph_global_workspace are legacy reference variants, not production-stage attribution"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
