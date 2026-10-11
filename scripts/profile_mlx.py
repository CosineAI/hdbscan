"""Measure evaluated Metal stages on a frozen embedding corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from benchmarks.benchmark import timings
from hdbscan_gpu import hdbscan
from hdbscan_gpu._kernels import (
    CONDENSE_SOURCE, EOM_SOURCE, INTEGER_WORKSPACE, LINKAGE_SOURCE, PRIM_SOURCE,
    REAL_WORKSPACE, SORT_SOURCE, header, metal_source,
)
from hdbscan_gpu._mlx import _SPLIT_HIERARCHY_SOURCE, _kernels, _split_kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--threads", type=int, nargs="+", choices=[32, 64, 128, 256, 512, 1024], default=[256, 512, 1024])
    parser.add_argument("--hierarchy-stages", action="store_true")
    parser.add_argument("--cooperative-variants", action="store_true")
    args = parser.parse_args()
    corpus = np.load(args.input, allow_pickle=False)
    records = []
    with mx.stream(mx.gpu):
        for batch in args.batch_sizes:
            x = mx.array(corpus[:batch], dtype=mx.float32)
            if len(x) != batch:
                raise ValueError("not enough corpus windows")
            _, n, dimensions = x.shape
            params = mx.array([n, dimensions, batch, 2], dtype=mx.int32)
            options = mx.array([0.1], dtype=mx.float32)
            core = mx.zeros((batch, n), dtype=mx.float32)
            shapes = [(batch, n), (batch, INTEGER_WORKSPACE * n), (batch, REAL_WORKSPACE * n)]
            dtypes = [mx.int32, mx.int32, mx.float32]
            distance_kernel, _ = _kernels(dimensions=8 if dimensions == 8 else 0)

            def distance():
                return distance_kernel(
                    inputs=[x, params], grid=(batch * n * n, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(batch, n, n)], output_dtypes=[mx.float32], stream=mx.gpu,
                )[0]

            distances = distance()
            mx.eval(distances, core, x)
            operations = {
                "distances": distance,
                "core_distances_general": lambda: mx.partition(distances, kth=1, axis=-1)[..., 1],
                "full_api": lambda: hdbscan(
                    x, min_cluster_size=2, cluster_selection_epsilon=0.1, backend="mlx",
                ),
            }
            if n <= 1024:
                capacity = 1 << (n - 1).bit_length()
                edge_kernel, hierarchy_kernel = _split_kernels(capacity, True, True)
                edge_threads = max(32, capacity)

                def sorted_edges():
                    return edge_kernel(
                        inputs=[distances, core, params, options],
                        grid=(batch * edge_threads, 1, 1), threadgroup=(edge_threads, 1, 1),
                        output_shapes=[(batch, n - 1, 2), (batch, n - 1)],
                        output_dtypes=[mx.int16, mx.float32], stream=mx.gpu,
                    )

                def hierarchy(edges, kernel=hierarchy_kernel):
                    return kernel(
                        inputs=[*edges, params, options],
                        grid=(batch * 32, 1, 1), threadgroup=(32, 1, 1),
                        output_shapes=[(batch, n)], output_dtypes=[mx.int32], stream=mx.gpu,
                    )[0]

                edges = sorted_edges()
                mx.eval(edges)
                operations.update({
                    "sorted_edges": sorted_edges,
                    "hierarchy_from_sorted_edges": lambda: hierarchy(edges),
                    "split_graph": lambda: hierarchy(sorted_edges()),
                })
                if args.cooperative_variants:
                    expected = hierarchy(edges)
                    mx.eval(expected)
                    for binary in (False, True):
                        variant_kernel = mx.fast.metal_kernel(
                            name="profile_hierarchy_variant",
                            input_names=["vertices", "weights", "params", "options"], output_names=["labels"],
                            source=metal_source(_SPLIT_HIERARCHY_SOURCE),
                            header=header("mlx", "float32")
                            + "#define PARALLEL_PRIM\n#define PARALLEL_SORT\n#define METAL_PACKED_WORKSPACE\n"
                            + f"#define WORKSPACE_CAPACITY {capacity}\n#define COOPERATIVE_LABELS\n#define MINIMUM_TWO\n"
                            + ("#define COOPERATIVE_BINARY_EOM\n" if binary else ""),
                        )
                        observed = hierarchy(edges, variant_kernel)
                        if not bool(mx.all(observed == expected).item()):
                            raise AssertionError("cooperative hierarchy changed numeric labels")
                        operations[f"hierarchy_binary={binary}"] = (
                            lambda kernel=variant_kernel: hierarchy(edges, kernel)
                        )
            for name, capacity, register_prim in (
                ("global", 0, 0), ("packed", 1 << (n - 1).bit_length(), 0),
                ("registers", 1 << (n - 1).bit_length(), 4),
                ("one_register", 1 << (n - 1).bit_length(), 1),
            ):
                if capacity > 1024:
                    continue
                _, graph_kernel = _kernels(capacity, register_prim, unit_core=True)
                # Observable edge outputs prevent dead-code elimination of shared Prim.
                prim_kernel = mx.fast.metal_kernel(
                    name=f"profile_prim_{name}",
                    input_names=["distances", "core", "params", "options"],
                    output_names=["labels", "integers", "reals"],
                    source=metal_source(PRIM_SOURCE + r"""
for (int i = 0; i < n - 1; ++i) {
    integers[b * 32 * n + i] = edge_u[i];
    integers[b * 32 * n + n + i] = edge_v[i];
    reals[b * 10 * n + i] = edge_w[i];
}
"""),
                    header=header("mlx", "float32") + "#define PARALLEL_PRIM\n#define UNIT_CORE\n" + (
                        f"#define METAL_PACKED_WORKSPACE\n#define WORKSPACE_CAPACITY {capacity}\n" if capacity else ""
                    ) + (f"#define REGISTER_PRIM {register_prim}\n" if register_prim else ""),
                )

                def launch(kernel, threads):
                    return kernel(
                        inputs=[distances, core, params, options],
                        grid=(batch * threads, 1, 1), threadgroup=(threads, 1, 1),
                        output_shapes=shapes, output_dtypes=dtypes, stream=mx.gpu,
                    )

                for threads in args.threads:
                    if register_prim and n > register_prim * threads:
                        continue
                    operations[f"prim_{name}_{threads}"] = lambda kernel=prim_kernel, t=threads: launch(kernel, t)
                    operations[f"graph_{name}_{threads}"] = lambda kernel=graph_kernel, t=threads: launch(kernel, t)
                if args.hierarchy_stages and capacity and not register_prim:
                    prefix = PRIM_SOURCE
                    for stage, source, exported in (
                        ("sort", SORT_SOURCE, r"""
for (int i = 0; i < n - 1; ++i) {
    integers[b * 32 * n + i] = order[i];
    integers[b * 32 * n + n + i] = edge_u[i];
    reals[b * 10 * n + i] = edge_w[i];
}
"""),
                        ("linkage", LINKAGE_SOURCE, r"""
for (int i = n; i < 2 * n - 1; ++i) {
    integers[b * 32 * n + i - n] = left[NODE_INDEX(i)];
    integers[b * 32 * n + i] = right[NODE_INDEX(i)];
    reals[b * 10 * n + i - n] = distance[NODE_INDEX(i)];
}
"""),
                        ("condense", CONDENSE_SOURCE, ""),
                        ("eom", EOM_SOURCE, ""),
                    ):
                        prefix += source
                        if not exported:
                            exported = r"""
for (int i = n; i < next_label; ++i) {
    integers[b * 32 * n + i - n] = parent[CLUSTER_INDEX(i)];
    integers[b * 32 * n + i] = selected[CLUSTER_INDEX(i)];
    reals[b * 10 * n + i - n] = birth[CLUSTER_INDEX(i)];
    reals[b * 10 * n + i] = stability[CLUSTER_INDEX(i)];
}
for (int i = 0; i < n; ++i) labels[b * n + i] = point_parent[i];
"""
                        stage_kernel = mx.fast.metal_kernel(
                            name=f"profile_{stage}", input_names=["distances", "core", "params", "options"],
                            output_names=["labels", "integers", "reals"],
                            source=metal_source(prefix + exported),
                            header=header("mlx", "float32") + "#define PARALLEL_PRIM\n"
                            + f"#define METAL_PACKED_WORKSPACE\n#define WORKSPACE_CAPACITY {capacity}\n",
                        )
                        operations[f"through_{stage}"] = lambda kernel=stage_kernel: launch(kernel, 256)

            def complete(operation):
                def measured():
                    # MLX is lazy: synchronize alone does not evaluate a fresh result.
                    mx.eval(operation())
                    mx.synchronize()
                return measured

            stages = {
                name: timings(complete(operation), warmup=3, repeats=15, windows=batch)
                for name, operation in operations.items()
            }
            records.append({"shape": list(x.shape), "stages": stages})
            print(batch, {key: round(value["median_batch_ms"], 3) for key, value in stages.items()}, flush=True)
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "input_sha256": digest, "device": mx.device_info(), "cases": records,
        "scope": (
            "fresh evaluated outputs, completed GPU work; sorted_edges/hierarchy/split_graph use production kernels; "
            "fused graph and through_* variants are diagnostic alternatives, not additive production stage timings"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
