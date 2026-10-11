from functools import lru_cache

import mlx.core as mx

from ._kernels import (
    CLUSTER_SOURCE,
    CLUSTER_PREAMBLE,
    CLUSTER_THREADS,
    CONDENSE_SOURCE,
    DISTANCE_SOURCE,
    EOM_SOURCE,
    INTEGER_WORKSPACE,
    LABEL_SOURCE,
    LINKAGE_SOURCE,
    PRIM_SOURCE,
    REAL_WORKSPACE,
    SORT_SOURCE,
    header,
    metal_source,
)


@lru_cache(maxsize=192)
def _kernels(capacity=0, register_prim=0, dimensions=0, distance_checks=False, unit_core=False):
    common = header("mlx", "float32")
    distances = mx.fast.metal_kernel(
        name="hdbscan_distances",
        input_names=["x", "params"],
        output_names=["distances", "invalid"] if distance_checks else ["distances"],
        source=metal_source(DISTANCE_SOURCE),
        header=common + (f"#define DISTANCE_DIMENSIONS {dimensions}\n" if dimensions else "")
        + ("#define DISTANCE_CHECKS\n" if distance_checks else ""),
    )
    clusters = mx.fast.metal_kernel(
        name="hdbscan_clusters",
        input_names=["distances", "core", "params", "options"],
        output_names=["labels", "integers", "reals"],
        source=metal_source(CLUSTER_SOURCE),
        header=common + "#define PARALLEL_PRIM\n" + (
            f"#define METAL_PACKED_WORKSPACE\n#define PARALLEL_SORT\n#define WORKSPACE_CAPACITY {capacity}\n"
            if capacity else ""
        ) + (f"#define REGISTER_PRIM {register_prim}\n" if register_prim else "")
        + ("#define UNIT_CORE\n" if unit_core else ""),
    )
    return distances, clusters


_SPLIT_HIERARCHY_SOURCE = CLUSTER_PREAMBLE + r"""
for (int i = lane; i < n - 1; i += GROUP_SIZE) {
    int input = b * (n - 1) + i;
    edge_u[i] = vertices[2 * input];
    edge_v[i] = vertices[2 * input + 1];
    edge_w[i] = weights[input];
    order[i] = i;
}
for (int i = lane; i < 2 * n; i += GROUP_SIZE) uf[i] = i;
BARRIER();
""" + LINKAGE_SOURCE + CONDENSE_SOURCE + EOM_SOURCE + LABEL_SOURCE


@lru_cache(maxsize=32)
def _split_kernels(capacity, unit_core, minimum_two=False):
    common = (
        header("mlx", "float32")
        + "#define PARALLEL_PRIM\n#define METAL_PACKED_WORKSPACE\n#define PARALLEL_SORT\n"
        + f"#define WORKSPACE_CAPACITY {capacity}\n"
        + ("#define UNIT_CORE\n" if unit_core else "")
    )
    edges = mx.fast.metal_kernel(
        name="hdbscan_sorted_edges", input_names=["distances", "core", "params", "options"],
        output_names=["vertices", "weights"],
        source=metal_source(PRIM_SOURCE + SORT_SOURCE + r"""
for (int i = lane; i < n - 1; i += GROUP_SIZE) {
    int edge = order[i], output = b * (n - 1) + i;
    vertices[2 * output] = edge_u[edge];
    vertices[2 * output + 1] = edge_v[edge];
    weights[output] = edge_w[edge];
}
"""),
        header=common + "#define REGISTER_PRIM 1\n",
    )
    hierarchy = mx.fast.metal_kernel(
        name="hdbscan_hierarchy", input_names=["vertices", "weights", "params", "options"],
        output_names=["labels"],
        source=metal_source(_SPLIT_HIERARCHY_SOURCE),
        header=common + "#define COOPERATIVE_LABELS\n#define COOPERATIVE_BINARY_EOM\n"
        + ("#define MINIMUM_TWO\n" if minimum_two else ""),
    )
    return edges, hierarchy


def cluster(x, min_cluster_size, min_samples, epsilon, precision):
    if not mx.metal.is_available():
        raise RuntimeError("the MLX backend requires a Metal GPU")
    if isinstance(x, mx.array) and (mx.issubdtype(x.dtype, mx.complexfloating) or x.dtype == mx.bool_):
        raise ValueError("X must contain real numeric values")
    with mx.stream(mx.gpu):
        x = mx.array(x, dtype=mx.float32) if not isinstance(x, mx.array) else x.astype(mx.float32)
        unbatched = x.ndim == 2
        if unbatched:
            x = x[None]
        batch, n, dimensions = x.shape
        params = mx.array([n, dimensions, batch, min_cluster_size], dtype=mx.int32)
        options = mx.array([epsilon], dtype=mx.float32)
        capacity = 1 << (n - 1).bit_length() if n <= 1024 else 0
        threads = max(32, capacity) if capacity else CLUSTER_THREADS
        distance_kernel, cluster_kernel = _kernels(
            capacity, register_prim=1 if capacity else 0, dimensions=8 if dimensions == 8 else 0,
            distance_checks=True, unit_core=min_samples <= 2,
        )
        distances, invalid = distance_kernel(
            inputs=[x, params],
            grid=(batch * n * n, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, n, n), ((batch * n * n + 31) // 32,)],
            output_dtypes=[mx.float32, mx.int32],
            stream=mx.gpu,
        )
        validation = int(mx.max(invalid).item())
        if validation == 2:
            raise ValueError("X must contain only finite values in the requested precision")
        if validation:
            raise ValueError("pairwise distances overflow the requested precision")
        # With self-inclusive k <= 2, mutual reachability equals Euclidean distance.
        core = (
            distances if min_samples <= 2 else
            mx.partition(distances, kth=min_samples - 1, axis=-1, stream=mx.gpu)[..., min_samples - 1]
        )
        if capacity:
            edge_kernel, hierarchy_kernel = _split_kernels(capacity, min_samples <= 2, min_cluster_size == 2)
            vertices, weights = edge_kernel(
                inputs=[distances, core, params, options],
                grid=(batch * threads, 1, 1), threadgroup=(threads, 1, 1),
                output_shapes=[(batch, n - 1, 2), (batch, n - 1)],
                output_dtypes=[mx.int16, mx.float32], stream=mx.gpu,
            )
            labels = hierarchy_kernel(
                inputs=[vertices, weights, params, options],
                grid=(batch * 32, 1, 1), threadgroup=(32, 1, 1),
                output_shapes=[(batch, n)], output_dtypes=[mx.int32], stream=mx.gpu,
            )[0]
        else:
            labels = cluster_kernel(
                inputs=[distances, core, params, options],
                grid=(batch * threads, 1, 1),
                threadgroup=(threads, 1, 1),
                output_shapes=[
                    (batch, n), (batch, INTEGER_WORKSPACE * n), (batch, REAL_WORKSPACE * n),
                ],
                output_dtypes=[mx.int32, mx.int32, mx.float32],
                stream=mx.gpu,
            )[0]
        return labels[0] if unbatched else labels
