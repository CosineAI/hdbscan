import numpy as np
import pytest

from hdbscan_gpu._kernels import (
    CLUSTER_SOURCE, CLUSTER_THREADS, CONDENSE_SOURCE, EOM_SOURCE, INTEGER_ARRAYS, INTEGER_WORKSPACE,
    LABEL_SOURCE, LINKAGE_SOURCE, REAL_ARRAYS, REAL_WORKSPACE, header, metal_source,
)

# Retain the previous exhaustive scans as an independent device-side EOM oracle.
REFERENCE_EOM_SOURCE = r"""
for (int node = next_label - 1; node > n; --node) {
    Real subtree = 0;
    for (int child = node + 1; child < next_label; ++child)
        if (parent[child] == node) subtree += stability[child];
    if (subtree > stability[node]) {
        selected[node] = 0;
        stability[node] = subtree;
    } else {
        for (int child = node + 1; child < next_label; ++child) {
            int ancestor = parent[child];
            while (ancestor > node) ancestor = parent[ancestor];
            if (ancestor == node) selected[child] = 0;
        }
    }
}
"""

# Freeze the original epsilon and per-point ancestor walks independently of EOM.
REFERENCE_LABEL_SOURCE = r"""
if (epsilon > 0) {
    for (int node = n + 1; node < next_label; ++node) {
        if (!selected[node]) continue;
        int chosen = node;
        if (Real(1) / birth[node] < epsilon) {
            while (parent[chosen] != n) {
                chosen = parent[chosen];
                if (Real(1) / birth[chosen] > epsilon) break;
            }
        }
        label_map[chosen] = 1;
    }
    for (int node = n + 1; node < next_label; ++node)
        selected[node] = label_map[node] == 1;
}
int label = 0;
for (int node = n + 1; node < next_label; ++node)
    if (selected[node]) label_map[node] = label++;
for (int i = 0; i < n; ++i) {
    int cluster = point_parent[i];
    while (cluster != n && !selected[cluster]) cluster = parent[cluster];
    labels[b * n + i] = cluster == n ? -1 : label_map[cluster];
}
"""

REFERENCE_DISTANCE_SOURCE = r"""
int tid = GLOBAL_ID;
int n = params[0], d = params[1], batches = params[2];
if (tid >= batches * n * n) return;
int j = tid % n, i = (tid / n) % n, b = tid / (n * n);
Real total = 0;
for (int k = 0; k < d; ++k) {
    Real delta = x[(b * n + i) * d + k] - x[(b * n + j) * d + k];
    total = total + delta * delta;
}
distances[tid] = SQRT(total);
"""

REFERENCE_LINKAGE_SOURCE = r"""
for (int i = 0; i < n - 1; ++i) {
    int edge = order[i], a = edge_u[edge], c = edge_v[edge];
    int original_a = a, original_c = c;
    while (uf[a] != a) a = uf[a];
    while (uf[c] != c) c = uf[c];
    while (original_a != a) {
        int next = uf[original_a];
        uf[original_a] = a;
        original_a = next;
    }
    while (original_c != c) {
        int next = uf[original_c];
        uf[original_c] = c;
        original_c = next;
    }
    int node = n + i;
    left[node] = a; right[node] = c;
    sizes[node] = sizes[a] + sizes[c];
    distance[node] = edge_w[edge];
    uf[a] = node; uf[c] = node;
}
"""


REFERENCE_CONDENSE_SOURCE = r"""
int root = 2 * n - 2, head = 0, tail = 1, next_label = n + 1;
queue[0] = root;
relabel[root] = n;
while (head < tail) {
    int node = queue[head++], a = left[node], c = right[node];
    int cluster = relabel[node];
    Real lambda = distance[node] > 0 ? Real(1) / distance[node] : Real(INFINITY);
    bool large_a = sizes[a] >= minimum, large_c = sizes[c] >= minimum;
    if (large_a && large_c) {
        int children[2] = {a, c};
        for (int k = 0; k < 2; ++k) {
            int child = children[k], label = next_label++;
            relabel[child] = label;
            parent[label] = cluster;
            birth[label] = lambda;
            selected[label] = 1;
            stability[cluster] += (lambda - birth[cluster]) * Real(sizes[child]);
            queue[tail++] = child;
        }
    } else {
        int children[2] = {a, c};
        for (int k = 0; k < 2; ++k) {
            int child = children[k];
            if (sizes[child] >= minimum) {
                relabel[child] = cluster;
                queue[tail++] = child;
            } else {
                int prune_head = 0, prune_tail = 0, leaf = child;
                for (;;) {
                    if (leaf < n) {
                        point_parent[leaf] = cluster;
                        stability[cluster] += lambda - birth[cluster];
                    } else {
                        prune[prune_tail++] = left[leaf];
                        prune[prune_tail++] = right[leaf];
                    }
                    if (prune_head == prune_tail) break;
                    leaf = prune[prune_head++];
                }
            }
        }
    }
}
"""


@pytest.mark.parametrize("size", [2, 33, 129, 1024])
@pytest.mark.parametrize("dimensions", [1, 8, 31])
def test_distance_symmetry_and_specialization_are_bitwise_exact(backend, size, dimensions):
    module = backend.module
    rng = np.random.default_rng(size + dimensions)
    values = rng.normal(size=(3, size, dimensions)).astype(backend.precision)
    values[0, 0] = 0
    values[0, -1] = -0.0
    values[1, -1] = values[1, 0]
    values[2] *= 1e9
    native = backend.array(values)
    common = header(backend.name, backend.precision)
    if backend.name == "mlx":
        from hdbscan_gpu._mlx import _kernels

        kernels = [
            _kernels(dimensions=8 if dimensions == 8 else 0, distance_checks=True)[0],
            module.fast.metal_kernel(
                name="reference_distances", input_names=["x", "params"], output_names=["distances"],
                source=metal_source(REFERENCE_DISTANCE_SOURCE), header=common,
            ),
        ]
        with module.stream(module.gpu):
            params = module.array([size, dimensions, 3, 2], dtype=module.int32)
            results = []
            for index, kernel in enumerate(kernels):
                outputs = kernel(
                    inputs=[native, params], grid=(3 * size * size, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(3, size, size)] + ([((3 * size * size + 31) // 32,)] if index == 0 else []),
                    output_dtypes=[module.float32] + ([module.int32] if index == 0 else []), stream=module.gpu,
                )
                results.append(outputs[0])
                if index == 0:
                    assert not np.any(backend.host(outputs[1]))
    else:
        from hdbscan_gpu._cuda import _kernels

        kernels = [
            _kernels(backend.precision)[0],
            module.RawKernel(
                common + 'extern "C" __global__ void reference_distances('
                + "const Real* x, const int* params, Real* distances) {"
                + REFERENCE_DISTANCE_SOURCE + "}", "reference_distances", options=("--fmad=false",),
            ),
        ]
        params = module.asarray([size, dimensions, 3, 2], dtype=module.int32)
        results = []
        for kernel in kernels:
            result = module.empty((3, size, size), dtype=getattr(module, backend.precision))
            kernel(((result.size + 255) // 256,), (256,), (native, params, result))
            results.append(result)
    integer_type = np.uint32 if backend.precision == "float32" else np.uint64
    np.testing.assert_array_equal(
        backend.host(results[0]).view(integer_type), backend.host(results[1]).view(integer_type),
    )


def export_fields(fields):
    source = "\nif (lane == 0) {\n"
    for name, length in fields.items():
        real = name in REAL_ARRAYS
        names = REAL_ARRAYS if real else INTEGER_ARRAYS
        output = "reals" if real else "integers"
        stride = REAL_WORKSPACE if real else INTEGER_WORKSPACE
        value = f"{name}[i]"
        if name in ("parent", "selected", "label_map", "birth", "stability"):
            default = -1 if name in ("parent", "label_map") else 0
            value = f"i < n ? {default} : {name}[CLUSTER_INDEX(i)]"
        if name in ("left", "right", "sizes", "distance"):
            value = f"i >= n && i < 2 * n - 1 ? {name}[NODE_INDEX(i)] : 0"
        source += (
            f"\nfor (int i = 0; i < {length}; ++i) "
            f"{output}[b * {stride} * n + {2 * names.index(name)} * n + i] = {value};\n"
        )
    return source + "\n}\n"


def diagnostic_kernel(
    backend, *, reference_eom=False, shared_workspace=False, capacity=0, unit_core=False,
    register_prim=0, sort_depth=None, threads=CLUSTER_THREADS, linkage_only=False, minimum_two=False,
    condensation_only=False, parallel_sort=False,
):
    source = CLUSTER_SOURCE.split(CONDENSE_SOURCE)[0] if linkage_only else CLUSTER_SOURCE
    if reference_eom:
        source = source.replace(LINKAGE_SOURCE, REFERENCE_LINKAGE_SOURCE)
        source = source.replace(CONDENSE_SOURCE, REFERENCE_CONDENSE_SOURCE)
        source = source.replace(EOM_SOURCE, REFERENCE_EOM_SOURCE)
        source = source.replace(LABEL_SOURCE, REFERENCE_LABEL_SOURCE)
    if condensation_only:
        assert reference_eom
        source = source.split(REFERENCE_EOM_SOURCE)[0]
        source += "\nfor (int i = 0; i < n; ++i) labels[b * n + i] = 0;\n"
    if linkage_only:
        source += export_fields({"left": "2 * n", "right": "2 * n", "sizes": "2 * n", "distance": "2 * n"})
        source += "\nif (lane == 0) for (int i = 0; i < n; ++i) labels[b * n + i] = 0;\n"
    elif shared_workspace:
        # Export only defined fields. Scratch padding is deliberately uninitialised.
        edges = export_fields({
            "edge_u": "n - 1", "edge_v": "n - 1", "order": "n - 1", "edge_w": "n - 1",
        })
        source = source.replace(LINKAGE_SOURCE, edges + LINKAGE_SOURCE)
        source += export_fields({
            "parent": "2 * n", "selected": "2 * n", "label_map": "2 * n",
            "point_parent": "n", "birth": "2 * n", "stability": "2 * n",
        })
    common = header(backend.name, backend.precision)
    common += ("#define UNIT_CORE\n" if unit_core else "")
    common += ("#define MINIMUM_TWO\n" if minimum_two else "")
    common += ("#define PARALLEL_SORT\n" if parallel_sort or (backend.name == "mlx" and capacity) else "")
    common += (f"#define REGISTER_PRIM {register_prim}\n" if register_prim else "")
    common += (f"#define SORT_DEPTH_OVERRIDE {sort_depth}\n" if sort_depth is not None else "")
    if backend.name == "mlx":
        return backend.module.fast.metal_kernel(
            name="reference_eom", input_names=["distances", "core", "params", "options"],
            output_names=["labels", "integers", "reals"],
            source=metal_source(source), header=common + "#define PARALLEL_PRIM\n" + (
                f"#define METAL_PACKED_WORKSPACE\n#define WORKSPACE_CAPACITY {capacity}\n"
                if capacity else ""
            ),
        )
    return backend.module.RawKernel(
        common + "#define PARALLEL_PRIM\n"
        + ("#define CUDA_PACKED_WORKSPACE\n" if shared_workspace else "")
        + f'extern "C" __global__ __launch_bounds__({threads}) void reference_eom(const Real* distances, const Real* core, '
        + "const int* params, const Real* options, int* labels, int* integers, Real* reals) {"
        + source + "}", "reference_eom", options=("--fmad=false",),
    )


@pytest.mark.parametrize("size", [5, 17, 18, 33, 129, 1024, 4096])
@pytest.mark.parametrize("pattern", ["equal", "alternating", "quantized", "ascending", "descending"])
def test_gpu_indirect_quicksort_matches_pinned_numpy(backend, size, pattern):
    # There is no public standalone sorting helper. Inspect the actual kernel
    # workspace instead of duplicating its algorithm in a test-only kernel.
    assert np.__version__ == "2.3.5"
    rng = np.random.default_rng(size)
    weights = {
        "equal": np.ones(size - 1),
        "alternating": 1 + np.arange(size - 1) % 2,
        "quantized": rng.integers(1, 5, size - 1),
        "ascending": np.arange(1, size),
        "descending": np.arange(size - 1, 0, -1),
    }[pattern].astype(backend.precision)
    # A path graph with heavier non-path edges gives Prim the prescribed
    # edge order, with enough ties to distinguish quicksort from stable sort.
    distances = np.full((1, size, size), size * 10, dtype=backend.precision)
    np.fill_diagonal(distances[0], 0)
    for index, weight in enumerate(weights):
        distances[0, index, index + 1] = weight
        distances[0, index + 1, index] = weight
    packed = backend.name == "mlx" and size <= 1024
    integers, reals, _ = graph_workspace(
        backend, distances, np.zeros((1, size)), shared_workspace=packed, register_prim=1 if packed else 0,
        threads=max(32, 1 << (size - 1).bit_length()) if packed else CLUSTER_THREADS,
    )
    actual_order = integers["order"][0]
    actual_weights = reals["edge_w"][0]
    np.testing.assert_array_equal(actual_weights, weights)
    np.testing.assert_array_equal(actual_order, np.argsort(weights, kind="quicksort"))


def graph_workspace(
    backend, distances, core_distances, *, reference_eom=False, epsilon=0,
    shared_workspace=False, threads=CLUSTER_THREADS, unit_core=False, register_prim=0, sort_depth=None,
    linkage_only=False, minimum=2, minimum_two=False, condensation_only=False, parallel_sort=False,
):
    assert not (reference_eom and shared_workspace), "the independent oracle uses unpacked storage"
    batch, size, _ = distances.shape
    module = backend.module
    native_distances = backend.array(distances)
    core = backend.array(core_distances)
    options = backend.array([epsilon])
    if backend.name == "mlx":
        from hdbscan_gpu._mlx import _kernels

        with module.stream(module.gpu):
            params = module.array([size, 1, batch, minimum], dtype=module.int32)
            _, kernel = _kernels()
            if reference_eom or shared_workspace or unit_core or register_prim or sort_depth is not None or linkage_only or minimum_two or parallel_sort:
                kernel = diagnostic_kernel(
                    backend, reference_eom=reference_eom, shared_workspace=shared_workspace,
                    capacity=(1 << (size - 1).bit_length()) if shared_workspace else 0,
                    unit_core=unit_core, register_prim=register_prim, sort_depth=sort_depth, linkage_only=linkage_only,
                    minimum_two=minimum_two,
                    condensation_only=condensation_only,
                    parallel_sort=parallel_sort,
                )
            labels, integers, reals = kernel(
                inputs=[native_distances, core, params, options],
                grid=(batch * threads, 1, 1),
                threadgroup=(threads, 1, 1),
                output_shapes=[(batch, size), (batch, INTEGER_WORKSPACE * size), (batch, REAL_WORKSPACE * size)],
                output_dtypes=[module.int32, module.int32, module.float32],
                stream=module.gpu,
            )
    else:
        from hdbscan_gpu._cuda import _kernels, _packed_workspace_size

        params = module.asarray([size, 1, batch, minimum], dtype=module.int32)
        labels = module.empty((batch, size), dtype=module.int32)
        integers = module.empty((batch, INTEGER_WORKSPACE * size), dtype=module.int32)
        reals = module.empty((batch, REAL_WORKSPACE * size), dtype=getattr(module, backend.precision))
        _, kernel = _kernels(backend.precision)
        shared_mem = 0
        if reference_eom or shared_workspace or unit_core or register_prim or sort_depth is not None or threads != CLUSTER_THREADS or linkage_only or minimum_two or parallel_sort:
            kernel = diagnostic_kernel(
                backend, reference_eom=reference_eom, shared_workspace=shared_workspace,
                unit_core=unit_core, register_prim=register_prim, sort_depth=sort_depth, threads=threads,
                linkage_only=linkage_only,
                minimum_two=minimum_two,
                condensation_only=condensation_only,
                parallel_sort=parallel_sort,
            )
        if shared_workspace:
            shared_mem = _packed_workspace_size(backend.precision, size)
            attributes = module.cuda.Device().attributes
            capacity = attributes.get("MaxSharedMemoryPerBlockOptin", attributes["MaxSharedMemoryPerBlock"])
            if shared_mem + kernel.shared_size_bytes > capacity:
                pytest.skip("diagnostic shared workspace exceeds this GPU's per-block capacity")
            kernel.max_dynamic_shared_size_bytes = capacity - kernel.shared_size_bytes
        kernel(
            (batch,), (threads,), (native_distances, core, params, options, labels, integers, reals),
            shared_mem=shared_mem,
        )
    # Only transfer defined regions; workspace padding is intentionally unwritten.
    def fields(values, names, lengths):
        return {
            name: backend.host(values[:, names.index(name) * 2 * size:names.index(name) * 2 * size + length])
            for name, length in lengths.items()
        }

    return (
        fields(integers, INTEGER_ARRAYS, {"left": 2 * size, "right": 2 * size, "sizes": 2 * size} if linkage_only else {
            "order": size - 1, "edge_u": size - 1, "edge_v": size - 1,
            "parent": 2 * size, "selected": 2 * size, "label_map": 2 * size, "point_parent": size,
        }),
        fields(reals, REAL_ARRAYS, {"distance": 2 * size} if linkage_only else {
            "edge_w": size - 1, "birth": 2 * size, "stability": 2 * size,
        }),
        backend.host(labels),
    )


def sequential_prim(distances, core):
    size = len(core)
    visited = np.zeros(size, dtype=bool)
    best = np.full(size, np.inf, dtype=distances.dtype)
    sources = np.ones(size, dtype=np.int32)
    edges = []
    current = 0
    for _ in range(size - 1):
        visited[current] = True
        weights = np.maximum(np.maximum(core[current], core), distances[current])
        improved = (weights < best) & ~visited
        best[improved] = weights[improved]
        sources[improved] = current
        next_vertex = int(np.argmin(np.where(visited, np.inf, best)))
        edges.append((sources[next_vertex], next_vertex, best[next_vertex]))
        current = next_vertex
    return np.asarray(edges)


@pytest.mark.parametrize("size", [2, 257, 1024])
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 2.0])
@pytest.mark.parametrize("threads,register_prim", [(32, 0), (256, 0), (512, 0), (1024, 0), (128, 8), (256, 4), (512, 2), (1024, 1)])
def test_shared_workspace_matches_global_exactly(backend, size, epsilon, threads, register_prim):
    rng = np.random.default_rng(size)
    random = rng.integers(0, 12, size=(size, size)).astype(backend.precision) / 10
    random = np.minimum(random, random.T)
    groups = np.arange(size) // 2
    chain = (np.maximum(groups[:, None], groups[None, :]) + 1).astype(backend.precision)
    chain[groups[:, None] == groups[None, :]] = 0
    distances = np.stack([random, chain, np.zeros_like(chain)])
    for matrix in distances:
        np.fill_diagonal(matrix, 0)
    core = np.zeros((3, size), dtype=backend.precision)
    core[0] = rng.integers(0, 3, size=size) / 10
    expected = graph_workspace(backend, distances, core, epsilon=epsilon)
    actual = graph_workspace(
        backend, distances, core, epsilon=epsilon, shared_workspace=True, threads=threads, register_prim=register_prim,
    )
    for observed, reference in zip(actual[:2], expected[:2]):
        for name in observed:
            np.testing.assert_array_equal(observed[name], reference[name])
    np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize("size", [2, 33, 257, 1024, 4095, 4096])
def test_balanced_linkage_preserves_exact_tree(backend, size):
    rng = np.random.default_rng(size)
    random = rng.integers(0, 12, size=(size, size)).astype(backend.precision)
    random = np.minimum(random, random.T)
    groups = np.arange(size) // 2
    chain = (np.maximum(groups[:, None], groups[None, :]) + 1).astype(backend.precision)
    chain[groups[:, None] == groups[None, :]] = 0
    matrices = np.stack([random, chain, np.zeros_like(chain)])
    for matrix in matrices:
        np.fill_diagonal(matrix, 0)
    core = np.zeros((3, size), dtype=backend.precision)
    expected = graph_workspace(backend, matrices, core, reference_eom=True, linkage_only=True)
    for shared, parallel in ((False, False), (False, True), (True, True)):
        if shared and backend.name == "mlx" and size > 1024:
            continue
        packed = shared and backend.name == "mlx"
        actual = graph_workspace(
            backend, matrices, core, linkage_only=True, shared_workspace=shared,
            threads=max(32, 1 << (size - 1).bit_length()) if packed else CLUSTER_THREADS,
            register_prim=1 if packed else 0,
            parallel_sort=parallel,
        )
        for observed, reference in zip(actual[:2], expected[:2]):
            for name in observed:
                np.testing.assert_array_equal(observed[name], reference[name])


@pytest.mark.parametrize("backend", [pytest.param(("mlx", "float32"), id="metal-float32")], indirect=True)
@pytest.mark.parametrize("size", [2, 33, 513, 1023, 1024])
@pytest.mark.parametrize("unit_core", [False, True])
@pytest.mark.parametrize("minimum,minimum_two", [(2, False), (2, True), (5, False), (2**31 - 1, False)])
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 2.0])
def test_metal_split_transport_and_fresh_hierarchy(backend, size, unit_core, minimum, minimum_two, epsilon):
    from hdbscan_gpu._mlx import _split_kernels

    mx = backend.module
    rng = np.random.default_rng(size)
    random = rng.integers(0, 12, size=(size, size)).astype(np.float32) / 10
    random = np.minimum(random, random.T)
    groups = np.arange(size) // 2
    chain = (np.maximum(groups[:, None], groups[None, :]) + 1).astype(np.float32)
    chain[groups[:, None] == groups[None, :]] = 0
    matrices = np.stack([random, chain, np.zeros_like(chain)])
    for matrix in matrices:
        np.fill_diagonal(matrix, 0)
    core = np.zeros((3, size), dtype=np.float32)
    if not unit_core:
        core[0] = rng.integers(0, 3, size=size) / 10
    integers, reals, expected_labels = graph_workspace(
        backend, matrices, core, epsilon=epsilon, minimum=minimum, reference_eom=True,
    )
    order = integers["order"]
    expected_vertices = np.stack([
        np.take_along_axis(integers[name], order, axis=1) for name in ("edge_u", "edge_v")
    ], axis=-1).astype(np.int16)
    expected_weights = np.take_along_axis(reals["edge_w"], order, axis=1)
    capacity = 1 << (size - 1).bit_length()
    threads = max(32, capacity)
    edge_kernel, hierarchy_kernel = _split_kernels(capacity, unit_core, minimum_two)
    with mx.stream(mx.gpu):
        params = mx.array([size, 1, 3, minimum], dtype=mx.int32)
        options = mx.array([epsilon], dtype=mx.float32)
        vertices, weights = edge_kernel(
            inputs=[backend.array(matrices), backend.array(core), params, options],
            grid=(3 * threads, 1, 1), threadgroup=(threads, 1, 1),
            output_shapes=[(3, size - 1, 2), (3, size - 1)],
            output_dtypes=[mx.int16, mx.float32], stream=mx.gpu,
        )
        np.testing.assert_array_equal(backend.host(vertices), expected_vertices)
        np.testing.assert_array_equal(backend.host(weights).view(np.uint32), expected_weights.view(np.uint32))
        # Fresh buffers ensure the hierarchy needs no hidden state from Prim.
        actual = hierarchy_kernel(
            inputs=[mx.array(expected_vertices), mx.array(expected_weights), params, options],
            grid=(3 * 32, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(3, size)], output_dtypes=[mx.int32], stream=mx.gpu,
        )[0]
        np.testing.assert_array_equal(backend.host(actual), expected_labels)


@pytest.mark.parametrize("backend", [pytest.param(("mlx", "float32"), id="metal-float32")], indirect=True)
def test_metal_multiple_unevaluated_pipelines(backend, oracle, assert_partition):
    from hdbscan_gpu import hdbscan

    rng = np.random.default_rng(61)
    values = [rng.integers(-4, 5, size=(3, 129, 3)).astype(np.float32) / 8 for _ in range(2)]
    results = [
        hdbscan(backend.array(value), **backend.kwargs, min_cluster_size=2, cluster_selection_epsilon=0.1)
        for value in values
    ]
    backend.module.eval(*results)
    for result, value in zip(results, values):
        for actual, window in zip(backend.host(result), value):
            assert_partition(actual, oracle(window, min_cluster_size=2, cluster_selection_epsilon=0.1))


@pytest.mark.parametrize("backend", [pytest.param(("mlx", "float32"), id="metal-float32")], indirect=True)
@pytest.mark.parametrize("size", [33, 1024])
@pytest.mark.parametrize("minimum", [2, 5, 2**31 - 1])
@pytest.mark.parametrize("epsilon", [
    0.0, float(np.nextafter(np.float32(1), np.float32(0))), 1.0,
    float(np.nextafter(np.float32(1), np.float32(2))),
])
def test_metal_cooperative_hierarchy_preserves_bitwise_state(backend, size, minimum, epsilon):
    from hdbscan_gpu._mlx import _SPLIT_HIERARCHY_SOURCE

    mx = backend.module
    groups = np.arange(size) // 2
    chain = (np.maximum(groups[:, None], groups[None, :]) + 1).astype(np.float32)
    chain[groups[:, None] == groups[None, :]] = 0
    nested = np.full_like(chain, 10)
    for width, distance in ((8, 1), (4, 100 / 101), (2, 10 / 11)):
        membership = np.arange(size) // width
        nested[membership[:, None] == membership[None, :]] = distance
    ancestor_boundary = nested.copy()
    ancestor_boundary[groups[:, None] == groups[None, :]] = 0.1
    starting_boundary = np.full_like(chain, 10)
    membership = np.arange(size) // 8
    starting_boundary[membership[:, None] == membership[None, :]] = 1
    starting_boundary[groups[:, None] == groups[None, :]] = 0.1
    zero_path = np.ones_like(chain)
    adjacent = np.arange(size - 1)
    zero_path[adjacent, adjacent + 1] = zero_path[adjacent + 1, adjacent] = 0
    matrices = np.stack([
        chain, nested, np.ones_like(chain), np.zeros_like(chain), ancestor_boundary, starting_boundary, zero_path,
    ])
    for matrix in matrices:
        np.fill_diagonal(matrix, 0)
    core = np.zeros((len(matrices), size), dtype=np.float32)
    final = graph_workspace(backend, matrices, core, epsilon=epsilon, minimum=minimum, reference_eom=True)
    condensed = graph_workspace(
        backend, matrices, core, epsilon=epsilon, minimum=minimum, reference_eom=True, condensation_only=True,
    )
    if minimum == 2 and size == 1024:
        assert np.any(np.isinf(condensed[1]["stability"]))
        assert np.any(np.isnan(condensed[1]["stability"]))
        if epsilon > 0:
            ancestor_labels, starting_labels = final[2][4:6]
            ancestor_count = len(np.unique(ancestor_labels[ancestor_labels >= 0]))
            starting_count = len(np.unique(starting_labels[starting_labels >= 0]))
            assert ancestor_count == size // (4 if epsilon < 1 else 8)
            assert starting_count == size // (2 if epsilon <= 1 else 8)
    for parents in condensed[0]["parent"]:
        for parent in np.unique(parents[parents >= size]):
            children = np.flatnonzero(parents == parent)
            assert len(children) == 2 and children[1] == children[0] + 1
    order = final[0]["order"]
    vertices = np.stack([
        np.take_along_axis(final[0][name], order, axis=1) for name in ("edge_u", "edge_v")
    ], axis=-1).astype(np.int16)
    weights = np.take_along_axis(final[1]["edge_w"], order, axis=1)
    fields = {
        "parent": 2 * size, "selected": 2 * size, "label_map": 2 * size,
        "point_parent": size, "birth": 2 * size, "stability": 2 * size,
    }
    capacity = 1 << (size - 1).bit_length()
    common = (
        header("mlx", "float32")
        + "#define PARALLEL_PRIM\n#define PARALLEL_SORT\n#define METAL_PACKED_WORKSPACE\n"
        + f"#define WORKSPACE_CAPACITY {capacity}\n#define COOPERATIVE_LABELS\n"
        + ("#define MINIMUM_TWO\n" if minimum == 2 else "")
    )
    with mx.stream(mx.gpu):
        params = mx.array([size, 1, len(matrices), minimum], dtype=mx.int32)
        options = mx.array([epsilon], dtype=mx.float32)
        for binary in (False, True):
            for before_eom, expected in ((True, condensed), (False, final)):
                exported = export_fields({
                    name: length for name, length in fields.items()
                    if not (before_eom and name == "label_map")
                })
                # Continue the whole kernel so the production leader scope stays balanced.
                source = (
                    _SPLIT_HIERARCHY_SOURCE.replace(EOM_SOURCE, exported + EOM_SOURCE)
                    if before_eom else _SPLIT_HIERARCHY_SOURCE + exported
                )
                kernel = mx.fast.metal_kernel(
                    name="split_hierarchy_state", input_names=["vertices", "weights", "params", "options"],
                    output_names=["labels", "integers", "reals"],
                    source=metal_source(source), header=common + ("#define COOPERATIVE_BINARY_EOM\n" if binary else ""),
                )
                labels, integers, reals = kernel(
                    inputs=[mx.array(vertices), mx.array(weights), params, options],
                    grid=(len(matrices) * 32, 1, 1), threadgroup=(32, 1, 1),
                    output_shapes=[
                        (len(matrices), size), (len(matrices), INTEGER_WORKSPACE * size),
                        (len(matrices), REAL_WORKSPACE * size),
                    ],
                    output_dtypes=[mx.int32, mx.int32, mx.float32], stream=mx.gpu,
                )
                for name, length in fields.items():
                    if before_eom and name == "label_map":
                        continue
                    real = name in REAL_ARRAYS
                    names = REAL_ARRAYS if real else INTEGER_ARRAYS
                    start = names.index(name) * 2 * size
                    actual = backend.host((reals if real else integers)[:, start:start + length])
                    reference = expected[int(real)][name]
                    if real:
                        actual, reference = actual.view(np.uint32), reference.view(np.uint32)
                    np.testing.assert_array_equal(actual, reference, err_msg=name)
                np.testing.assert_array_equal(backend.host(labels), final[2])


@pytest.mark.parametrize("size", [1023, 1024, 1025, 2048, 4096])
@pytest.mark.parametrize("minimum", [2, 2**31 - 1])
def test_storage_boundary_and_large_minimum(backend, reference, assert_partition, size, minimum):
    from hdbscan_gpu import hdbscan

    values = np.repeat(np.arange((size + 1) // 2, dtype=backend.precision), 2)[:size, None]
    parameters = {"min_cluster_size": minimum, "min_samples": 2}
    actual = hdbscan(backend.array(values), **backend.kwargs, **parameters)
    assert_partition(backend.host(actual), reference(values, **parameters))


def test_cuda_runtime_shared_capacity_boundary(backend, reference, assert_partition, monkeypatch):
    if backend.name != "cuda":
        pytest.skip("CUDA dynamic shared-memory capacity")
    from hdbscan_gpu import hdbscan
    from hdbscan_gpu import _cuda

    shared_kernel = _cuda._shared_kernel
    kernel, capacity = shared_kernel(
        backend.precision, backend.module.cuda.Device().id, True, False, 0, CLUSTER_THREADS, False,
    )
    bytes_per_point = _cuda._packed_workspace_size(backend.precision, 1)
    maximum = min(4096, capacity // bytes_per_point)
    assert maximum >= 3
    forced_boundary = min(257, maximum - 1)
    limits = [(maximum, capacity), (forced_boundary, forced_boundary * bytes_per_point)]
    for boundary, limit in limits:
        # Lower only the dispatch threshold, not the real kernel's configured capacity.
        with monkeypatch.context() as patch:
            def limited_kernel(*args):
                configured, available = shared_kernel(*args)
                return configured, min(available, limit)

            patch.setattr(_cuda, "_shared_kernel", limited_kernel)
            for size in (boundary, boundary + 1):
                if size > 4096:
                    continue
                _, shared_bytes = _cuda._cluster_kernel(backend.precision, size, unit_core=True)
                assert bool(shared_bytes) == (size <= boundary)
                values = np.repeat(np.arange((size + 1) // 2, dtype=backend.precision), 2)[:size, None]
                parameters = {"min_cluster_size": 2, "min_samples": 2}
                actual = hdbscan(backend.array(values), **backend.kwargs, **parameters)
                assert_partition(backend.host(actual), reference(values, **parameters))
            # Finite stability and nontrivial minimum sizes expose corrupted leaf sizes.
            values = np.random.default_rng(17).normal(size=(boundary, 3)).astype(backend.precision)
            with monkeypatch.context() as forced:
                forced.setattr(_cuda, "_shared_kernel", lambda *args: (shared_kernel(*args)[0], 0))
                parameters = {"min_cluster_size": 5, "min_samples": 3}
                actual = hdbscan(backend.array(values), **backend.kwargs, **parameters)
                assert_partition(backend.host(actual), reference(values, **parameters))


def test_cuda_cached_shared_kernel_retains_optin(backend, reference, assert_partition):
    if backend.name != "cuda":
        pytest.skip("CUDA compilation-cache configuration")
    from hdbscan_gpu import _cuda

    kernel, capacity = _cuda._shared_kernel(
        backend.precision, backend.module.cuda.Device().id, True, True, 0, CLUSTER_THREADS, False,
    )
    _cuda._kernels.cache_clear()
    selected, required = _cuda._cluster_kernel(backend.precision, 33, unit_core=True, minimum_two=True)
    assert selected is kernel
    assert required <= kernel.max_dynamic_shared_size_bytes == capacity
    size = 2048
    if _cuda._packed_workspace_size(backend.precision, size) > capacity:
        pytest.skip("configured arena exceeds this GPU's shared-memory capacity")
    values = np.repeat(np.arange(size // 2, dtype=backend.precision), 2)[:, None]
    actual = _cuda.cluster(backend.array(values), 2, 2, 0, backend.precision)
    assert_partition(backend.host(actual), reference(values, min_cluster_size=2, min_samples=2))


def test_distance_overflow_in_partial_simd_group(backend):
    from hdbscan_gpu import hdbscan

    values = np.zeros((3, 33, 1), dtype=backend.precision)
    values[-1, -1, 0] = np.finfo(backend.precision).max
    with pytest.raises(ValueError, match="overflow"):
        hdbscan(backend.array(values), **backend.kwargs, min_cluster_size=2)
    values[0, 0, 0] = np.finfo(backend.precision).max
    values[-1, -1, 0] = np.nan
    # Only the final partial SIMD group sees this diagonal input error. It must
    # outrank earlier overflow flags rather than being lost in the reduction.
    with pytest.raises(ValueError, match="only finite"):
        hdbscan(backend.array(values), **backend.kwargs, min_cluster_size=2)


@pytest.mark.parametrize("size", [33, 129, 1024])
@pytest.mark.parametrize("depth", [-1, 0, 1])
def test_parallel_sort_preserves_depth_exhaustion_order(backend, size, depth):
    weights = np.ones(size - 1, dtype=np.float32)
    distances = np.full((1, size, size), size * 10, dtype=np.float32)
    np.fill_diagonal(distances[0], 0)
    for i, weight in enumerate(weights):
        distances[0, i, i + 1] = distances[0, i + 1, i] = weight
    core = np.zeros((1, size), dtype=np.float32)
    expected = graph_workspace(backend, distances, core, sort_depth=depth)
    actual = graph_workspace(
        backend, distances, core, sort_depth=depth, shared_workspace=True,
        register_prim=1, threads=max(32, 1 << (size - 1).bit_length()), parallel_sort=True,
    )
    for observed, reference in zip(actual[:2], expected[:2]):
        for name in observed:
            np.testing.assert_array_equal(observed[name], reference[name])
    np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize("samples", [1, 2])
def test_unit_core_shortcut_preserves_ordered_graph(backend, samples):
    rng = np.random.default_rng(517)
    values = rng.integers(-4, 5, size=(4, 65, 3)).astype(backend.precision)
    delta = values[:, :, None] - values[:, None, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    actual_core = np.partition(distances, samples - 1, axis=-1)[..., samples - 1]
    expected = graph_workspace(backend, distances, actual_core, epsilon=0.1)
    actual = graph_workspace(backend, distances, actual_core, epsilon=0.1, unit_core=True)
    for observed, reference in zip(actual[:2], expected[:2]):
        for name in observed:
            np.testing.assert_array_equal(observed[name], reference[name])
    np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize("size", [33, 1024])
@pytest.mark.parametrize("unit_core", [False, True])
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 2.0])
def test_minimum_two_preserves_exact_hierarchy(backend, size, unit_core, epsilon):
    rng = np.random.default_rng(size)
    random = rng.integers(0, 12, size=(size, size)).astype(backend.precision) / 10
    random = np.minimum(random, random.T)
    groups = np.arange(size) // 2
    chain = (np.maximum(groups[:, None], groups[None, :]) + 1).astype(backend.precision)
    chain[groups[:, None] == groups[None, :]] = 0
    matrices = np.stack([random, chain, np.zeros_like(chain)])
    for matrix in matrices:
        np.fill_diagonal(matrix, 0)
    core = np.zeros((3, size), dtype=backend.precision)
    if not unit_core:
        core[0] = rng.integers(0, 3, size=size) / 10
    expected = graph_workspace(backend, matrices, core, epsilon=epsilon, reference_eom=True)
    actual = graph_workspace(
        backend, matrices, core, epsilon=epsilon, shared_workspace=True,
        register_prim=4, minimum_two=True, unit_core=unit_core,
    )
    for name in actual[0]:
        np.testing.assert_array_equal(actual[0][name], expected[0][name])
    bits = np.uint32 if backend.precision == "float32" else np.uint64
    for name in actual[1]:
        np.testing.assert_array_equal(actual[1][name].view(bits), expected[1][name].view(bits))
    np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize("size", [2, 255, 256, 257, 513])
@pytest.mark.parametrize("threads", [32, 64, 128, 256, 512, 1024])
def test_ordered_prim_edges_preserve_sources_and_global_ties(backend, size, threads):
    rng = np.random.default_rng(size)
    equal = np.ones((size, size), dtype=backend.precision)
    retained_source = np.full((size, size), 9, dtype=backend.precision)
    if size >= 4:
        for a, b, value in [(0, 1, 1), (0, 2, 2), (1, 3, 3), (2, 3, 3)]:
            retained_source[a, b] = retained_source[b, a] = value
    quantized = rng.integers(1, 7, size=(size, size)).astype(backend.precision)
    quantized = np.minimum(quantized, quantized.T)
    matrices = np.stack([equal, retained_source, quantized])
    for matrix in matrices:
        np.fill_diagonal(matrix, 0)
    core = np.zeros((3, size), dtype=backend.precision)
    core[2] = rng.integers(0, 4, size=size)
    integers, reals, _ = graph_workspace(backend, matrices, core, threads=threads)
    for index, matrix in enumerate(matrices):
        expected = sequential_prim(matrix, core[index])
        for column, name in enumerate(("edge_u", "edge_v")):
            np.testing.assert_array_equal(integers[name][index], expected[:, column])
        np.testing.assert_array_equal(reals["edge_w"][index], expected[:, 2])


@pytest.mark.parametrize("size", [32, 128])
@pytest.mark.parametrize("within_pair", [0.0, 0.25])
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 2.0])
@pytest.mark.parametrize("minimum", [2, 5, 2**31 - 1])
def test_eom_adjacency_matches_exhaustive_device_oracle(backend, size, within_pair, epsilon, minimum):
    groups = np.arange(size) // 2
    chain = (np.maximum(groups[:, None], groups[None, :]) + 1).astype(backend.precision)
    chain[groups[:, None] == groups[None, :]] = within_pair
    equal = np.ones_like(chain)
    nested = np.full_like(chain, 10)
    for width, distance in ((8, 1), (4, 100 / 101), (2, 10 / 11)):
        membership = np.arange(size) // width
        nested[membership[:, None] == membership[None, :]] = distance
    matrices = np.stack([chain, equal, nested])
    for matrix in matrices:
        np.fill_diagonal(matrix, 0)
    core = np.zeros((3, size), dtype=backend.precision)
    actual = graph_workspace(backend, matrices, core, epsilon=epsilon, minimum=minimum)
    expected = graph_workspace(backend, matrices, core, reference_eom=True, epsilon=epsilon, minimum=minimum)
    for name in ("parent", "selected", "label_map", "point_parent"):
        np.testing.assert_array_equal(actual[0][name], expected[0][name])
    for name in ("birth", "stability"):
        np.testing.assert_array_equal(actual[1][name], expected[1][name])
    np.testing.assert_array_equal(actual[2], expected[2])
    if epsilon == 0 and minimum == 2:
        parent = expected[0]["parent"][2]
        selected = expected[0]["selected"][2]
        point_parent = expected[0]["point_parent"][2]
        birth = expected[1]["birth"][2]
        stability = expected[1]["stability"][2]
        relationships = 0
        for ancestor in np.flatnonzero(selected == 1):
            for middle in np.flatnonzero(parent == ancestor):
                leaves = np.flatnonzero(parent == middle)
                if not len(leaves) or any(np.any(parent == leaf) for leaf in leaves):
                    continue
                original_stability = sum(
                    (birth[leaf] - birth[middle]) * np.count_nonzero(point_parent == leaf)
                    for leaf in leaves
                )
                # The intermediate loses, but its terminal children initially win.
                assert np.sum(stability[leaves]) > original_stability
                assert np.all(stability[leaves] > 0)
                assert selected[middle] == 0 and np.all(selected[leaves] == 0)
                relationships += 1
        assert relationships > 0, "fixture must exercise suppression through a losing intermediate"
