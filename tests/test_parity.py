from itertools import combinations_with_replacement

import numpy as np
import pytest

from hdbscan_gpu import hdbscan


def curated_corpus():
    rng = np.random.default_rng(20261010)
    separated = np.concatenate([
        rng.normal(center, 0.08, (24, 2))
        for center in ((-3, -3), (0, 3), (3, -3))
    ])
    overlap = np.concatenate([
        rng.normal(center, 0.45, (30, 2))
        for center in ((-0.5, 0), (0.5, 0), (0, 0.8))
    ])
    varied = np.concatenate([
        rng.normal(center, scale, (count, 2))
        for center, scale, count in (
            ((-4, 0), 0.025, 30),
            ((0, 4), 0.3, 35),
            ((4, 0), 0.8, 40),
        )
    ])
    grid = np.array([(x, y) for x in range(5) for y in range(5)])
    tiny = np.concatenate([
        rng.normal(center, 0.01, (count, 2))
        for center, count in (((0, 0), 2), ((2, 2), 3), ((5, 0), 9))
    ])
    return {
        "well-separated": separated.astype(np.float32),
        "overlap": overlap.astype(np.float32),
        "density-variation": varied.astype(np.float32),
        "random-quantized": rng.integers(-12, 13, (48, 3)).astype(np.float32) / 8,
        "duplicate-heavy": np.repeat(
            np.array([[0, 0], [0.125, 0], [4, 0], [4.125, 0], [20, 20]], np.float32),
            [7, 5, 6, 8, 1],
            axis=0,
        ),
        "all-identical": np.ones((24, 3), np.float32),
        "symmetric-grid-ties": grid.astype(np.float32),
        "symmetric-cross-ties": np.tile(
            np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], np.float32),
            (4, 1),
        ),
        "tiny-clusters": tiny.astype(np.float32),
        "isolated-outliers": np.concatenate([
            separated, np.array([[20, 20], [-20, 20], [0, -30]], np.float32),
        ]),
        "two-samples": np.array([[0, 0], [1, 1]], np.float32),
    }


CORPUS = curated_corpus()
PARAMETERS = [
    pytest.param(2, 1, id="mcs2-ms1"),
    pytest.param(2, 2, id="mcs2-ms2"),
    pytest.param(5, None, id="mcs5-default-ms"),
    pytest.param(7, 3, id="mcs7-ms3"),
]
BOUNDARY = np.array(
    [[0], [0], [0.125], [0.125], [4], [4], [4.125], [4.125]],
    dtype=np.float32,
)
EPSILON_CASES = [
    pytest.param(0.0, 4, id="zero"),
    pytest.param(0.1, 4, id="point-one"),
    pytest.param(float(np.nextafter(np.float32(0.125), np.float32(0))), 4, id="below"),
    pytest.param(0.125, 4, id="exact"),
    pytest.param(float(np.nextafter(np.float32(0.125), np.float32(1))), 2, id="above"),
]


def compare(backend, oracle, assert_partition, values, **parameters):
    expected = oracle(values, **parameters)
    result = hdbscan(values, **backend.kwargs, **parameters)
    backend.assert_device_labels(result, values.shape[:-1])
    assert_partition(backend.host(result), expected)
    return expected


@pytest.mark.parametrize("epsilon,clusters", EPSILON_CASES)
def test_cpu_oracle_boundary_smoke(oracle, assert_partition, epsilon, clusters):
    labels = oracle(
        BOUNDARY, min_cluster_size=2, min_samples=1,
        cluster_selection_epsilon=epsilon,
    )
    expected = np.repeat(np.arange(clusters), len(BOUNDARY) // clusters)
    assert_partition(labels, expected)


def test_cpu_oracle_noise_and_clusters_smoke(oracle, assert_partition):
    assert_partition(oracle(np.zeros((8, 2)), min_cluster_size=2), np.full(8, -1))
    labels = oracle(CORPUS["well-separated"], min_cluster_size=5)
    assert_partition(labels, np.repeat(np.arange(3), 24))


def test_partition_comparison_rejects_noise_merges_and_splits(assert_partition):
    assert_partition([7, 7, 4, 4, -1], [0, 0, 1, 1, -1])
    for bad in ([0, 0, 1, 1, 1], [0, 0, 0, 0, -1], [0, 2, 1, 1, -1]):
        with pytest.raises(AssertionError):
            assert_partition(bad, [0, 0, 1, 1, -1])


@pytest.mark.parametrize("name", CORPUS)
@pytest.mark.parametrize("minimum,samples", PARAMETERS)
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_curated_parity(backend, oracle, assert_partition, name, minimum, samples, epsilon):
    values = CORPUS[name]
    # The two-point corpus has its own valid parameter combinations rather
    # than skipping combinations with min_samples > N.
    if (samples if samples is not None else minimum) > len(values):
        samples = len(values)
    compare(
        backend, oracle, assert_partition, values,
        min_cluster_size=minimum, min_samples=samples,
        cluster_selection_epsilon=epsilon,
    )


@pytest.mark.parametrize("epsilon,clusters", EPSILON_CASES)
def test_epsilon_boundary(backend, oracle, assert_partition, epsilon, clusters):
    labels = compare(
        backend, oracle, assert_partition, BOUNDARY,
        min_cluster_size=2, min_samples=1,
        cluster_selection_epsilon=epsilon,
    )
    assert len(set(labels)) == clusters


@pytest.mark.parametrize("minimum,samples", [(2, 8), (8, 1), (9, 2)])
def test_all_noise(backend, oracle, assert_partition, minimum, samples):
    values = np.arange(16, dtype=np.float32).reshape(8, 2)
    expected = compare(
        backend, oracle, assert_partition, values,
        min_cluster_size=minimum, min_samples=samples,
    )
    np.testing.assert_array_equal(expected, -np.ones(8, dtype=np.int32))


# Every multiset of size four or five from the four square vertices: 91
# configurations, including repeated points, ties, and degenerate clusters.
SMALL_CONFIGURATIONS = [
    pytest.param(indices, id="".join(map(str, indices)))
    for size in (4, 5)
    for indices in combinations_with_replacement(range(4), size)
]
GRID = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)


@pytest.mark.parametrize("indices", SMALL_CONFIGURATIONS)
@pytest.mark.parametrize("samples", [1, 2])
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_exhaustive_small_domain(backend, oracle, assert_partition, indices, samples, epsilon):
    compare(
        backend, oracle, assert_partition, GRID[list(indices)],
        min_cluster_size=2, min_samples=samples,
        cluster_selection_epsilon=epsilon,
    )


@pytest.mark.parametrize("indices", [(0, 0, 1, 2, 3), (0, 1, 1, 2, 2), (0, 0, 2, 3, 3)])
@pytest.mark.parametrize("order", [(4, 3, 2, 1, 0), (2, 4, 0, 3, 1), (1, 0, 4, 2, 3)])
def test_small_domain_permutations(backend, oracle, assert_partition, indices, order):
    # Tied MSTs can legitimately depend on input order. Compare each order
    # with its own CPU oracle, not with an assumed permutation invariance.
    compare(
        backend, oracle, assert_partition, GRID[list(indices)][list(order)],
        min_cluster_size=2, min_samples=1,
    )


@pytest.mark.parametrize("seed", range(36))
def test_seeded_randomized_sweep(backend, oracle, assert_partition, seed):
    rng = np.random.default_rng(seed + 1701)
    count = int(rng.integers(8, 81))
    dimensions = int(rng.integers(1, 9))
    if seed % 3 == 0:
        values = rng.integers(-8, 9, (count, dimensions)).astype(np.float32) / 4
    elif seed % 3 == 1:
        values = rng.normal(size=(count, dimensions)).astype(np.float32)
    else:
        centers = rng.normal(0, 4, (3, dimensions))
        values = (
            centers[np.arange(count) % 3] + rng.normal(0, 0.1, (count, dimensions))
        ).astype(np.float32)
    minimum = int(rng.integers(2, min(10, count) + 1))
    samples = None if seed % 4 == 0 else int(rng.integers(1, min(8, count) + 1))
    compare(
        backend, oracle, assert_partition, values,
        min_cluster_size=minimum, min_samples=samples,
        cluster_selection_epsilon=0.1 if seed % 2 else 0.0,
    )


@pytest.mark.parametrize("minimum,samples,epsilon", [(5, None, 0.0), (12, 6, 0.1)])
def test_real_shape_1024_by_8(backend, oracle, assert_partition, minimum, samples, epsilon):
    rng = np.random.default_rng(102408)
    centers = rng.normal(0, 5, (8, 8))
    values = (centers[np.arange(1024) % 8] + rng.normal(0, 0.2, (1024, 8))).astype(np.float32)
    compare(
        backend, oracle, assert_partition, values,
        min_cluster_size=minimum, min_samples=samples,
        cluster_selection_epsilon=epsilon,
    )


@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_batch_individual_and_position_invariance(backend, oracle, assert_partition, epsilon):
    rng = np.random.default_rng(31415)
    windows = np.stack([
        rng.normal(size=(32, 3)),
        rng.integers(-4, 5, (32, 3)) / 2,
        np.zeros((32, 3)),
    ]).astype(np.float32)
    parameters = dict(min_cluster_size=3, min_samples=2, cluster_selection_epsilon=epsilon)
    individual = []
    for window in windows:
        result = hdbscan(window, **backend.kwargs, **parameters)
        backend.assert_device_labels(result, (32,))
        labels = backend.host(result)
        assert_partition(labels, oracle(window, **parameters))
        individual.append(labels)
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 0, 2]):
        result = hdbscan(windows[order], **backend.kwargs, **parameters)
        backend.assert_device_labels(result, (3, 32))
        for position, labels in enumerate(backend.host(result)):
            assert_partition(labels, individual[order[position]])
    singleton = hdbscan(windows[:1], **backend.kwargs, **parameters)
    backend.assert_device_labels(singleton, (1, 32))
    assert_partition(backend.host(singleton)[0], individual[0])


def test_default_min_samples(backend, oracle, assert_partition):
    values = CORPUS["well-separated"]
    implicit = hdbscan(values, **backend.kwargs, min_cluster_size=4)
    explicit = hdbscan(values, **backend.kwargs, min_cluster_size=4, min_samples=4)
    assert_partition(backend.host(implicit), backend.host(explicit))
    assert_partition(backend.host(implicit), oracle(values, min_cluster_size=4))


def test_documented_float32_epsilon_resolution_limit(backend, oracle, assert_partition):
    # This is an explicit representational limitation, not an xfail or a
    # tolerance around arbitrary mismatches. Coordinates are exactly float32.
    # The next float64 epsilon above 1/8 merges the four duplicate pairs,
    # but float32 cannot distinguish it from the exact (unmerged) boundary.
    epsilon = float(np.nextafter(np.float64(0.125), np.float64(1)))
    assert epsilon > 0.125
    assert np.float32(epsilon) == np.float32(0.125)
    parameters = dict(min_cluster_size=2, min_samples=1)
    full_precision = oracle(BOUNDARY, cluster_selection_epsilon=epsilon, **parameters)
    rounded = oracle(BOUNDARY, cluster_selection_epsilon=0.125, **parameters)
    assert_partition(full_precision, [0, 0, 0, 0, 1, 1, 1, 1])
    assert_partition(rounded, [0, 0, 1, 1, 2, 2, 3, 3])
    actual = hdbscan(
        BOUNDARY, **backend.kwargs, cluster_selection_epsilon=epsilon, **parameters,
    )
    backend.assert_device_labels(actual, (8,))
    assert_partition(
        backend.host(actual), rounded if backend.precision == "float32" else full_precision,
    )
