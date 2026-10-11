import numpy as np
import pytest

from hdbscan_gpu import hdbscan


GAP = 2.0**-30
COLLAPSING_POINTS = np.repeat(
    1 + np.array([0, GAP, 16 * GAP, 17 * GAP], dtype=np.float64), 2,
)[:, None]
BOUNDARY_POINTS = np.repeat(
    1 + 2.0**-40 + np.array([0, 0.125, 4, 4.125], dtype=np.float64), 2,
)[:, None]
BOUNDARIES = [
    pytest.param(float(np.nextafter(0.125, 0)), 4, id="below"),
    pytest.param(0.125, 4, id="exact"),
    pytest.param(float(np.nextafter(0.125, 1)), 2, id="above"),
]
PARAMETERS = {"min_cluster_size": 2, "min_samples": 1}


def test_float64_oracle_retains_collapsing_coordinates(oracle_float64, assert_partition):
    assert COLLAPSING_POINTS.dtype == np.float64
    assert len(np.unique(COLLAPSING_POINTS)) == 4
    collapsed = COLLAPSING_POINTS.astype(np.float32).astype(np.float64)
    assert len(np.unique(collapsed)) == 1
    assert_partition(
        oracle_float64(COLLAPSING_POINTS, **PARAMETERS),
        np.repeat(np.arange(4), 2),
    )
    assert_partition(oracle_float64(collapsed, **PARAMETERS), np.full(8, -1))


@pytest.mark.parametrize("epsilon,clusters", BOUNDARIES)
def test_float64_oracle_boundary(oracle_float64, assert_partition, epsilon, clusters):
    assert np.all(BOUNDARY_POINTS != BOUNDARY_POINTS.astype(np.float32).astype(np.float64))
    assert np.float32(epsilon) == np.float32(0.125)
    expected = oracle_float64(
        BOUNDARY_POINTS, cluster_selection_epsilon=epsilon, **PARAMETERS,
    )
    assert_partition(expected, np.repeat(np.arange(clusters), 8 // clusters))


@pytest.mark.parametrize("backend", [("cuda", "float64")], indirect=True, ids=["cuda-float64"])
def test_cuda_float64_preserves_coordinates(backend, oracle_float64, assert_partition):
    expected = oracle_float64(COLLAPSING_POINTS, **PARAMETERS)
    assert_partition(expected, np.repeat(np.arange(4), 2))
    for values in (COLLAPSING_POINTS, backend.array(COLLAPSING_POINTS)):
        # Either host conversion or the native kernel accidentally using
        # float32 would collapse every coordinate to 1 and return all noise.
        labels = hdbscan(values, **backend.kwargs, **PARAMETERS)
        backend.assert_device_labels(labels, (8,))
        assert_partition(backend.host(labels), expected)


@pytest.mark.parametrize("backend", [("cuda", "float64")], indirect=True, ids=["cuda-float64"])
def test_cuda_float64_epsilon_boundary(backend, oracle_float64, assert_partition):
    windows = np.stack([BOUNDARY_POINTS, BOUNDARY_POINTS[::-1]])
    native = backend.array(windows)
    # Coordinates include a 2**-40 offset, so this is not the float32 corpus
    # promoted to float64. The adjacent float64 epsilons also round to the
    # same float32 value, making inadvertent epsilon downcasts observable.
    for epsilon in (float(np.nextafter(0.125, 0)), 0.125, float(np.nextafter(0.125, 1))):
        labels = hdbscan(
            native, **backend.kwargs, cluster_selection_epsilon=epsilon, **PARAMETERS,
        )
        backend.assert_device_labels(labels, (2, 8))
        for actual, window in zip(backend.host(labels), windows, strict=True):
            expected = oracle_float64(
                window, cluster_selection_epsilon=epsilon, **PARAMETERS,
            )
            assert_partition(actual, expected)
