from dataclasses import dataclass
import importlib
import importlib.util

import numpy as np
import pytest


@dataclass(frozen=True)
class DeviceBackend:
    name: str
    precision: str
    module: object

    @property
    def kwargs(self):
        return {"backend": self.name, "precision": self.precision}

    def array(self, values):
        dtype = getattr(self.module, self.precision)
        if self.name == "mlx":
            with self.module.stream(self.module.gpu):
                return self.module.array(values, dtype=dtype)
        return self.module.asarray(values, dtype=dtype)

    def host(self, values):
        if self.name == "mlx":
            self.module.eval(values)
            return np.array(values)
        return self.module.asnumpy(values)

    def assert_device_labels(self, labels, shape):
        if self.name == "mlx":
            assert isinstance(labels, self.module.array)
            self.module.eval(labels)
            assert labels.__dlpack_device__()[0] == 8  # DLPack kDLMetal.
        else:
            assert isinstance(labels, self.module.ndarray)
            assert labels.device.id == self.module.cuda.Device().id
            assert labels.__dlpack_device__()[0] == 2  # DLPack kDLCUDA.
        assert labels.dtype == self.module.int32
        assert labels.shape == shape
        assert np.all(self.host(labels) >= -1)


@pytest.fixture(scope="session", params=[
    pytest.param(("mlx", "float32"), id="metal-float32"),
    pytest.param(("cuda", "float32"), id="cuda-float32"),
    pytest.param(("cuda", "float64"), id="cuda-float64"),
])
def backend(request):
    name, precision = request.param
    dependency = "mlx" if name == "mlx" else "cupy"
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{name}/{precision}: optional dependency {dependency!r} is not installed")
    module = importlib.import_module("mlx.core" if name == "mlx" else "cupy")
    if name == "mlx":
        if not module.metal.is_available():
            pytest.skip("mlx/float32: MLX is installed but no Metal GPU is available")
    else:
        try:
            count = module.cuda.runtime.getDeviceCount()
        except module.cuda.runtime.CUDARuntimeError as error:
            # Initialization/driver absence is a hardware prerequisite failure.
            # Other runtime failures, including kernel failures, must fail tests.
            if error.status not in (3, 35, 100, 803, 804):
                raise
            pytest.skip(f"cuda/{precision}: CUDA device initialization unavailable: {error}")
        if count == 0:
            pytest.skip(f"cuda/{precision}: CuPy is installed but no CUDA GPU is available")
    return DeviceBackend(name, precision, module)


@pytest.fixture(scope="session")
def reference():
    import sklearn
    from sklearn.cluster import HDBSCAN

    assert sklearn.__version__ == "1.9.1", "Parity requires scikit-learn==1.9.1"
    assert np.__version__ == "2.3.5", "Tie ordering requires numpy==2.3.5"

    def fit(values, **parameters):
        return HDBSCAN(
            algorithm="auto",
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=False,
            alpha=1.0,
            copy=True,
            **parameters,
        ).fit_predict(values)

    return fit


@pytest.fixture(scope="session")
def oracle(reference):
    def fit(values, **parameters):
        # Shared Metal/CUDA parity uses identical float32-quantized inputs;
        # sklearn then performs its own float64 computations.
        return reference(np.asarray(values, dtype=np.float32), **parameters)

    return fit


@pytest.fixture(scope="session")
def oracle_float64(reference):
    def fit(values, **parameters):
        values = np.asarray(values)
        assert values.dtype == np.float64, "Strict float64 parity must not quantize inputs"
        return reference(values, **parameters)

    return fit


@pytest.fixture
def assert_partition():
    def check(actual, expected):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual == -1, expected == -1)
        # A bijection catches both cluster merges and splits without assuming
        # that independent implementations number their clusters identically.
        forward = {}
        reverse = {}
        for observed, reference in zip(actual.flat, expected.flat):
            if reference == -1:
                continue
            assert observed >= 0
            assert forward.setdefault(int(observed), int(reference)) == reference
            assert reverse.setdefault(int(reference), int(observed)) == observed

    return check
