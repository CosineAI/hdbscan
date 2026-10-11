import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hdbscan_gpu import hdbscan


VALUES = np.array([[0], [0.01], [0.02], [3], [3.01], [3.02]], np.float32)


@pytest.mark.parametrize("parameters,match", [
    ({"backend": "cpu"}, "backend"),
    ({"backend": "auto"}, "backend"),
    ({"backend": "CUDA"}, "backend"),
    ({"backend": None}, "backend"),
    ({"precision": "float16"}, "precision"),
    ({"precision": np.float32}, "precision"),
    ({"precision": "float64", "backend": "mlx"}, "Metal"),
    ({"min_cluster_size": 1}, "min_cluster_size"),
    ({"min_cluster_size": 2.0}, "min_cluster_size"),
    ({"min_cluster_size": True}, "min_cluster_size"),
    ({"min_cluster_size": np.bool_(True)}, "min_cluster_size"),
    ({"min_cluster_size": None}, "min_cluster_size"),
    ({"min_cluster_size": 2**31, "min_samples": 1}, "int32"),
    ({"min_samples": 0}, "min_samples"),
    ({"min_samples": -1}, "min_samples"),
    ({"min_samples": 2.0}, "min_samples"),
    ({"min_samples": True}, "min_samples"),
    ({"min_samples": np.bool_(True)}, "min_samples"),
    ({"min_samples": 7}, "min_samples"),
    ({"cluster_selection_epsilon": -0.1}, "epsilon"),
    ({"cluster_selection_epsilon": float("nan")}, "epsilon"),
    ({"cluster_selection_epsilon": float("inf")}, "epsilon"),
    ({"cluster_selection_epsilon": True}, "epsilon"),
    ({"cluster_selection_epsilon": "0.1"}, "epsilon"),
    ({"cluster_selection_epsilon": 1e40}, "representable"),
])
def test_parameter_validation_without_device_dispatch(monkeypatch, parameters, match):
    import builtins

    original_import = builtins.__import__

    def no_backend_import(name, *args, **kwargs):
        if name in ("_mlx", "_cuda", "mlx", "cupy"):
            raise AssertionError(f"Invalid parameters dispatched to {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_backend_import)
    with pytest.raises(ValueError, match=match):
        hdbscan(VALUES, **parameters)


@pytest.mark.parametrize("values,match", [
    (np.zeros(6), "rank"),
    (np.array(0), "rank"),
    (np.zeros((1, 2, 3, 4)), "rank"),
    (np.zeros((0, 2)), "at least 2"),
    (np.zeros((1, 2)), "at least 2"),
    (np.zeros((2, 0)), "nonempty"),
    (np.zeros((0, 6, 2)), "nonempty"),
    (np.zeros((6, 2), dtype=bool), "real numeric"),
    (np.ones((6, 2), dtype=complex), "real numeric"),
    (np.full((6, 2), "1"), "real numeric"),
    (np.full((6, 2), object(), dtype=object), "real numeric"),
    (np.full((6, 2), np.nan), "finite"),
    (np.full((6, 2), np.inf), "finite"),
    (np.full((6, 2), -np.inf), "finite"),
    (np.zeros((4097, 1), np.float32), "4096"),
    (np.broadcast_to(np.float32(0), (5, 4096, 1)), "512 MiB"),
])
def test_host_input_validation(values, match):
    with pytest.raises(ValueError, match=match):
        hdbscan(values, min_cluster_size=2, min_samples=1)


def test_cuda_float64_workspace_limit_before_import():
    values = np.broadcast_to(np.float32(0), (3, 4096, 1))
    with pytest.raises(ValueError, match="512 MiB"):
        hdbscan(values, backend="cuda", precision="float64")


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.uint32])
def test_host_numeric_dtypes_and_no_mutation(backend, oracle, assert_partition, dtype):
    values = (VALUES * 100).astype(dtype)
    before = values.copy()
    values.flags.writeable = False
    labels = hdbscan(values, **backend.kwargs, min_cluster_size=2, min_samples=1)
    backend.assert_device_labels(labels, (6,))
    assert_partition(backend.host(labels), oracle(values, min_cluster_size=2, min_samples=1))
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("layout", ["reversed", "column-strided", "fortran", "list"])
def test_host_strides_lists_and_numpy_scalar_parameters(backend, oracle, assert_partition, layout):
    values = np.concatenate([VALUES, VALUES + 1, VALUES + 2, VALUES + 3], axis=1)
    if layout == "reversed":
        values = values[::-1, ::-1]
    elif layout == "column-strided":
        values = values[:, ::2]
    elif layout == "fortran":
        values = np.asfortranarray(values)
    expected = oracle(values, min_cluster_size=2, min_samples=1)
    result = hdbscan(
        values.tolist() if layout == "list" else values,
        **backend.kwargs, min_cluster_size=np.int64(2),
        min_samples=np.int32(1), cluster_selection_epsilon=np.float64(0),
    )
    backend.assert_device_labels(result, (6,))
    assert_partition(backend.host(result), expected)


def test_native_strided_input_and_no_mutation(backend, oracle, assert_partition):
    host = np.arange(48, dtype=np.float32).reshape(8, 6)
    native = backend.array(host)
    # Both column strides and a transpose are relevant to Metal's contiguous
    # kernel-input handling. Batch slicing also exercises a nonzero offset.
    views = [
        native[:, ::2], native[::-1, ::-2], native.T,
        backend.array(np.stack([host, host + 100]))[1:],
    ]
    for view in views:
        before = backend.host(view).copy()
        labels = hdbscan(view, **backend.kwargs, min_cluster_size=2, min_samples=1)
        backend.assert_device_labels(labels, before.shape[:-1])
        after = backend.host(view)
        np.testing.assert_array_equal(after, before)
        if before.ndim == 2:
            assert_partition(backend.host(labels), oracle(before, min_cluster_size=2, min_samples=1))
        else:
            assert_partition(
                backend.host(labels)[0], oracle(before[0], min_cluster_size=2, min_samples=1),
            )
    np.testing.assert_array_equal(backend.host(native), host)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_native_nonfinite_rejected(backend, value):
    values = backend.array(np.full((6, 2), value))
    with pytest.raises(ValueError, match="finite"):
        hdbscan(values, **backend.kwargs)


@pytest.mark.parametrize("dtype_name", ["bool_", "complex64"])
def test_native_nonnumeric_rejected(backend, dtype_name):
    values = backend.array(VALUES).astype(getattr(backend.module, dtype_name))
    with pytest.raises(ValueError, match="real numeric"):
        hdbscan(values, **backend.kwargs)


def test_native_backend_mismatch_rejected_without_fallback(backend):
    values = backend.array(VALUES)
    other = "cuda" if backend.name == "mlx" else "mlx"
    with pytest.raises(ValueError, match="match the selected backend"):
        hdbscan(values, backend=other)


def test_overflow_rejected(backend):
    largest = np.finfo(backend.precision).max / 4
    values = np.array([[-largest], [-largest], [largest], [largest]])
    with pytest.raises(ValueError, match="overflow"):
        hdbscan(values, **backend.kwargs, min_cluster_size=2)


def test_requested_precision_conversion_rejects_nonfinite(backend):
    if backend.precision == "float64":
        values = np.full((6, 1), np.inf)
    else:
        values = np.full((6, 1), 1e40, dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        hdbscan(values, **backend.kwargs)


def test_default_cpu_stream_still_returns_gpu_labels(backend, oracle, assert_partition, monkeypatch):
    if backend.name == "mlx":
        from hdbscan_gpu import _mlx

        mx = backend.module
        kernels = _mlx._kernels
        split_kernels = _mlx._split_kernels
        dispatched_streams = []

        def record_stream(kernel):
            def invoke(**kwargs):
                dispatched_streams.append(kwargs["stream"])
                return kernel(**kwargs)
            return invoke

        monkeypatch.setattr(
            _mlx, "_kernels",
            lambda capacity=0, register_prim=0, dimensions=0, distance_checks=False, unit_core=False:
            tuple(map(record_stream, kernels(capacity, register_prim, dimensions, distance_checks, unit_core))),
        )
        monkeypatch.setattr(
            _mlx, "_split_kernels",
            lambda capacity, unit_core, minimum_two:
            tuple(map(record_stream, split_kernels(capacity, unit_core, minimum_two))),
        )
        original = mx.default_device()
        try:
            mx.set_default_device(mx.cpu)
            with mx.stream(mx.cpu):
                native = mx.array(VALUES)
                assert mx.default_device() == mx.cpu
                result = hdbscan(native, **backend.kwargs, min_cluster_size=2, min_samples=1)
                assert mx.default_device() == mx.cpu
            backend.assert_device_labels(result, (6,))
            assert dispatched_streams == [mx.gpu, mx.gpu, mx.gpu]
        finally:
            mx.set_default_device(original)
    else:
        # CUDA has no CPU stream: a nondefault CUDA stream must still return
        # a device array without changing the caller's stream selection.
        stream = backend.module.cuda.Stream(non_blocking=True)
        with stream:
            result = hdbscan(VALUES, **backend.kwargs, min_cluster_size=2, min_samples=1)
            assert backend.module.cuda.get_current_stream() == stream
        stream.synchronize()
        backend.assert_device_labels(result, (6,))
    assert_partition(backend.host(result), oracle(VALUES, min_cluster_size=2, min_samples=1))


def test_production_call_without_sklearn_or_host_clustering(backend):
    # A fresh interpreter makes this meaningful even after oracle tests have
    # loaded sklearn in the pytest process. Blocking imports after installing
    # the finder checks lazy imports too.
    script = r"""
import importlib.abc
import sys
import numpy as np

class NoCpuClustering(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sklearn", "hdbscan"}:
            raise AssertionError("Production imported CPU clustering: " + fullname)

sys.meta_path.insert(0, NoCpuClustering())
from hdbscan_gpu import hdbscan
backend, precision = sys.argv[1:]
values = np.array([[0], [.01], [.02], [3], [3.01], [3.02]], np.float32)
if backend == "mlx":
    import mlx.core as device
    with device.stream(device.gpu):
        values = device.array(values)
else:
    import cupy as device
    values = device.asarray(values)
    def forbidden(*args, **kwargs):
        raise AssertionError("Native device input transferred to host")
    device.asnumpy = forbidden

original_array = np.array
def guard(function):
    def checked(value, *args, **kwargs):
        if type(value).__module__.split(".")[0] in {"mlx", "cupy"}:
            raise AssertionError("Native device input transferred to NumPy")
        return function(value, *args, **kwargs)
    return checked
np.array, np.asarray = guard(np.array), guard(np.asarray)
labels = hdbscan(values, backend=backend, precision=precision, min_cluster_size=2, min_samples=1)
if backend == "mlx":
    device.eval(labels)
    assert labels.__dlpack_device__()[0] == 8
    actual = original_array(labels)
else:
    assert labels.__dlpack_device__()[0] == 2
    actual = labels.get()
assert actual.dtype == np.int32
assert actual.shape == (6,)
assert (actual >= 0).all()
assert len(set(actual[:3])) == len(set(actual[3:])) == 1
assert actual[0] != actual[3]
assert not any(name.split(".")[0] in {"sklearn", "hdbscan"} for name in sys.modules)
print("device-only production call passed")
"""
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [sys.executable, "-c", script, backend.name, backend.precision],
        cwd=root, env=environment, text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "device-only production call passed" in result.stdout
