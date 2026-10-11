import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import benchmark


@pytest.mark.parametrize("actual,expected,equal", [
    ([7, 7, 3, 3, -1], [0, 0, 1, 1, -1], True),
    ([0, 0, 1, 1], [1, 1, 0, 0], True),
    ([-1, -1], [-1, -1], True),
    ([], [], True),
    ([0, 0, 0, 0], [0, 0, 1, 1], False),
    ([0, 1, 2, 2], [0, 0, 1, 1], False),
    ([0, 0, 1, 1], [0, 0, 1, -1], False),
    ([0, -1, 1, 1], [0, 0, 1, -1], False),
    ([0, 1], [0], False),
    ([[0, 0]], [0, 0], False),
])
def test_partition_equal_requires_bijection_and_exact_noise(actual, expected, equal):
    assert benchmark.partition_equal(actual, expected) is equal
    assert benchmark.partition_equal(expected, actual) is equal


@pytest.mark.parametrize("shape", [(1, 16, 1), (2, 16, 3), (3, 7, 8)])
def test_make_windows_is_reproducible_finite_float32(shape):
    values = benchmark.make_windows(*shape, seed=1729)
    assert values.shape == shape
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values, benchmark.make_windows(*shape, seed=1729))
    assert not np.array_equal(values, benchmark.make_windows(*shape, seed=1730))


@pytest.mark.parametrize("warmup", [0, 2])
def test_timings_excludes_warmup_and_reports_correct_statistics(monkeypatch, warmup):
    events = []
    clock = iter([10, 10.25, 20, 20.5, 30, 30.75, 40, 41])

    def now():
        events.append("clock")
        return next(clock)

    def operation():
        events.append("operation")

    monkeypatch.setattr(benchmark, "perf_counter", now)
    result = benchmark.timings(operation, warmup=warmup, repeats=4, windows=3)
    assert events == ["operation"] * warmup + ["clock", "operation", "clock"] * 4
    assert result["samples_seconds"] == [0.25, 0.5, 0.75, 1.0]
    assert result["median_batch_ms"] == 625.0
    assert result["p95_batch_ms"] == pytest.approx(962.5)
    assert result["windows_per_second"] == pytest.approx(4.8)


@pytest.mark.parametrize("warmup", [0, 1])
def test_timings_propagates_operation_failures(warmup):
    def operation():
        raise RuntimeError("operation did not complete")

    with pytest.raises(RuntimeError, match="operation did not complete"):
        benchmark.timings(operation, warmup=warmup, repeats=1, windows=1)


@pytest.mark.parametrize("cpu_executor", ["threads", "processes"])
def test_benchmark_cli_emits_valid_timings_and_exact_parity(backend, cpu_executor):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable, "-m", "benchmarks.benchmark",
            "--backend", backend.name, "--precision", backend.precision,
            "--samples", "16", "--dimensions", "2", "--batch-sizes", "1", "2",
            "--min-cluster-size", "2", "--min-samples", "1", "--epsilon", "0",
            "--warmup", "1", "--repeats", "2",
            "--cpu-workers", "2", "--cpu-executor", cpu_executor,
        ],
        cwd=root, text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    def reject_nonfinite(value):
        raise AssertionError(f"Invalid JSON numeric constant: {value}")

    report = json.loads(result.stdout, parse_constant=reject_nonfinite)
    assert report["schema_version"] == 1
    assert report["backend"] == backend.name
    assert report["precision"] == backend.precision
    assert report["sklearn"] == "1.9.1"
    assert report["numpy"] == "2.3.5"
    assert report["warmup"] == 1
    assert report["repeats"] == 2
    assert report["cpu_workers"] == 2
    assert report["cpu_executor"] == cpu_executor
    if cpu_executor == "processes":
        assert math.isfinite(report["cpu_worker_startup_seconds"])
        assert report["cpu_worker_startup_seconds"] > 0
    assert len(report["cases"]) == 2
    for case, batch in zip(report["cases"], (1, 2), strict=True):
        assert case["shape"] == [batch, 16, 2]
        assert case["matching_windows"] == case["total_windows"] == batch
        assert case["partition_and_noise_parity"] is True
        assert math.isfinite(case["first_gpu_call_seconds"])
        assert case["first_gpu_call_seconds"] > 0
        for name in ("cpu", "gpu_device_resident", "gpu_host_to_host"):
            timing = case[name]
            samples = timing["samples_seconds"]
            assert len(samples) == 2
            assert all(math.isfinite(value) and value > 0 for value in samples)
            median = float(np.median(samples))
            assert timing["median_batch_ms"] == pytest.approx(median * 1000)
            assert timing["p95_batch_ms"] == pytest.approx(float(np.quantile(samples, 0.95)) * 1000)
            assert timing["windows_per_second"] == pytest.approx(batch / median)
        assert case["resident_speedup_vs_cpu"] == pytest.approx(
            case["cpu"]["median_batch_ms"] / case["gpu_device_resident"]["median_batch_ms"],
        )


@pytest.mark.parametrize("failure_index", [None, 0, 4])
def test_corpus_validation_checks_tail_and_partition_splits(failure_index):
    windows = np.arange(5)[:, None, None]
    batches = []

    def cpu_one(window):
        return np.array([0, 0, 1, 1, -1])

    def gpu_batch(values):
        batches.append(len(values))
        return np.stack([
            [0, 1, 2, 2, -1] if int(window[0, 0]) == failure_index else [8, 8, 9, 9, -1]
            for window in values
        ])

    result = benchmark.validate_windows(windows, cpu_one, gpu_batch, batch_size=2)
    assert batches == [2, 2, 1]
    assert result["total_windows"] == 5
    assert result["matching_windows"] == 5 - int(failure_index is not None)
    assert result["partition_and_noise_parity"] is (failure_index is None)
    if failure_index is not None:
        assert result["failures"] == [{
            "window_index": failure_index, "noise_disagreements": 0,
            "cpu_clusters": 2, "gpu_clusters": 3,
        }]


def test_failed_corpus_gate_never_runs_timings(backend, tmp_path, monkeypatch):
    path = tmp_path / "windows.npy"
    np.save(path, np.zeros((3, 4, 2), dtype=np.float32))
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--backend", backend.name, "--precision", backend.precision,
        "--input", str(path), "--output", str(output), "--batch-sizes", "1", "2",
    ])
    failure = {
        "total_windows": 3, "matching_windows": 2, "partition_and_noise_parity": False,
        "failures": [{"window_index": 2}],
    }
    monkeypatch.setattr(benchmark, "validate_windows", lambda *args: failure)

    def unexpected_timings(*args):
        pytest.fail("a failed corpus must not be benchmarked")

    monkeypatch.setattr(benchmark, "timings", unexpected_timings)
    assert benchmark.main() == 2
    report = json.loads(output.read_text())
    assert report["corpus_parity"] == failure
    assert report["cases"] == []
    assert len(report["input_sha256"]) == 64
