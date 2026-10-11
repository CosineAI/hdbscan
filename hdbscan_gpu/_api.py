import math
from numbers import Integral, Real

import numpy as np


MAX_SAMPLES = 4096
MAX_DISTANCE_BYTES = 512 * 1024 * 1024


def hdbscan(
    X,
    *,
    min_cluster_size=5,
    min_samples=None,
    cluster_selection_epsilon=0.0,
    backend="mlx",
    precision="float32",
):
    """Return device labels for Euclidean EOM HDBSCAN, excluding the root.

    Accepts [N, D] or [B, N, D]; min_samples includes the point itself.
    Host arrays are explicitly transferred. Native backend arrays are never
    copied to NumPy. Two scalar GPU reductions validate finite inputs and
    distances, synchronizing execution; clustering intermediates stay on GPU.

    Metal supports float32 only; CUDA supports float32 and float64.
    Windows are limited to 4096 points and the two dense distance buffers
    together to 512 MiB. Float32 rounding can change near-tied hierarchies
    relative to sklearn's float64 calculation. Labels are int32, with -1 noise.
    """
    if backend not in ("mlx", "cuda"):
        raise ValueError("backend must be 'mlx' or 'cuda'")
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    if backend == "mlx" and precision != "float32":
        raise ValueError("Metal supports only precision='float32'")
    for name, value, lower in (
        ("min_cluster_size", min_cluster_size, 2),
        ("min_samples", min_samples, 1),
    ):
        if value is None and name == "min_samples":
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
        if value < lower:
            raise ValueError(f"{name} must be >= {lower}")
    if (
        isinstance(cluster_selection_epsilon, (bool, np.bool_))
        or not isinstance(cluster_selection_epsilon, Real)
        or not math.isfinite(cluster_selection_epsilon)
        or cluster_selection_epsilon < 0
    ):
        raise ValueError("cluster_selection_epsilon must be finite and >= 0")
    if min_samples is None:
        min_samples = min_cluster_size

    module = type(X).__module__.split(".")[0]
    if module in ("mlx", "cupy"):
        if (module == "mlx") != (backend == "mlx"):
            raise ValueError("native device input must match the selected backend")
    else:
        if hasattr(X, "__cuda_array_interface__") or hasattr(X, "__dlpack_device__") and not isinstance(X, np.ndarray):
            raise TypeError("use a native MLX/CuPy array or a NumPy host array")
        X = np.asarray(X)
        if X.dtype.kind not in "fiu":
            raise ValueError("X must contain real numeric values")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")

    if X.ndim not in (2, 3):
        raise ValueError("X must have rank 2 [N,D] or rank 3 [B,N,D]")
    n, dimensions = X.shape[-2:]
    batch = X.shape[0] if X.ndim == 3 else 1
    if batch < 1 or dimensions < 1 or n < 2:
        raise ValueError("X requires nonempty batches/features and at least 2 samples")
    if n > MAX_SAMPLES:
        raise ValueError(f"at most {MAX_SAMPLES} samples per window are supported")
    if min_samples > n:
        raise ValueError("min_samples must not exceed the number of samples")
    itemsize = 4 if precision == "float32" else 8
    if 2 * batch * n * n * itemsize > MAX_DISTANCE_BYTES:
        raise ValueError("dense distance workspace exceeds the 512 MiB limit")
    if min_cluster_size > np.iinfo(np.int32).max:
        raise ValueError("min_cluster_size must fit in int32")
    if cluster_selection_epsilon > np.finfo(precision).max:
        raise ValueError("cluster_selection_epsilon is not representable in the requested precision")

    if isinstance(X, np.ndarray):
        # MLX's DLPack importer rejects negative-stride host views.
        X = np.ascontiguousarray(X)
    if backend == "mlx":
        from ._mlx import cluster
    else:
        from ._cuda import cluster
    return cluster(
        X,
        int(min_cluster_size),
        int(min_samples),
        float(cluster_selection_epsilon),
        precision,
    )
