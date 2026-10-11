from functools import lru_cache

import cupy as cp

from ._kernels import (
    CLUSTER_SOURCE,
    CLUSTER_THREADS,
    DISTANCE_SOURCE,
    INTEGER_WORKSPACE,
    REAL_WORKSPACE,
    header,
)

@lru_cache(maxsize=64)
def _kernels(
    precision, shared_workspace=False, unit_core=False, minimum_two=False,
    register_prim=0, threads=CLUSTER_THREADS, parallel_sort=False,
):
    common = header("cuda", precision)
    distances = cp.RawKernel(
        common
        + 'extern "C" __global__ void distances_kernel(const Real* x, const int* params, Real* distances) {'
        + DISTANCE_SOURCE
        + "}",
        "distances_kernel",
        options=("--fmad=false",),
    )
    clusters = cp.RawKernel(
        common
        + "#define PARALLEL_PRIM\n"
        + ("#define CUDA_PACKED_WORKSPACE\n" if shared_workspace else "")
        + ("#define UNIT_CORE\n" if unit_core else "")
        + ("#define MINIMUM_TWO\n" if minimum_two else "")
        + (f"#define REGISTER_PRIM {register_prim}\n" if register_prim else "")
        + ("#define PARALLEL_SORT\n" if parallel_sort else "")
        + f'extern "C" __global__ __launch_bounds__({threads}) void clusters_kernel(const Real* distances, const Real* core, '
        + "const int* params, const Real* options, int* labels, int* integers, Real* reals) {"
        + CLUSTER_SOURCE
        + "}",
        "clusters_kernel",
        options=("--fmad=false",),
    )
    return distances, clusters


@lru_cache(maxsize=None)
def _shared_kernel(precision, device_id, unit_core, minimum_two, register_prim, threads, parallel_sort):
    device = cp.cuda.Device(device_id)
    with device:
        _, kernel = _kernels(
            precision, shared_workspace=True, unit_core=unit_core,
            minimum_two=minimum_two, register_prim=register_prim, threads=threads, parallel_sort=parallel_sort,
        )
        attributes = device.attributes
        limit = attributes.get("MaxSharedMemoryPerBlockOptin", attributes["MaxSharedMemoryPerBlock"])
        capacity = limit - kernel.shared_size_bytes
        kernel.max_dynamic_shared_size_bytes = capacity
    # Retain the configured object even if the general compilation cache evicts it.
    return kernel, capacity


def _packed_workspace_size(precision, size):
    return size * (19 + 3 * (4 if precision == "float32" else 8))


def _cluster_kernel(
    precision, size, unit_core=False, minimum_two=False, register_prim=0, threads=CLUSTER_THREADS,
    parallel_sort=False,
):
    required = _packed_workspace_size(precision, size)
    kernel, capacity = _shared_kernel(
        precision, cp.cuda.Device().id, unit_core, minimum_two, register_prim, threads, parallel_sort,
    )
    if required <= capacity:
        return kernel, required
    return _kernels(
        precision, unit_core=unit_core, minimum_two=minimum_two, register_prim=register_prim, threads=threads,
        parallel_sort=parallel_sort,
    )[1], 0


def _production_kernel(precision, size, minimum, samples):
    threads = max(32, 1 << (size - 1).bit_length()) if size <= 1024 else CLUSTER_THREADS
    kernel, shared = _cluster_kernel(
        precision, size, unit_core=samples <= 2, minimum_two=minimum == 2,
        register_prim=1 if size <= 1024 else 0, threads=threads, parallel_sort=size <= 1024,
    )
    return kernel, shared, threads


def cluster(x, min_cluster_size, min_samples, epsilon, precision):
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("the CUDA backend requires a CUDA GPU")
    if isinstance(x, cp.ndarray) and x.dtype.kind not in "fiu":
        raise ValueError("X must contain real numeric values")
    dtype = cp.float32 if precision == "float32" else cp.float64
    x = cp.ascontiguousarray(cp.asarray(x, dtype=dtype))
    if not bool(cp.all(cp.isfinite(x)).item()):
        raise ValueError("X must contain only finite values in the requested precision")
    unbatched = x.ndim == 2
    if unbatched:
        x = x[None]
    batch, n, dimensions = x.shape
    params = cp.asarray([n, dimensions, batch, min_cluster_size], dtype=cp.int32)
    options = cp.asarray([epsilon], dtype=dtype)
    distance_kernel, _ = _kernels(precision)
    cluster_kernel, shared_mem, threads = _production_kernel(
        precision, n, min_cluster_size, min_samples,
    )
    distances = cp.empty((batch, n, n), dtype=dtype)
    distance_kernel(((distances.size + 255) // 256,), (256,), (x, params, distances))
    if not bool(cp.all(cp.isfinite(distances)).item()):
        raise ValueError("pairwise distances overflow the requested precision")
    # With self-inclusive k <= 2, both core radii are <= every off-diagonal edge.
    core = (
        distances if min_samples <= 2 else
        cp.ascontiguousarray(cp.partition(distances, min_samples - 1, axis=-1)[..., min_samples - 1])
    )
    labels = cp.empty((batch, n), dtype=cp.int32)
    integers = cp.empty((1,) if shared_mem else (batch, INTEGER_WORKSPACE * n), dtype=cp.int32)
    reals = cp.empty((1,) if shared_mem else (batch, REAL_WORKSPACE * n), dtype=dtype)
    cluster_kernel(
        (batch,),
        (threads,),
        (distances, core, params, options, labels, integers, reals),
        shared_mem=shared_mem,
    )
    return labels[0] if unbatched else labels
