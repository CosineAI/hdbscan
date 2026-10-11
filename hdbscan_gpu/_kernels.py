"""Shared GPU kernel bodies; no host clustering implementation."""

# Prim, linkage, condensation, and EOM are derived from scikit-learn:
# Authors: The scikit-learn developers
# Copyright (c) 2007-2026 The scikit-learn developers. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# The indirect introsort follows NumPy 2.3.5 npysort/quicksort.cpp and
# npysort_heapsort.h, originally by Charles R. Harris for numarray.
# Copyright (c) 2005-2025, NumPy Developers. All rights reserved.
# The same BSD-3-Clause conditions and disclaimer above apply.

INTEGER_ARRAYS = (
    "in_tree", "sources", "edge_u", "edge_v", "order", "uf", "left",
    "right", "sizes", "relabel", "queue", "prune", "parent", "selected",
    "label_map", "point_parent",
)
REAL_ARRAYS = ("best", "edge_w", "distance", "birth", "stability")
INTEGER_WORKSPACE = len(INTEGER_ARRAYS) * 2
REAL_WORKSPACE = len(REAL_ARRAYS) * 2
CLUSTER_THREADS = 256

DISTANCE_SOURCE = r"""
int tid = GLOBAL_ID;
int n = params[0], batches = params[2];
#ifdef DISTANCE_DIMENSIONS
const int d = DISTANCE_DIMENSIONS;
#else
int d = params[1];
#endif
int overflowed = 0;
if (tid < batches * n * n) {
int j, i, b;
if (n == 1024) {
    // Avoid repeated integer division for the common fixed-width embedding window.
    j = tid & 1023; i = (tid >> 10) & 1023; b = tid >> 20;
} else {
    j = tid % n; i = (tid / n) % n; b = tid / (n * n);
}
if (j >= i) {
Real total = 0;
#ifdef DISTANCE_DIMENSIONS
#pragma unroll
#endif
for (int k = 0; k < d; ++k) {
    Real delta = x[(b * n + i) * d + k] - x[(b * n + j) * d + k];
    total = total + delta * delta;
}
Real distance = SQRT(total);
distances[tid] = distance;
if (j != i) distances[(b * n + j) * n + i] = distance;
#ifdef DISTANCE_CHECKS
overflowed = !metal::isfinite(distance);
// Diagonal threads validate each input exactly once, with error priority over overflow.
if (i == j) {
    for (int k = 0; k < d; ++k)
        if (!metal::isfinite(x[(b * n + i) * d + k])) overflowed = 2;
}
#endif
}
}
#ifdef DISTANCE_CHECKS
int invalid_group = simd_max(overflowed);
if (tid < batches * n * n && (tid & 31) == 0) invalid[tid / 32] = invalid_group;
#endif
"""

_WORKSPACE = r"""
#if defined(METAL_PACKED_WORKSPACE) || defined(CUDA_PACKED_WORKSPACE)
#define PACKED_WORKSPACE
// Aliases change lifetime after Prim, linkage and condensation, never within a stage.
// All inter-thread writes are threadgroup-local; device inputs are immutable.
#ifdef METAL_PACKED_WORKSPACE
#undef BARRIER
#define BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
#define WORKSPACE_DEVICE threadgroup
threadgroup short packed_ints[9 * WORKSPACE_CAPACITY];
threadgroup uchar packed_selected[WORKSPACE_CAPACITY];
threadgroup Real packed_reals[3 * WORKSPACE_CAPACITY];
#else
#define WORKSPACE_DEVICE
#define WORKSPACE_CAPACITY n
extern __shared__ Real shared_workspace[];
// Reals first preserve alignment even when n is odd.
Real* packed_reals = shared_workspace;
short* packed_ints = (short*)(packed_reals + 3 * n);
unsigned char* packed_selected = (unsigned char*)(packed_ints + 9 * n);
#endif
#define WORK_INDEX short
#define NODE_INDEX(id) ((id) - n)
#define CLUSTER_INDEX(id) ((id) - n)
#define NODE_SIZE(id) ((id) < n ? 1 : int(sizes[NODE_INDEX(id)]))
WORKSPACE_DEVICE short* in_tree = packed_ints;
WORKSPACE_DEVICE short* sources = packed_ints + WORKSPACE_CAPACITY;
WORKSPACE_DEVICE short* edge_u = packed_ints + 2 * WORKSPACE_CAPACITY;
WORKSPACE_DEVICE short* edge_v = packed_ints + 3 * WORKSPACE_CAPACITY;
WORKSPACE_DEVICE short* order = packed_ints + 4 * WORKSPACE_CAPACITY;
WORKSPACE_DEVICE short* uf = packed_ints;
WORKSPACE_DEVICE short* left = packed_ints + 5 * WORKSPACE_CAPACITY;
WORKSPACE_DEVICE short* right = packed_ints + 6 * WORKSPACE_CAPACITY;
WORKSPACE_DEVICE short* sizes = packed_ints + 7 * WORKSPACE_CAPACITY;
WORKSPACE_DEVICE short* relabel = edge_u;
WORKSPACE_DEVICE short* queue = edge_v;
WORKSPACE_DEVICE short* prune = packed_ints;
WORKSPACE_DEVICE short* parent = order;
WORKSPACE_DEVICE unsigned char* selected = packed_selected;
WORKSPACE_DEVICE short* label_map = queue;
WORKSPACE_DEVICE short* point_parent = packed_ints + 8 * WORKSPACE_CAPACITY;
WORKSPACE_DEVICE Real* best = packed_reals;
WORKSPACE_DEVICE Real* edge_w = packed_reals + WORKSPACE_CAPACITY;
WORKSPACE_DEVICE Real* distance = best;
WORKSPACE_DEVICE Real* birth = edge_w;
WORKSPACE_DEVICE Real* stability = packed_reals + 2 * WORKSPACE_CAPACITY;
#else
#define WORKSPACE_DEVICE DEVICE
#define WORK_INDEX int
#define NODE_INDEX(id) (id)
#define CLUSTER_INDEX(id) (id)
#define NODE_SIZE(id) sizes[id]
""" + f"""
DEVICE int* integer_base = integers + b * {INTEGER_WORKSPACE} * n;
DEVICE Real* real_base = reals + b * {REAL_WORKSPACE} * n;
""" + "\n".join(
    f"DEVICE int* {name} = integer_base + {2 * i} * n;"
    for i, name in enumerate(INTEGER_ARRAYS)
) + "\n" + "\n".join(
    f"DEVICE Real* {name} = real_base + {2 * i} * n;"
    for i, name in enumerate(REAL_ARRAYS)
) + "\n#endif\n"

CLUSTER_PREAMBLE = r"""
#ifdef PARALLEL_PRIM
int b = GROUP_ID, lane = LOCAL_ID;
#else
int b = GLOBAL_ID;
#endif
int n = params[0], batches = params[2], minimum = params[3];
if (b >= batches) return;
Real epsilon = options[0];
""" + _WORKSPACE

PRIM_SOURCE = CLUSTER_PREAMBLE + r"""
#ifdef PACKED_WORKSPACE
for (int i = lane; i < n; i += GROUP_SIZE) {
    in_tree[i] = 0;
    sources[i] = 1;
    best[i] = INFINITY;
}
#else
#ifdef PARALLEL_PRIM
for (int i = lane; i < 2 * n; i += GROUP_SIZE) {
#else
for (int i = 0; i < 2 * n; ++i) {
#endif
    in_tree[i] = 0;
    sources[i] = 1;
    best[i] = INFINITY;
    uf[i] = i;
    sizes[i] = 1;
    relabel[i] = -1;
    parent[i] = -1;
    selected[i] = 0;
    label_map[i] = -1;
    birth[i] = 0;
    stability[i] = 0;
}
#endif
#ifdef PARALLEL_PRIM
SHARED int current_vertex;
SHARED Real warp_weights[32];
SHARED int warp_vertices[32];
#ifdef REGISTER_PRIM
SHARED int warp_sources[32];
Real local_best[REGISTER_PRIM];
int local_sources[REGISTER_PRIM];
int visited = 0;
#pragma unroll
for (int slot = 0; slot < REGISTER_PRIM; ++slot) {
    local_best[slot] = INFINITY;
    local_sources[slot] = 1;
}
#define PRIM_BEST(slot, point) local_best[slot]
#define PRIM_SOURCE_ID(slot, point) local_sources[slot]
#else
#define PRIM_BEST(slot, point) best[point]
#define PRIM_SOURCE_ID(slot, point) sources[point]
#endif
if (lane == 0) {
    current_vertex = 0;
    in_tree[0] = 1;
}
BARRIER();
for (int i = 0; i < n - 1; ++i) {
    int current = current_vertex;
    Real candidate = REAL_MAX;
    int next = 0, source = 0;
#ifdef REGISTER_PRIM
#pragma unroll
    for (int slot = 0; slot < REGISTER_PRIM; ++slot) {
        int j = lane + slot * GROUP_SIZE;
        if (j >= n) continue;
        if (j == current) visited |= 1 << slot;
        if (visited & (1 << slot)) continue;
#else
    for (int j = lane; j < n; j += GROUP_SIZE) {
        if (in_tree[j]) continue;
#endif
#ifdef UNIT_CORE
        Real weight = distances[(b * n + current) * n + j];
#else
        Real weight = MAXIMUM(core[b * n + current], core[b * n + j]);
        weight = MAXIMUM(weight, distances[(b * n + current) * n + j]);
#endif
        // Equal edges must retain the previous source, exactly as in serial Prim.
        if (weight < PRIM_BEST(slot, j)) {
            PRIM_BEST(slot, j) = weight;
            PRIM_SOURCE_ID(slot, j) = current;
        }
        weight = PRIM_BEST(slot, j);
        if (weight < candidate || (weight == candidate && j < next)) {
            candidate = weight;
            next = j;
            source = PRIM_SOURCE_ID(slot, j);
        }
    }
    // Reduce by global vertex index, not lane index, to preserve scan-order ties.
#ifdef METAL_NATIVE_REDUCE
reduce_candidate(candidate, next, source);
#else
    for (int offset = 16; offset > 0; offset >>= 1) {
        Real other = SHUFFLE_DOWN(candidate, offset);
        int other_vertex = SHUFFLE_DOWN(next, offset);
#ifdef REGISTER_PRIM
        int other_source = SHUFFLE_DOWN(source, offset);
#endif
        if ((lane & 31) + offset < 32 &&
            (other < candidate || (other == candidate && other_vertex < next))) {
            candidate = other;
            next = other_vertex;
#ifdef REGISTER_PRIM
            source = other_source;
#endif
        }
    }
#endif
    if (GROUP_SIZE > 32) {
        if ((lane & 31) == 0) {
            warp_weights[lane >> 5] = candidate;
            warp_vertices[lane >> 5] = next;
#ifdef REGISTER_PRIM
            warp_sources[lane >> 5] = source;
#endif
        }
        BARRIER();
        if (lane < 32) {
            candidate = lane < GROUP_SIZE / 32 ? warp_weights[lane] : REAL_MAX;
            next = lane < GROUP_SIZE / 32 ? warp_vertices[lane] : 0;
#ifdef REGISTER_PRIM
            source = lane < GROUP_SIZE / 32 ? warp_sources[lane] : 0;
#endif
#ifdef METAL_NATIVE_REDUCE
            reduce_candidate(candidate, next, source);
#else
            for (int offset = 16; offset > 0; offset >>= 1) {
                Real other = SHUFFLE_DOWN(candidate, offset);
                int other_vertex = SHUFFLE_DOWN(next, offset);
#ifdef REGISTER_PRIM
                int other_source = SHUFFLE_DOWN(source, offset);
#endif
                if (lane + offset < 32 &&
                    (other < candidate || (other == candidate && other_vertex < next))) {
                    candidate = other;
                    next = other_vertex;
#ifdef REGISTER_PRIM
                    source = other_source;
#endif
                }
            }
#endif
        }
    }
#ifndef REGISTER_PRIM
    if (GROUP_SIZE == 32) BARRIER();
#endif
    if (lane == 0) {
#ifdef REGISTER_PRIM
        edge_u[i] = candidate < REAL_MAX ? source : 0;
#else
        edge_u[i] = candidate < REAL_MAX ? sources[next] : 0;
#endif
        edge_v[i] = next;
        edge_w[i] = candidate;
        order[i] = i;
        current_vertex = next;
        if (i < n - 2) in_tree[next] = 1;
    }
    BARRIER();
}
#ifndef PARALLEL_SORT
if (lane != 0) return;
#endif
#else
int current = 0;
for (int i = 0; i < n - 1; ++i) {
    in_tree[current] = 1;
    Real candidate = REAL_MAX;
    int next = 0, source = 0;
    for (int j = 0; j < n; ++j) {
        if (in_tree[j]) continue;
        Real old = best[j];
        Real weight = MAXIMUM(core[b * n + current], core[b * n + j]);
        weight = MAXIMUM(weight, distances[(b * n + current) * n + j]);
        if (weight < old) {
            best[j] = weight;
            sources[j] = current;
            if (weight < candidate) {
                candidate = weight;
                source = current;
                next = j;
            }
        } else if (old < candidate) {
            candidate = old;
            source = sources[j];
            next = j;
        }
    }
    edge_u[i] = source;
    edge_v[i] = next;
    edge_w[i] = candidate;
    order[i] = i;
    current = next;
}
#endif
"""

_HEAPSORT_SOURCE = r"""
        int count = hi - lo + 1;
        for (int root = count / 2; root >= 1; --root) {
            int value = order[lo + root - 1], pos = root, child = 2 * root;
            while (child <= count) {
                if (child < count && edge_w[order[lo + child - 1]] < edge_w[order[lo + child]]) ++child;
                if (edge_w[value] < edge_w[order[lo + child - 1]]) {
                    order[lo + pos - 1] = order[lo + child - 1];
                    pos = child;
                    child *= 2;
                } else break;
            }
            order[lo + pos - 1] = value;
        }
        for (int end = count; end > 1;) {
            int value = order[lo + end - 1];
            order[lo + end - 1] = order[lo];
            --end;
            int pos = 1, child = 2;
            while (child <= end) {
                if (child < end && edge_w[order[lo + child - 1]] < edge_w[order[lo + child]]) ++child;
                if (edge_w[value] < edge_w[order[lo + child - 1]]) {
                    order[lo + pos - 1] = order[lo + child - 1];
                    pos = child;
                    child *= 2;
                } else break;
            }
            order[lo + pos - 1] = value;
        }
"""

_PARTITION_SOURCE = r"""
            int mid = lo + ((hi - lo) >> 1);
            if (edge_w[order[mid]] < edge_w[order[lo]]) { SWAP(order[mid], order[lo]); }
            if (edge_w[order[hi]] < edge_w[order[mid]]) { SWAP(order[hi], order[mid]); }
            if (edge_w[order[mid]] < edge_w[order[lo]]) { SWAP(order[mid], order[lo]); }
            Real pivot = edge_w[order[mid]];
            int i = lo, j = hi - 1;
            SWAP(order[mid], order[j]);
            for (;;) {
                do { ++i; } while (edge_w[order[i]] < pivot);
                do { --j; } while (pivot < edge_w[order[j]]);
                if (i >= j) break;
                SWAP(order[i], order[j]);
            }
            SWAP(order[i], order[hi - 1]);
            --depth;
"""

_INSERTION_SOURCE = r"""
        for (int i = lo + 1; i <= hi; ++i) {
            int value = order[i], j = i;
            while (j > lo && edge_w[value] < edge_w[order[j - 1]]) {
                order[j] = order[j - 1]; --j;
            }
            order[j] = value;
        }
"""

SORT_SOURCE = r"""
// The partition algorithm is identical in both paths, including unstable ties.
#ifdef PARALLEL_SORT
// The six queues reuse dead Prim state and not-yet-live linkage arrays.
WORKSPACE_DEVICE WORK_INDEX* current_lo = in_tree;
WORKSPACE_DEVICE WORK_INDEX* current_hi = sources;
WORKSPACE_DEVICE WORK_INDEX* current_depth = left;
WORKSPACE_DEVICE WORK_INDEX* next_lo = right;
WORKSPACE_DEVICE WORK_INDEX* next_hi = sizes;
WORKSPACE_DEVICE WORK_INDEX* next_depth = point_parent;
SHARED int task_count;
SHARED ATOMIC_UINT pending_count;
if (lane == 0) {
    int depth = 0;
    for (int count = n - 1; count > 1; count >>= 1) depth += 2;
#ifdef SORT_DEPTH_OVERRIDE
    depth = SORT_DEPTH_OVERRIDE;
#endif
    task_count = 1;
    current_lo[0] = 0;
    current_hi[0] = n - 2;
    current_depth[0] = depth;
}
BARRIER();
while (task_count > 0) {
    if (lane == 0) ATOMIC_STORE(&pending_count, 0);
    BARRIER();
    for (int task = lane; task < task_count; task += GROUP_SIZE) {
        int lo = current_lo[task], hi = current_hi[task], depth = current_depth[task];
        if (depth < 0) {
""" + _HEAPSORT_SOURCE + r"""
        } else {
            // Only deferred tasks recheck depth. Continue the smaller child in place,
            // even when its depth becomes negative, exactly as NumPy does.
            while (hi - lo > 15) {
""" + _PARTITION_SOURCE + r"""
                int target = ATOMIC_ADD(&pending_count, 1);
                if (i - lo < hi - i) {
                    next_lo[target] = i + 1; next_hi[target] = hi;
                    hi = i - 1;
                } else {
                    next_lo[target] = lo; next_hi[target] = i - 1;
                    lo = i + 1;
                }
                next_depth[target] = depth;
            }
""" + _INSERTION_SOURCE + r"""
        }
    }
    BARRIER();
    if (lane == 0) task_count = ATOMIC_LOAD(&pending_count);
    BARRIER();
    WORKSPACE_DEVICE WORK_INDEX* temporary = current_lo; current_lo = next_lo; next_lo = temporary;
    temporary = current_hi; current_hi = next_hi; next_hi = temporary;
    temporary = current_depth; current_depth = next_depth; next_depth = temporary;
}
for (int i = lane; i < 2 * n; i += GROUP_SIZE) {
    uf[i] = i;
#ifndef PACKED_WORKSPACE
    // The sort queue reused leaf cardinalities; packed storage uses literal ones.
    if (i < n) sizes[i] = 1;
#endif
}
BARRIER();
#else
// Smaller partition first bounds the pending stack by log2(n); n <= 4096.
#ifdef PARALLEL_PRIM
SHARED int stack_lo[16], stack_hi[16], stack_depth[16];
#else
int stack_lo[16], stack_hi[16], stack_depth[16];
#endif
int top = 0;
int lo = 0, hi = n - 2, depth = 0;
for (int count = n - 1; count > 1; count >>= 1) depth += 2;
#ifdef SORT_DEPTH_OVERRIDE
depth = SORT_DEPTH_OVERRIDE;
#endif
for (;;) {
    if (depth < 0) {
""" + _HEAPSORT_SOURCE + r"""
    } else {
        while (hi - lo > 15) {
""" + _PARTITION_SOURCE + r"""
            if (i - lo < hi - i) {
                stack_lo[top] = i + 1; stack_hi[top] = hi;
                hi = i - 1;
            } else {
                stack_lo[top] = lo; stack_hi[top] = i - 1;
                lo = i + 1;
            }
            stack_depth[top++] = depth;
        }
""" + _INSERTION_SOURCE + r"""
    }
    if (top == 0) break;
    --top;
    lo = stack_lo[top]; hi = stack_hi[top]; depth = stack_depth[top];
}
#endif
"""

LINKAGE_SOURCE = r"""
#ifdef PARALLEL_SORT
if (lane == 0) {
#endif
#if defined(PACKED_WORKSPACE) && !defined(PARALLEL_SORT)
for (int i = 0; i < 2 * n; ++i) uf[i] = i;
#endif
// Keep representatives among leaves; linkage IDs no longer extend lookup chains.
WORKSPACE_DEVICE WORK_INDEX* component_node = uf + n;
for (int i = 0; i < n; ++i) component_node[i] = i;
for (int i = 0; i < n - 1; ++i) {
    int edge = order[i], a = edge_u[edge], c = edge_v[edge];
    while (uf[a] != a) {
        uf[a] = uf[uf[a]];
        a = uf[a];
    }
    while (uf[c] != c) {
        uf[c] = uf[uf[c]];
        c = uf[c];
    }
    int node = n + i;
    int child_a = component_node[a], child_c = component_node[c];
    int size_a = NODE_SIZE(child_a), size_c = NODE_SIZE(child_c);
    left[NODE_INDEX(node)] = child_a; right[NODE_INDEX(node)] = child_c;
    sizes[NODE_INDEX(node)] = size_a + size_c;
    distance[NODE_INDEX(node)] = edge_w[edge];
    if (size_a < size_c) {
        uf[a] = c;
        component_node[c] = node;
    } else {
        uf[c] = a;
        component_node[a] = node;
    }
}
#ifdef PARALLEL_SORT
}
BARRIER();
#endif
"""

CONDENSE_SOURCE = r"""
#ifdef COOPERATIVE_BINARY_EOM
#if !(defined(METAL_PACKED_WORKSPACE) && defined(PARALLEL_SORT) && defined(COOPERATIVE_LABELS))
#error Cooperative binary EOM requires the split packed Metal hierarchy
#endif
SHARED int condensed_end;
#endif
#ifdef PACKED_WORKSPACE
#ifdef PARALLEL_SORT
for (int i = lane; i < n; i += GROUP_SIZE) {
#else
for (int i = 0; i < n; ++i) {
#endif
    relabel[i] = -1;
    parent[i] = -1;
    selected[i] = 0;
    birth[i] = 0;
    stability[i] = 0;
}
#endif
#ifdef PARALLEL_SORT
BARRIER();
#ifdef COOPERATIVE_LABELS
if (lane == 0) {
#else
if (lane != 0) return;
#endif
#endif
int root = 2 * n - 2, head = 0, tail = 1, next_label = n + 1;
queue[0] = root;
relabel[NODE_INDEX(root)] = n;
// Queue order determines condensed cluster IDs and final numeric labels.
while (head < tail) {
    int node = queue[head++], a = left[NODE_INDEX(node)], c = right[NODE_INDEX(node)];
    int cluster = relabel[NODE_INDEX(node)];
    Real cluster_birth = birth[CLUSTER_INDEX(cluster)];
    Real cluster_stability = stability[CLUSTER_INDEX(cluster)];
    Real lambda = distance[NODE_INDEX(node)] > 0 ? Real(1) / distance[NODE_INDEX(node)] : Real(INFINITY);
#ifdef MINIMUM_TWO
    bool large_a = a >= n, large_c = c >= n;
#else
    bool large_a = NODE_SIZE(a) >= minimum, large_c = NODE_SIZE(c) >= minimum;
#endif
    if (large_a && large_c) {
        int children[2] = {a, c};
        for (int k = 0; k < 2; ++k) {
            int child = children[k], label = next_label++;
            relabel[NODE_INDEX(child)] = label;
            parent[CLUSTER_INDEX(label)] = cluster;
            birth[CLUSTER_INDEX(label)] = lambda;
            selected[CLUSTER_INDEX(label)] = 1;
            cluster_stability += (lambda - cluster_birth) * Real(NODE_SIZE(child));
            queue[tail++] = child;
        }
    } else {
        int children[2] = {a, c};
        for (int k = 0; k < 2; ++k) {
            int child = children[k];
            if (k == 0 ? large_a : large_c) {
                relabel[NODE_INDEX(child)] = cluster;
                queue[tail++] = child;
            } else {
#ifdef MINIMUM_TWO
                // A subtree with fewer than two points is necessarily one leaf.
                point_parent[child] = cluster;
                cluster_stability += lambda - cluster_birth;
#else
                int prune_head = 0, prune_tail = 0, leaf = child;
                for (;;) {
                    if (leaf < n) {
                        point_parent[leaf] = cluster;
                        cluster_stability += lambda - cluster_birth;
                    } else {
                        prune[prune_tail++] = left[NODE_INDEX(leaf)];
                        prune[prune_tail++] = right[NODE_INDEX(leaf)];
                    }
                    if (prune_head == prune_tail) break;
                    leaf = prune[prune_head++];
                }
#endif
            }
        }
    }
    stability[CLUSTER_INDEX(cluster)] = cluster_stability;
}
"""

HIERARCHY_SOURCE = SORT_SOURCE + LINKAGE_SOURCE + CONDENSE_SOURCE

EOM_SOURCE = r"""
#ifdef COOPERATIVE_BINARY_EOM
condensed_end = next_label;
}
BARRIER();
int next_label = condensed_end;
WORKSPACE_DEVICE WORK_INDEX* first_child = left;
for (int i = lane; i < n; i += GROUP_SIZE) {
    first_child[i] = -1;
    label_map[i] = -1;
    right[i] = 0;
    if (epsilon > 0 && i > 0 && n + i < next_label) {
        // Preserve the reciprocal round trip and distinct strict comparisons.
        Real separation = Real(1) / birth[i];
        right[i] = (separation < epsilon ? 1 : 0) | (separation > epsilon ? 2 : 0);
    }
}
BARRIER();
// Each condensed split creates exactly one consecutive child pair for its parent.
for (int child = n + 1 + 2 * lane; child < next_label; child += 2 * GROUP_SIZE)
    first_child[CLUSTER_INDEX(parent[CLUSTER_INDEX(child)])] = child;
BARRIER();
if (lane == 0) {
#else
// Linkage children are dead after condensation; reuse their storage for adjacency.
WORKSPACE_DEVICE WORK_INDEX* first_child = left;
WORKSPACE_DEVICE WORK_INDEX* next_sibling = right;
for (int node = n; node < next_label; ++node) {
    first_child[CLUSTER_INDEX(node)] = -1;
    next_sibling[CLUSTER_INDEX(node)] = -1;
}
// Descending insertion preserves the original ascending-ID addition order.
for (int child = next_label - 1; child > n; --child) {
    int ancestor = parent[CLUSTER_INDEX(child)];
    next_sibling[CLUSTER_INDEX(child)] = first_child[CLUSTER_INDEX(ancestor)];
    first_child[CLUSTER_INDEX(ancestor)] = child;
}
#endif
for (int node = next_label - 1; node > n; --node) {
    Real subtree = 0;
#ifdef COOPERATIVE_BINARY_EOM
    int child = first_child[CLUSTER_INDEX(node)];
    if (child != -1) {
        subtree += stability[CLUSTER_INDEX(child)];
        subtree += stability[CLUSTER_INDEX(child + 1)];
    }
#else
    for (int child = first_child[CLUSTER_INDEX(node)]; child != -1; child = next_sibling[CLUSTER_INDEX(child)])
        subtree += stability[CLUSTER_INDEX(child)];
#endif
    if (subtree > stability[CLUSTER_INDEX(node)]) {
        selected[CLUSTER_INDEX(node)] = 0;
        stability[CLUSTER_INDEX(node)] = subtree;
    }
}
// Parents have smaller IDs. Propagate every winning ancestor in one forward pass.
WORKSPACE_DEVICE WORK_INDEX* blocked = in_tree;
blocked[CLUSTER_INDEX(n)] = 0;
for (int node = n + 1; node < next_label; ++node) {
    int inherited = blocked[CLUSTER_INDEX(parent[CLUSTER_INDEX(node)])];
    blocked[CLUSTER_INDEX(node)] = inherited || selected[CLUSTER_INDEX(node)];
    if (inherited) selected[CLUSTER_INDEX(node)] = 0;
}
"""

LABEL_SOURCE = r"""
#if defined(PACKED_WORKSPACE) && !defined(COOPERATIVE_BINARY_EOM)
for (int i = 0; i < n; ++i) label_map[i] = -1;
#endif
if (epsilon > 0) {
    WORKSPACE_DEVICE WORK_INDEX* epsilon_stop = sizes;
    for (int node = n + 1; node < next_label; ++node) {
        int ancestor = parent[CLUSTER_INDEX(node)];
        epsilon_stop[CLUSTER_INDEX(node)] = ancestor == n ? node :
#ifdef COOPERATIVE_BINARY_EOM
            ((right[CLUSTER_INDEX(ancestor)] & 2) ? ancestor :
#else
            (Real(1) / birth[CLUSTER_INDEX(ancestor)] > epsilon ? ancestor :
#endif
             epsilon_stop[CLUSTER_INDEX(ancestor)]);
        if (!selected[CLUSTER_INDEX(node)]) continue;
        // Starting and ancestor comparisons deliberately differ at equality.
#ifdef COOPERATIVE_BINARY_EOM
        int chosen = (right[CLUSTER_INDEX(node)] & 1) ?
#else
        int chosen = Real(1) / birth[CLUSTER_INDEX(node)] < epsilon ?
#endif
            epsilon_stop[CLUSTER_INDEX(node)] : node;
        label_map[CLUSTER_INDEX(chosen)] = 1;
    }
    for (int node = n + 1; node < next_label; ++node)
        selected[CLUSTER_INDEX(node)] = label_map[CLUSTER_INDEX(node)] == 1;
}
int label = 0;
for (int node = n + 1; node < next_label; ++node)
    if (selected[CLUSTER_INDEX(node)]) label_map[CLUSTER_INDEX(node)] = label++;
// Condensed IDs are topological. Resolve each ancestor once instead of per point.
WORKSPACE_DEVICE WORK_INDEX* resolved = relabel;
resolved[CLUSTER_INDEX(n)] = -1;
for (int node = n + 1; node < next_label; ++node)
    resolved[CLUSTER_INDEX(node)] = selected[CLUSTER_INDEX(node)] ?
        label_map[CLUSTER_INDEX(node)] : resolved[CLUSTER_INDEX(parent[CLUSTER_INDEX(node)])];
#ifdef COOPERATIVE_LABELS
}
BARRIER();
for (int i = lane; i < n; i += GROUP_SIZE) {
    labels[b * n + i] = relabel[CLUSTER_INDEX(point_parent[i])];
}
#else
for (int i = 0; i < n; ++i) {
    labels[b * n + i] = resolved[CLUSTER_INDEX(point_parent[i])];
}
#endif
"""

CLUSTER_SOURCE = PRIM_SOURCE + HIERARCHY_SOURCE + EOM_SOURCE + LABEL_SOURCE


def metal_source(source):
    # MLX discovers builtin arguments in the body, not in the preamble.
    for name, expression in {
        "GLOBAL_ID": "int(thread_position_in_grid.x)",
        "GROUP_ID": "int(threadgroup_position_in_grid.x)",
        "LOCAL_ID": "int(thread_index_in_threadgroup)",
        "GROUP_SIZE": "int(threads_per_threadgroup.x)",
    }.items():
        source = source.replace(name, expression)
    return source


def header(backend, precision):
    real = "float" if precision == "float32" else "double"
    maximum = "3.4028234663852886e+38F" if real == "float" else "1.7976931348623157e+308"
    if backend == "mlx":
        definitions = (
            "#define DEVICE device\n"
            "#define GLOBAL_ID int(thread_position_in_grid.x)\n"
            "#define SQRT metal::precise::sqrt\n"
            "#define MAXIMUM metal::max\n"
            "#define SHARED threadgroup\n"
            "#define BARRIER() threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)\n"
            "#define SHUFFLE_DOWN(value, offset) simd_shuffle_down(value, offset)\n"
            "#define METAL_NATIVE_REDUCE\n"
            "#define ATOMIC_UINT atomic_uint\n"
            "#define ATOMIC_STORE(pointer, value) atomic_store_explicit(pointer, value, memory_order_relaxed)\n"
            "#define ATOMIC_ADD(pointer, value) atomic_fetch_add_explicit(pointer, value, memory_order_relaxed)\n"
            "#define ATOMIC_LOAD(pointer) atomic_load_explicit(pointer, memory_order_relaxed)\n"
        )
    else:
        # NVRTC supplies device math built-ins, but not host C library headers.
        definitions = (
            "#define INFINITY (__int_as_float(0x7f800000))\n"
            "#define DEVICE\n"
            "#define GLOBAL_ID int(blockIdx.x * blockDim.x + threadIdx.x)\n"
            "#define SQRT sqrt\n"
            "#define MAXIMUM fmax\n"
            "#define SHARED __shared__\n"
            "#define GROUP_ID int(blockIdx.x)\n"
            "#define LOCAL_ID int(threadIdx.x)\n"
            "#define GROUP_SIZE int(blockDim.x)\n"
            "#define BARRIER() __syncthreads()\n"
            "#define SHUFFLE_DOWN(value, offset) __shfl_down_sync(0xffffffff, value, offset)\n"
            "#define ATOMIC_UINT unsigned int\n"
            "#define ATOMIC_STORE(pointer, value) (*(pointer) = (value))\n"
            "#define ATOMIC_ADD(pointer, value) atomicAdd(pointer, value)\n"
            "#define ATOMIC_LOAD(pointer) (*(pointer))\n"
        )
    common = (
        definitions
        + f"typedef {real} Real;\n#define REAL_MAX {maximum}\n"
        + "#define SWAP(a, b) do { int temporary = (a); (a) = (b); (b) = temporary; } while (0)\n"
    )
    if backend == "mlx":
        common += r"""
inline void reduce_candidate(thread Real& weight, thread int& point, thread int& source) {
    Real minimum = simd_min(weight);
    int chosen = simd_min(weight == minimum ? point : 2147483647);
    source = simd_min(weight == minimum && point == chosen ? source : 2147483647);
    weight = minimum;
    point = chosen;
}
"""
    return common
