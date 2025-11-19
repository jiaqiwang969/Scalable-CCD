#include <metal_stdlib>
using namespace metal;

inline bool intersects_yz(float2 a_min, float2 a_max, float2 b_min, float2 b_max)
{
    return a_max.x >= b_min.x && a_min.x <= b_max.x && a_max.y >= b_min.y
        && a_min.y <= b_max.y;
}

inline bool share_a_vertex(int4 av, int4 bv)
{
    int a0 = av.x, a1 = av.y, a2 = av.z;
    int b0 = bv.x, b1 = bv.y, b2 = bv.z;
    return a0 == b0 || a0 == b1 || a0 == b2 || a1 == b0 || a1 == b1
        || a1 == b2 || a2 == b0 || a2 == b1 || a2 == b2;
}

inline int flip_id(int id) { return -id - 1; }

struct MiniBox {
    float2 min;
    float2 max;
    int4 vertex_ids;
    int element_id;

    inline bool intersects(const MiniBox other) const
    {
        return max.x >= other.min.x && min.x <= other.max.x
            && max.y >= other.min.y && min.y <= other.max.y;
    }
};

kernel void sweep_and_prune_one_list(
    device const float2* sorted_major [[buffer(0)]],
    device const float2* mini_min [[buffer(1)]],
    device const float2* mini_max [[buffer(2)]],
    device const int4* vertex_ids [[buffer(3)]],
    device const int* element_ids [[buffer(4)]],
    constant uint& num_boxes [[buffer(5)]],
    constant uint& start_box_id [[buffer(6)]],
    constant uint& max_overlap_cutoff [[buffer(7)]],
    device int2* overlaps [[buffer(8)]],
    device atomic_uint* real_count [[buffer(9)]],
    constant uint& overlaps_capacity [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint box_id = start_box_id + gid;
    if (gid >= max_overlap_cutoff)
        return;
    if (box_id + 1 >= num_boxes)
        return;

    float a_max = sorted_major[box_id].y;
    uint next = box_id + 1u;
    float b_min = sorted_major[next].x;

    while (a_max >= b_min && next < num_boxes) {
        if (intersects_yz(
                mini_min[box_id], mini_max[box_id], mini_min[next],
                mini_max[next])
            && !share_a_vertex(vertex_ids[box_id], vertex_ids[next])) {
            int a_eid = element_ids[box_id];
            int b_eid = element_ids[next];
            int x = (a_eid < b_eid) ? a_eid : b_eid;
            int y = (a_eid < b_eid) ? b_eid : a_eid;
            uint idx = atomic_fetch_add_explicit(
                real_count, 1u, memory_order_relaxed);
            if (idx < overlaps_capacity) {
                overlaps[idx] = int2(x, y);
            }
        }
        ++next;
        if (next < num_boxes) {
            b_min = sorted_major[next].x;
        }
    }
}

kernel void sweep_and_prune_two_lists(
    device const float2* sorted_major [[buffer(0)]],
    device const float2* mini_min [[buffer(1)]],
    device const float2* mini_max [[buffer(2)]],
    device const int4* vertex_ids [[buffer(3)]],
    device const int* element_ids [[buffer(4)]],
    constant uint& num_boxes [[buffer(5)]],
    constant uint& start_box_id [[buffer(6)]],
    constant uint& max_overlap_cutoff [[buffer(7)]],
    device int2* overlaps [[buffer(8)]],
    device atomic_uint* real_count [[buffer(9)]],
    constant uint& overlaps_capacity [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint box_id = start_box_id + gid;
    if (gid >= max_overlap_cutoff)
        return;
    if (box_id + 1 >= num_boxes)
        return;

    float a_max = sorted_major[box_id].y;
    uint next = box_id + 1u;
    float b_min = sorted_major[next].x;

    while (a_max >= b_min && next < num_boxes) {
        int a_id = element_ids[box_id];
        int b_id = element_ids[next];
        bool cross = ((a_id >= 0) && (b_id < 0)) || ((a_id < 0) && (b_id >= 0));
        if (cross && intersects_yz(
                         mini_min[box_id], mini_max[box_id], mini_min[next],
                         mini_max[next])
            && !share_a_vertex(vertex_ids[box_id], vertex_ids[next])) {
            int neg = (a_id < 0) ? a_id : b_id;
            int pos = (a_id < 0) ? b_id : a_id;
            int x = flip_id(neg), y = pos;
            uint idx = atomic_fetch_add_explicit(
                real_count, 1u, memory_order_relaxed);
            if (idx < overlaps_capacity) {
                overlaps[idx] = int2(x, y);
            }
        }
        ++next;
        if (next < num_boxes) {
            b_min = sorted_major[next].x;
        }
    }
}

// Larger queue helps deep chains in dense scenes; adjust here if needed.
constant uint QUEUE_SIZE = 256;

inline bool queue_push(
    threadgroup atomic_uint& start,
    threadgroup atomic_uint& end,
    threadgroup int2* storage,
    const int2 value)
{
    uint s = atomic_load_explicit(&start, memory_order_relaxed);
    uint e = atomic_load_explicit(&end, memory_order_relaxed);
    if ((e - s) >= QUEUE_SIZE) {
        return false;
    }
    uint idx = atomic_fetch_add_explicit(&end, 1u, memory_order_relaxed);
    storage[idx % QUEUE_SIZE] = value;
    return true;
}

inline void queue_copy_to_local(
    threadgroup atomic_uint& start,
    threadgroup atomic_uint& end,
    threadgroup int2* storage,
    threadgroup int2* local_cache,
    threadgroup atomic_uint& cache_count,
    uint lid)
{
    if (lid == 0) {
        uint s = atomic_load_explicit(&start, memory_order_relaxed);
        uint e = atomic_load_explicit(&end, memory_order_relaxed);
        uint count = min(static_cast<uint>(QUEUE_SIZE), e - s);
        for (uint i = 0; i < count; ++i) {
            local_cache[i] = storage[(s + i) % QUEUE_SIZE];
        }
        atomic_store_explicit(&cache_count, count, memory_order_relaxed);
        atomic_store_explicit(&start, s + count, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void sweep_and_tiniest_queue_one_list(
    device const float2* sorted_major [[buffer(0)]],
    device const float2* mini_min [[buffer(1)]],
    device const float2* mini_max [[buffer(2)]],
    device const int4* vertex_ids [[buffer(3)]],
    device const int* element_ids [[buffer(4)]],
    constant uint& num_boxes [[buffer(5)]],
    constant uint& start_box_id [[buffer(6)]],
    constant uint& max_overlap_cutoff [[buffer(7)]],
    device int2* overlaps [[buffer(8)]],
    device atomic_uint* real_count [[buffer(9)]],
    constant uint& overlaps_capacity [[buffer(10)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup int2 queue_storage[QUEUE_SIZE];
    threadgroup int2 queue_cache[QUEUE_SIZE];
    threadgroup atomic_uint queue_start;
    threadgroup atomic_uint queue_end;
    threadgroup atomic_uint cache_count;

    if (lid == 0) {
        atomic_store_explicit(&queue_start, 0u, memory_order_relaxed);
        atomic_store_explicit(&queue_end, 0u, memory_order_relaxed);
        atomic_store_explicit(&cache_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint box_id = start_box_id + gid;
    if (gid >= max_overlap_cutoff) {
        return;
    }
    if (box_id + 1 >= num_boxes) {
        return;
    }

    float a_max = sorted_major[box_id].y;
    uint next = box_id + 1u;
    float b_min = sorted_major[next].x;

    if (a_max >= b_min) {
        queue_push(queue_start, queue_end, queue_storage, int2(box_id, box_id + 1));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (true) {
        queue_copy_to_local(
            queue_start, queue_end, queue_storage, queue_cache, cache_count, lid);
        uint count = atomic_load_explicit(&cache_count, memory_order_relaxed);
        if (count == 0) {
            break;
        }
        if (lid < count) {
            int2 res = queue_cache[lid];
            MiniBox a_mini = MiniBox{ mini_min[res.x], mini_max[res.x], vertex_ids[res.x], element_ids[res.x] };
            MiniBox b_mini = MiniBox{ mini_min[res.y], mini_max[res.y], vertex_ids[res.y], element_ids[res.y] };

            if (a_mini.intersects(b_mini)
                && !share_a_vertex(vertex_ids[res.x], vertex_ids[res.y])) {
                int a_eid = element_ids[res.x];
                int b_eid = element_ids[res.y];
                int x = (a_eid < b_eid) ? a_eid : b_eid;
                int y = (a_eid < b_eid) ? b_eid : a_eid;
                uint idx = atomic_fetch_add_explicit(real_count, 1u, memory_order_relaxed);
                if (idx < overlaps_capacity) {
                    overlaps[idx] = int2(x, y);
                }
            }

            if (res.y + 1 < static_cast<int>(num_boxes)) {
                float na_max = sorted_major[res.x].y;
                float nb_min = sorted_major[res.y + 1].x;
                if (na_max >= nb_min) {
                    queue_push(queue_start, queue_end, queue_storage, int2(res.x, res.y + 1));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void sweep_and_tiniest_queue_two_lists(
    device const float2* sorted_major [[buffer(0)]],
    device const float2* mini_min [[buffer(1)]],
    device const float2* mini_max [[buffer(2)]],
    device const int4* vertex_ids [[buffer(3)]],
    device const int* element_ids [[buffer(4)]],
    constant uint& num_boxes [[buffer(5)]],
    constant uint& start_box_id [[buffer(6)]],
    constant uint& max_overlap_cutoff [[buffer(7)]],
    device int2* overlaps [[buffer(8)]],
    device atomic_uint* real_count [[buffer(9)]],
    constant uint& overlaps_capacity [[buffer(10)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup int2 queue_storage[QUEUE_SIZE];
    threadgroup int2 queue_cache[QUEUE_SIZE];
    threadgroup atomic_uint queue_start;
    threadgroup atomic_uint queue_end;
    threadgroup atomic_uint cache_count;

    if (lid == 0) {
        atomic_store_explicit(&queue_start, 0u, memory_order_relaxed);
        atomic_store_explicit(&queue_end, 0u, memory_order_relaxed);
        atomic_store_explicit(&cache_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint box_id = start_box_id + gid;
    if (gid >= max_overlap_cutoff) {
        return;
    }
    if (box_id + 1 >= num_boxes) {
        return;
    }

    float a_max = sorted_major[box_id].y;
    uint next = box_id + 1u;
    float b_min = sorted_major[next].x;

    if (a_max >= b_min) {
        queue_push(queue_start, queue_end, queue_storage, int2(box_id, box_id + 1));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (true) {
        queue_copy_to_local(
            queue_start, queue_end, queue_storage, queue_cache, cache_count, lid);
        uint count = atomic_load_explicit(&cache_count, memory_order_relaxed);
        if (count == 0) {
            break;
        }
        if (lid < count) {
            int2 res = queue_cache[lid];
            int a_id = element_ids[res.x];
            int b_id = element_ids[res.y];
            bool cross = ((a_id >= 0 && b_id < 0) || (a_id < 0 && b_id >= 0));
            if (cross && intersects_yz(
                             mini_min[res.x], mini_max[res.x], mini_min[res.y],
                             mini_max[res.y])
                && !share_a_vertex(vertex_ids[res.x], vertex_ids[res.y])) {
                int neg = (a_id < 0) ? a_id : b_id;
                int pos = (a_id < 0) ? b_id : a_id;
                uint idx = atomic_fetch_add_explicit(real_count, 1u, memory_order_relaxed);
                if (idx < overlaps_capacity) {
                    overlaps[idx] = int2(flip_id(neg), pos);
                }
            }
            if (res.y + 1 < static_cast<int>(num_boxes)) {
                float na_max = sorted_major[res.x].y;
                float nb_min = sorted_major[res.y + 1].x;
                if (na_max >= nb_min) {
                    queue_push(queue_start, queue_end, queue_storage, int2(res.x, res.y + 1));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
