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

