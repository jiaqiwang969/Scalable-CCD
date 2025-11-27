#include <metal_stdlib>
using namespace metal;

// --- Configuration & Types ---

// Force float for Metal
typedef float Scalar;
typedef float3 Vector3;

struct CCDConfig {
    Scalar co_domain_tolerance;
    int max_iter;
    bool use_ms;
    bool allow_zero_toi;
};

struct Interval {
    Scalar lower;
    Scalar upper;
    
    Interval() = default;
    Interval(Scalar l, Scalar u) : lower(l), upper(u) {}
};

struct SplitInterval {
    Interval first;
    Interval second;
    
    SplitInterval(Interval interval) {
        Scalar mid = (interval.lower + interval.upper) * 0.5f;
        first = Interval(interval.lower, mid);
        second = Interval(mid, interval.upper);
    }
};

struct CCDDomain {
    Interval tuv[3]; // t, u, v
    int query_id;
    
    void init(int i) {
        tuv[0] = Interval(0.0f, 1.0f);
        tuv[1] = Interval(0.0f, 1.0f);
        tuv[2] = Interval(0.0f, 1.0f);
        query_id = i;
    }
};

struct DomainCorner {
    Scalar t, u, v;
    
    void update_tuv(const thread CCDDomain& domain, uint8_t corner) {
        t = (corner & 1) ? domain.tuv[0].upper : domain.tuv[0].lower;
        u = (corner & 2) ? domain.tuv[1].upper : domain.tuv[1].lower;
        v = (corner & 4) ? domain.tuv[2].upper : domain.tuv[2].lower;
    }
};

struct CCDData {
    Vector3 v0s, v1s, v2s, v3s;
    Vector3 v0e, v1e, v2e, v3e;
    Scalar ms;
    Scalar tol[3];
    Vector3 err; // Changed to Vector3
    int nbr_checks;
    // toi, aid, bid are not strictly needed for the kernel calculation if we use a global toi
    // but we might need them if we do per-query TOI. For now, stick to global TOI.
};

struct CCDBuffer {
    device CCDDomain* data;
    uint starting_size;
    uint capacity;
    device atomic_uint* head; // Changed to pointer for atomic access
    device atomic_uint* tail;
    device atomic_int* overflow_flag;
};

// --- Helper Functions ---

inline bool sum_less_than_one(Scalar num1, Scalar num2) {
    // FLT_EPSILON is approx 1.19209e-07
    return num1 + num2 <= 1.0f / (1.0f - 1.19209e-07f);
}

inline Scalar max_Linf_4(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4,
                         Vector3 p1e, Vector3 p2e, Vector3 p3e, Vector3 p4e) {
    Scalar m1 = max(max(abs(p1e - p1).x, max(abs(p1e - p1).y, abs(p1e - p1).z)),
                    max(abs(p2e - p2).x, max(abs(p2e - p2).y, abs(p2e - p2).z)));
    Scalar m2 = max(max(abs(p3e - p3).x, max(abs(p3e - p3).y, abs(p3e - p3).z)),
                    max(abs(p4e - p4).x, max(abs(p4e - p4).y, abs(p4e - p4).z)));
    return max(m1, m2);
}

// --- Tolerance Computation ---

// Forward declarations
void compute_face_vertex_tolerance(device CCDData& data_in, constant CCDConfig& config);
void compute_edge_edge_tolerance(device CCDData& data_in, constant CCDConfig& config);
void get_numerical_error(device CCDData& data_in, bool use_ms, bool is_vf);

kernel void compute_tolerance_kernel(
    device CCDData* data [[buffer(0)]],
    constant CCDConfig& config [[buffer(1)]],
    constant bool& is_vf [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;

    if (is_vf) {
        compute_face_vertex_tolerance(data[id], config);
    } else {
        compute_edge_edge_tolerance(data[id], config);
    }

    data[id].nbr_checks = 0;
    get_numerical_error(data[id], config.use_ms, is_vf);
}

kernel void initialize_buffer_kernel(
    device CCDDomain* bufferData [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;
    bufferData[id].init(id);
}

kernel void ccd_kernel(
    device CCDDomain* bufferData [[buffer(0)]],
    device atomic_uint* bufferHead [[buffer(1)]],
    device atomic_uint* bufferTail [[buffer(2)]],
    device atomic_int* bufferOverflow [[buffer(3)]],
    constant uint& bufferCapacity [[buffer(4)]],
    constant uint& bufferStartingSize [[buffer(5)]],
    
    device CCDData* data [[buffer(6)]],
    device atomic_uint* toi [[buffer(7)]],
    constant CCDConfig& config [[buffer(8)]],
    constant bool& is_vf [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= bufferStartingSize) return;

    // Construct buffer helper
    CCDBuffer buffer;
    buffer.data = bufferData;
    buffer.head = bufferHead;
    buffer.tail = bufferTail;
    buffer.overflow_flag = bufferOverflow;
    buffer.capacity = bufferCapacity;
    buffer.starting_size = bufferStartingSize;

    // Monotonic access
    uint head = atomic_load_explicit(buffer.head, memory_order_relaxed);
    
    CCDDomain domain_in = buffer.data[(head + id) % buffer.capacity];
    
    device CCDData& data_in = data[domain_in.query_id];
    
    float current_toi = as_type<float>(atomic_load_explicit(toi, memory_order_relaxed));
    if (domain_in.tuv[0].lower >= current_toi) return;

    Scalar true_tol = 0;
    bool box_in = false;
    
    if (origin_in_inclusion_function(data_in, domain_in, true_tol, box_in, is_vf)) {
        Vector3 widths = Vector3(
            domain_in.tuv[0].upper - domain_in.tuv[0].lower,
            domain_in.tuv[1].upper - domain_in.tuv[1].lower,
            domain_in.tuv[2].upper - domain_in.tuv[2].lower
        );
        
        if (all(widths <= Vector3(data_in.tol[0], data_in.tol[1], data_in.tol[2]))) {
            atomic_min_float(toi, domain_in.tuv[0].lower);
            return;
        }
        
        if (box_in && (config.allow_zero_toi || domain_in.tuv[0].lower > 0)) {
            atomic_min_float(toi, domain_in.tuv[0].lower);
            return;
        }
        
        if (true_tol <= config.co_domain_tolerance && (config.allow_zero_toi || domain_in.tuv[0].lower > 0)) {
            atomic_min_float(toi, domain_in.tuv[0].lower);
            return;
        }
        
        int split = split_dimension(data_in, widths);
        bool sure_in = bisect(domain_in, split, toi, buffer, is_vf);
        
        if (sure_in) {
            atomic_min_float(toi, domain_in.tuv[0].lower);
            return;
        }
    }
}

kernel void shift_queue_start_kernel(
    device atomic_uint* bufferHead [[buffer(0)]],
    device atomic_uint* bufferTail [[buffer(1)]],
    device uint* bufferStartingSize [[buffer(2)]] // Pointer to update starting size
) {
    uint head = atomic_load_explicit(bufferHead, memory_order_relaxed);
    uint tail = atomic_load_explicit(bufferTail, memory_order_relaxed);
    uint startSize = *bufferStartingSize;
    
    // Advance head by starting_size
    head += startSize;
    atomic_store_explicit(bufferHead, head, memory_order_relaxed);
    
    // Update starting_size for next pass
    *bufferStartingSize = tail - head;
}
