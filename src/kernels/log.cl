kernel void log(global float* A, global float* B) {
    const int idx = get_global_id(0);
    B[idx] = log(A[idx]);
} 
