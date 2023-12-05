kernel void sigmoid(global float* A, global float* B) {
    const int idx = get_global_id(0);
    B[idx] = 1.f/(1.f+exp(-A[idx]));
} 
