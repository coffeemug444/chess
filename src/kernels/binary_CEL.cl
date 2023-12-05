kernel void binary_CEL(global float* A, global float* B, global float* C) {
    const int idx = get_global_id(0);

    const float true_v = A[idx];
    const float pred_v = B[idx];

    C[idx] = -(true_v*log(pred_v)) + (1-true_v)*log(1-pred_v);
} 
