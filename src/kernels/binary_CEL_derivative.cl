kernel void binary_CEL_derivative(global float* A, global float* B, global float* C) {
    const int idx = get_global_id(0);

    const float true_v = A[idx];
    const float pred_v = B[idx];

    C[idx] = -(true_v/pred_v - (1.f-true_v)/(1.f-pred_v));
} 
