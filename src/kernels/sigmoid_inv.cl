kernel void sigmoid_inv(global float* A, global float* B) {
    const int idx = get_global_id(0);
    const float neg_exp = exp(-A[idx]);
    B[idx] = neg_exp/((1+neg_exp)*(1+neg_exp));
} 
