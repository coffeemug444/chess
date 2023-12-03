kernel void dot_mat_eq( global float* A, global float* B) {
   const int idx = get_global_id(0);
   
   A[idx] = A[idx] * B[idx];
}