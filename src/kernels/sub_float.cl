kernel void sub_float( global float* A, global float* B, float sub) {
   const int idx = get_global_id(0);
   
   B[idx] = A[idx] - sub;
}