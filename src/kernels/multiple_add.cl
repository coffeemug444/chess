kernel void multiple_add( global float* A, global float* B, global float* out, int B_size) {
   const int idx = get_global_id(0);
   
   int real_idx = idx % B_size;
   
   out[idx] = A[real_idx] + B[idx];
}