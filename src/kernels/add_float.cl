kernel void add_float( global float* A, global float* B, float add) {
   const int idx = get_global_id(0);
   
   B[idx] = A[idx] + add;
}