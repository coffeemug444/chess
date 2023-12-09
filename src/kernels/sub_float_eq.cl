kernel void sub_float_eq( global float* A, float sub) {
   const int idx = get_global_id(0);
   
   A[idx] = A[idx] - sub;
}