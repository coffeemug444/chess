kernel void mul_float_eq( global float* A, float mul) {
   const int idx = get_global_id(0);
   
   A[idx] = A[idx] * mul;
}