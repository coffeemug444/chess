kernel void add_float_eq( global float* A, float add) {
   const int idx = get_global_id(0);
   
   A[idx] = A[idx] + add;
}