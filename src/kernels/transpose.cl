kernel void transpose( global float* A, global float* B, int W, int H) {
   const int idx = get_global_id(0);
   
   int row = idx / W;
   int col = idx % W;
   B[col*H + row] = A[idx];
}