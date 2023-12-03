kernel void multiple_transpose( global float* A, global float* B, int W, int H) {
   const int idx = get_global_id(0);
   
   const int real_idx = idx % (W*H);
   const int offset = idx - real_idx;

   int row = real_idx / W;
   int col = real_idx % W;
   B[offset + col*H + row] = A[idx];
}