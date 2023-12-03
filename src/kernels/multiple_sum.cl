kernel void multiple_sum( global float* input, global float* output, int numArrays, int arraySize) {
   const int idx = get_global_id(0);
   
   float sum = 0;
   for (int i = 0; i < numArrays; i += 1) {
      sum += input[idx + i*arraySize];
   }
   output[idx] = sum;
}