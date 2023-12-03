kernel void multiple_matmul( global float* A, global float* B, global float* C, int common, int B_w, int A_h) {
    const int idx = get_global_id(0);
    
    const int OUT_NUM_ELEM = A_h*B_w;
    const int B_NUM_ELEM = B_w*common;
    const int real_idx = idx % OUT_NUM_ELEM;
    const int CURRENT_MATRIX = idx / OUT_NUM_ELEM;
    const int B_offset = CURRENT_MATRIX*B_NUM_ELEM;

    
    const int row = real_idx / B_w;
    const int col = real_idx % B_w;

    float sum = 0;
    for (int w = 0; w < common; w++) {
        sum += A[row*common + w]*B[B_offset + B_w*w + col];
    }
    C[idx] = sum;
}