kernel void multiple_multi_matmul( global float* A, global float* B, global float* C, int common, int B_w, int A_h) {
    const int idx = get_global_id(0);
    
    const int OUT_NUM_ELEM = A_h*B_w;
    const int B_NUM_ELEM = B_w*common;
    const int A_NUM_ELEM = common*A_h;
    const int real_idx = idx % OUT_NUM_ELEM;
    const int CURRENT_MATRIX = idx / OUT_NUM_ELEM;
    const int B_offset = CURRENT_MATRIX*B_NUM_ELEM;
    const int A_offset = CURRENT_MATRIX*A_NUM_ELEM;

    
    const int row = real_idx / B_w;
    const int col = real_idx % B_w;

    float sum = 0;
    for (int w = 0; w < common; w++) {
        sum += A[A_offset + row*common + w]*B[B_offset + B_w*w + col];
    }
    C[idx] = sum;
}