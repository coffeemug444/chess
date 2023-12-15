kernel void rotate_conv( global float* INPUT, 
                         global float* OUTPUT, 
                         int convkernel_w, 
                         int convkernel_h)
{
    const int idx = get_global_id(0);

    int filter = idx / (convkernel_w*convkernel_h);
    int input_row = (idx % (convkernel_w*convkernel_h)) / convkernel_w;
    int input_col = (idx % (convkernel_w*convkernel_h)) % convkernel_w;

    int output_row = (convkernel_h-1) - input_row;
    int output_col = (convkernel_w-1) - input_col;

    OUTPUT[(filter*convkernel_w*convkernel_h) + output_row*convkernel_w + output_col] = INPUT[idx];
}