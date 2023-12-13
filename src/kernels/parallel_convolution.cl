kernel void parallel_convolution( global float* CONVKERNEL, 
                                 global float* INPUT, 
                                 global float* OUTPUT, 
                                 int convkernel_w, 
                                 int convkernel_h,
                                 int input_w,
                                 int input_h,
                                 int channels,
                                 int filters,
                                 int output_w,
                                 int output_h,
                                 int u_padding,
                                 int l_padding)
{
    const int idx = get_global_id(0);

    int kernel_elements = convkernel_w*convkernel_h;
    int output_elements = output_w*output_h*filters;
    int input_elements = input_w*output_h*channels;

    int input_num = idx / output_elements;

    int filter = idx / (output_w*output_h);

    int out_row = (idx % (output_w*output_h)) / output_h;
    int out_col = (idx % (output_w*output_h)) % output_w;
    
    float total = 0;

    for (int conv_row = 0; conv_row < convkernel_h; conv_row++)
    {
        int input_row = out_row - u_padding + conv_row;

        // if this kernel position is going to be outside the input
        // space then we don't add anything to the total
        if (input_row < 0) continue;
        if (input_row >= input_h) continue;

        for (int conv_col = 0; conv_col < convkernel_w; conv_col++)
        {
            int input_col = out_col - l_padding + conv_col;

            if (input_col < 0) continue;
            if (input_col >= input_w) continue;

            for (int channel = 0; channel < channels; channel++)
            {
                int conv_idx = (kernel_elements * filter) + (convkernel_w * convkernel_h * channel) + (convkernel_w * conv_row) + conv_col;
                int input_idx = (input_elements * input_num) + (input_w * input_h * channel) + (input_row * input_w) + input_col;
                total += CONVKERNEL[conv_idx] * INPUT[input_idx];
            }
        }
    }

    OUTPUT[idx] = total;
}