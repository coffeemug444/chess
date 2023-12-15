kernel void convolution( global float* CONVKERNEL, 
                         global float* INPUT, 
                         global float* OUTPUT, 
                         int convkernel_w, 
                         int convkernel_h,
                         int input_w,
                         int input_h,
                         int channels,
                         int filters,
                         int output_w,
                         int output_h)
{
    const int idx = get_global_id(0);

    int kernel_elements = convkernel_w*convkernel_h;
    int output_elements = output_w*output_h;
    int channel_elements = input_w*input_h;

    int filter = idx / output_elements;

    int out_row = (idx % output_elements) / output_h;
    int out_col = (idx % output_elements) % output_w;
    
    float total = 0;

    for (int conv_row = 0; conv_row < convkernel_h; conv_row++)
    {
        int input_row = out_row + conv_row;
        for (int conv_col = 0; conv_col < convkernel_w; conv_col++)
        {
            int input_col = out_col + conv_col;

            for (int channel = 0; channel < channels; channel++)
            {
                float kernel_val = CONVKERNEL[filter*kernel_elements + conv_row*convkernel_w + conv_col];
                float input_val = INPUT[channel*channel_elements + input_row*input_w + input_col];
                total += kernel_val*input_val;
            }

        }
    }

    OUTPUT[idx] = total;
}