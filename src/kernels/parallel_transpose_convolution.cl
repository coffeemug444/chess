kernel void parallel_transpose_convolution( global float* CONVKERNEL, 
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
    int input_elements = input_w*input_h;

    int total_output_elements = output_elements*filters;
    int total_input_elements = input_elements*channels;

    int input = idx/total_input_elements;

    int input_row = ((idx%total_input_elements) % input_elements) / input_w;
    int input_col = ((idx%total_input_elements) % input_elements) % input_w;
    
    float total = 0;

    for (int conv_row = 0; conv_row < convkernel_h; conv_row++)
    {
        int output_row = input_row + conv_row;
        for (int conv_col = 0; conv_col < convkernel_w; conv_col++)
        {
            int output_col = input_col + conv_col;

            for (int filter = 0; filter < filters; filter++)
            {
                float kernel_val = CONVKERNEL[filter*kernel_elements + conv_row*convkernel_w + conv_col];
                float output_val = OUTPUT[input*total_input_elements + filter*input_elements + output_row*input_w + output_col];
                // total += kernel_val*output_val;
                total += kernel_val;
            }

        }
    }

    INPUT[idx] = total / channels;
}