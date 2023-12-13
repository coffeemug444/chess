kernel void convolution( global float* CONVKERNEL, 
                         global float* INPUT, 
                         global float* OUTPUT, 
                         int convkernel_w, 
                         int convkernel_h,
                         int input_w,
                         int input_h,
                         int output_w,
                         int u_padding,
                         int l_padding)
{
    const int idx = get_global_id(0);

    int out_row = idx / output_w;
    int out_col = idx % output_w;
    
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

            int conv_idx = convkernel_w * conv_row + conv_col;
            int input_idx = input_row * input_w + input_col;

            total += CONVKERNEL[conv_idx] * INPUT[input_idx];
        }
    }

    OUTPUT[idx] = total;
}