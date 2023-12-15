kernel void pad( global float* INPUT, 
                 global float* OUTPUT, 
                 int input_width,
                 int input_height,
                 int input_channels,
                 int lpad,
                 int rpad,
                 int upad,
                 int dpad) {

   const int idx = get_global_id(0);

   int output_width = input_width + lpad + rpad;
   int output_height = input_height + upad + dpad;

   int output_channel_elements = output_width*output_height;
   int input_channel_elements = input_width*input_height;
   int channel = idx / output_channel_elements;

   int row = (idx%output_channel_elements) / output_width - lpad;
   int col = (idx%output_channel_elements) % output_width - upad;

   bool out_of_bounds = (row < 0 || row >= input_height) || 
                        (col < 0 || col >= input_width);

   if (out_of_bounds)
   {
      OUTPUT[idx] = 0;
   }
   else
   {
      OUTPUT[idx] = INPUT[channel*input_channel_elements + row*input_width + col];
   }
}