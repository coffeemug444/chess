#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <ranges>
#include <thread>
#include <random>

#include "piece.hpp"
#include "board.hpp"
#include "nnet.hpp"
#include "layerFullyConnected.hpp"
#include "layerSoftmax.hpp"
#include "convKernel.hpp"

#include <iostream>


int main() 
{
   unsigned channels = 2;
   unsigned filters = 1;

   unsigned input_h = 5;
   unsigned input_w = 5;
   unsigned kernel_h = 3;
   unsigned kernel_w = 3;

   auto input = Mat::ones(input_h*input_w*channels,1);
   auto kernel = ConvKernel{channels, 
                            kernel_h,
                            kernel_w,
                            filters,
                            SAME,
                            input_h,
                            input_w,
                            Mat::ones(kernel_h*kernel_w, 1)};

   auto [padded_h, padded_w] = kernel.getPaddedHeightWidth();

   auto padded = Mat(padded_h, padded_w, 
      kernel.pad(input.getbuffer())
   );

   auto padded_vals = padded.getVals();

   int i = 0;
   for (unsigned channel = 0; channel < channels; channel++)
   {   
      for (unsigned row = 0; row < padded_h; row++)
      {
         std::string delim = "";
         for (unsigned col = 0; col < padded_w; col++)
         {
            std::cout << delim << padded_vals[i];
            delim = ", ";
            i++;
         }
         std::cout << '\n';
      }
      std::cout << '\n';
      std::cout << '\n';
   }
}




