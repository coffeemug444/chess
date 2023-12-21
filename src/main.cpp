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



   
   int channels = 2;
   int filters = 2;

   Mat output {5*5*filters,1,std::vector<float>{
      0,0,0,0,0,
      0,2,2,2,0,
      0,2,2,2,0,
      0,2,2,2,0,
      0,0,0,0,0,

      0,0,0,0,0,
      0,2,2,2,0,
      0,2,2,2,0,
      0,2,2,2,0,
      0,0,0,0,0,
   }};

   ConvKernel conv_kernel {channels,3,3,filters, SAME,5,5,Mat::ones(3*3*filters,1)};

   ParallelMat outputs {{output, output}};
   ParallelMat inputs = conv_kernel ^ outputs;

   for (Mat input : inputs.toVector())
   {
      auto vals = input.getVals();
      for (int channel = 0; channel < channels; channel++)
      {   
         for (int row = 0; row < 5; row++)
         {
            std::string delim = "";
            for (int col = 0; col < 5; col++)
            {
               std::cout << delim << vals[channel*5*5 + row*5 + col];
               delim = ", ";
            }
            std::cout << '\n';
         }
         std::cout << '\n';
      }
      std::cout << "---------------\n";
   }
}




