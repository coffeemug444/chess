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

   Mat input {5*5*channels,1,std::vector<float>{
      0,0,0,0,0,
      0,0,0,0,0,
      0,0,1,0,0,
      0,0,0,0,0,
      0,0,0,0,0,

      0,0,0,0,0,
      0,0,0,0,0,
      0,0,1,0,0,
      0,0,0,0,0,
      0,0,0,0,0,
   }};

   ParallelMat inputs{{input, input}};

   ConvKernel conv_kernel {channels,3,3,filters, SAME,5,5,Mat::ones(3*3*filters,1)};

   ParallelMat outputs = conv_kernel * inputs;

   for (Mat output : outputs.toVector())
   {
      auto vals = output.getVals();
      for (int filter = 0; filter < filters; filter++)
      {   
         for (int row = 0; row < 5; row++)
         {
            std::string delim = "";
            for (int col = 0; col < 5; col++)
            {
               std::cout << delim << vals[filter*5*5 + row*5 + col];
               delim = ", ";
            }
            std::cout << '\n';
         }
         std::cout << '\n';
      }
      std::cout << "---------------\n";
   }

}




